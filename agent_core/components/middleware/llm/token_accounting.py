from __future__ import annotations

import logging
from typing import Any, cast

from agent_core.components.middleware.llm.base import (
    LLMCallContext,
    LLMMiddleware,
)
from agent_core.execution_context import (
    get_current_execution_scope,
)
from agent_core.llm import LLMResponse
from agent_core.protocols import CostPersister, CostSink, EventSink

logger = logging.getLogger(__name__)


class TokenAccountingMiddleware(LLMMiddleware):
    """Tracks cumulative token usage per task and charges BudgetState.

    After each LLM call, extracts input/output token counts from response
    metadata, accumulates them, charges the BudgetState, and optionally
    emits SSE events for frontend display.

    Cost accounting flows through the injected ``CostSink``; callers that want
    zero accounting pass ``cost_sink=None`` (or simply omit the kwarg). A host
    typically injects a persistent tracker in its server runtime and an
    in-memory or no-op one in its stateless SDK path.

    Durable persistence (``persist_cost``) is opt-in through the injected
    ``CostPersister``; when ``None`` -- the default -- the method is a no-op.
    Both seams are Protocols so that neither the cost schema nor the database
    session handling has to be known here.
    """

    def __init__(
        self,
        event_store: EventSink | None = None,
        *,
        cost_sink: CostSink | None = None,
        cost_persister: CostPersister | None = None,
        scene: str = "",
        usage_aggregator: Any = None,
    ) -> None:
        raw_cost_sink: object = cost_sink
        if (
            cost_persister is not None
            and cost_sink is not None
            and not isinstance(raw_cost_sink, CostSink)
        ):
            raise TypeError(
                "cost_sink must implement CostSink.record and CostSink.get_summary "
                "when cost_persister is configured"
            )
        self._event_store = event_store
        # Per-task cumulative counters: task_id → {input, output, total, llm_calls}
        self._usage: dict[str, dict[str, int]] = {}
        # Per-task model tracking for cost estimation. ``cost_sink`` is
        # injected by the composition root (bootstrap_runtime / SDK)
        # rather than self-resolved through the registry; this keeps
        # ``components/middleware/`` from importing ``state/`` directly
        # (Phase 6 layering invariant).
        self._cost_tracker: CostSink | None = cost_sink
        # Where the final summary lands. A Protocol rather than a database
        # session factory: the schema, the column names and the transaction
        # boundary are host concerns, and holding a session factory here is
        # what previously kept this class in the product.
        self._cost_persister: CostPersister | None = cost_persister
        self._primary_model: dict[str, str] = {}  # task_id → model name
        # Heavy-mode scene tag (e.g. "main_llm" / "dag_model" /
        # "outline_llm" / "report_llm"). Empty → no scene wiring;
        # ``usage_aggregator`` then never sees scene either, matching
        # the existing single-bucket flow.
        self._scene = scene
        # Optional sink for the SDK ``UsageAggregator`` so the same LLM
        # call can flow into both the per-task cost tracker (this class)
        # and the protocol-level final.usage. Duck-typed to avoid an
        # sdk_cli → components import.
        self._usage_aggregator = usage_aggregator

    @property
    def name(self) -> str:
        return "token_accounting"

    def get_usage(self, task_id: str) -> dict[str, int]:
        """Get cumulative usage for a task. Returns copy."""
        return dict(self._usage.get(task_id, {"input": 0, "output": 0, "total": 0, "llm_calls": 0}))

    def _extract_usage(self, response: LLMResponse) -> tuple[int, int, int, int]:
        """Extract (input, output, cache_read, cache_creation) token counts.

        Native :class:`LLMResponse` carries a single flat ``usage`` dict in
        OpenAI-wire shape regardless of provider — the infra clients
        normalise both OpenAI (``prompt_tokens`` / ``completion_tokens`` /
        ``prompt_tokens_details.cached_tokens``) and Anthropic
        (``input_tokens`` / ``output_tokens`` / ``cache_read_input_tokens``)
        into ``{prompt_tokens, completion_tokens, total_tokens,
        cached_tokens}``. Cache-creation tokens are not surfaced by the
        native clients, so that count is always 0.
        """
        raw_usage: object = getattr(response, "usage", None)
        if not isinstance(raw_usage, dict):
            return 0, 0, 0, 0
        usage = cast("dict[str, Any]", raw_usage)

        inp = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
        out = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
        cache_read = (
            usage.get("cached_tokens")
            or usage.get("cache_read_input_tokens")
            or 0
        )
        cache_create = usage.get("cache_creation_input_tokens", 0) or 0
        return int(inp), int(out), int(cache_read), int(cache_create)

    async def after_llm(
        self, ctx: LLMCallContext, response: LLMResponse
    ) -> LLMResponse:
        input_tokens, output_tokens, cache_read, cache_create = self._extract_usage(response)
        total = input_tokens + output_tokens

        if total == 0:
            return response

        task_id = ctx.task_id or "unknown"

        # Accumulate
        if task_id not in self._usage:
            self._usage[task_id] = {"input": 0, "output": 0, "total": 0, "llm_calls": 0}
        acc = self._usage[task_id]
        acc["input"] += input_tokens
        acc["output"] += output_tokens
        acc["total"] += total
        acc["llm_calls"] += 1

        # Cost tracking
        raw_rm: object = getattr(response, "response_metadata", None)
        rm: dict[str, Any] = (
            cast("dict[str, Any]", raw_rm) if isinstance(raw_rm, dict) else {}
        )
        # Prefer the provider-reported model carried on ``LLMResponse.model``;
        # fall back to anything stamped in response_metadata, then to the
        # model id we stashed in ctx.metadata (set by LLMProxy._make_ctx).
        # Gateways like api.miromind.site often return an empty model.
        model_name = (
            getattr(response, "model", "")
            or rm.get("model_name")
            or rm.get("model")
            or ctx.metadata.get("model_id", "")
        )
        # Vendor label stamped by a fallback chain. Empty when the LLM is
        # constructed without a chain wrapper — caller can still bucket
        # by model alone.
        provider = str(rm.get("provider_actually_used") or "")
        if model_name and task_id != "unknown" and self._cost_tracker is not None:
            try:
                self._cost_tracker.record(
                    task_id, model_name, input_tokens, output_tokens,
                )
                self._primary_model.setdefault(task_id, model_name)
            except Exception:
                pass

        # Mirror this call into the SDK UsageAggregator (if injected) so
        # heavy-mode aux LLMs (dag_model / outline_llm / report_llm) feed
        # the same final.usage as the main agent. Duck-typed call;
        # silently no-op on any signature mismatch.
        #
        # Cache fields use the 2026-05-28 split: pass both
        # ``cache_read_tokens`` (was ``cached_tokens`` semantically) and
        # ``cache_write_tokens`` (was ``cache_creation_tokens``).
        # Older aggregator builds that only accept the legacy
        # ``cached_tokens`` kwarg are handled via the cascading TypeError
        # fallback below.
        if self._usage_aggregator is not None and model_name:
            try:
                self._usage_aggregator.record_llm_call(
                    provider=provider,
                    model=model_name,
                    prompt_tokens=int(input_tokens),
                    completion_tokens=int(output_tokens),
                    cache_read_tokens=int(cache_read or 0),
                    cache_write_tokens=int(cache_create or 0),
                    scene=self._scene,
                )
            except TypeError:
                # Older aggregator API — try legacy kwarg only.
                try:
                    self._usage_aggregator.record_llm_call(
                        provider=provider,
                        model=model_name,
                        prompt_tokens=int(input_tokens),
                        completion_tokens=int(output_tokens),
                        cached_tokens=int(cache_read or 0),
                        scene=self._scene,
                    )
                except TypeError:
                    # Even older — no provider / scene support.
                    # ``contextlib.suppress`` would read worse threaded into the
                    # middle of this cascade, which is being carried across
                    # unchanged on purpose — collapsing it is a behavior change
                    # that deserves its own review.
                    try:  # noqa: SIM105
                        self._usage_aggregator.record_llm_call(
                            model=model_name,
                            prompt_tokens=int(input_tokens),
                            completion_tokens=int(output_tokens),
                            cached_tokens=int(cache_read or 0),
                        )
                    except Exception:
                        pass
                except Exception:
                    pass
            except Exception:
                pass

        # Store in context metadata for downstream consumers
        ctx.metadata["token_usage"] = {
            "this_call": {
                "input": input_tokens,
                "output": output_tokens,
                "total": total,
                "cache_read": cache_read,
                "cache_creation": cache_create,
            },
            "cumulative": dict(acc),
            "model": model_name,
        }

        # Charge BudgetState if available in execution scope
        try:
            scope = get_current_execution_scope()
            if scope and "budget_state" in scope.metadata:
                from agent_core.models.task_budget import BudgetCharge
                budget_state = scope.metadata["budget_state"]
                budget_state.charge(BudgetCharge(
                    primitive="llm_call",
                    llm_calls=1,
                    tokens=total,
                ))
        except Exception:
            pass  # Budget charging is best-effort

        # Emit SSE event for frontend
        if self._event_store and task_id != "unknown":
            try:
                from agent_core.events import EventType
                await self._event_store.append(
                    task_id=task_id,
                    event_type=EventType.AGENT_ACTION,
                    payload={
                        "trace_type": "token_usage",
                        "this_call": {
                            "input": input_tokens,
                            "output": output_tokens,
                            "cache_read": cache_read,
                            "cache_creation": cache_create,
                        },
                        "cumulative": dict(acc),
                        "model": model_name,
                    },
                    agent_role="system",
                )
            except Exception:
                pass  # SSE emission is best-effort

        logger.debug(
            "TokenAccounting task=%s: +%d tokens (cumulative: %d input, %d output, %d total)",
            task_id, total, acc["input"], acc["output"], acc["total"],
        )
        return response

    async def persist_cost(self, task_id: str) -> None:
        """Hand the task's cumulative cost summary to the host. Call on completion.

        No-op when ``cost_sink`` or ``cost_persister`` is missing — the SDK /
        stateless runtime path has nothing to persist into, and that is the
        default rather than an error.

        Swallows persister failures on purpose: accounting is observability, and
        a database that is down must not fail the task whose cost it describes.
        The traceback is kept at debug level.
        """
        if self._cost_tracker is None or self._cost_persister is None:
            return
        try:
            await self._cost_persister.persist(
                task_id,
                self._cost_tracker.get_summary(task_id),
                self._primary_model.get(task_id, ""),
            )
        except Exception:
            logger.debug("Failed to persist cost for task %s", task_id, exc_info=True)

    def reset(self, task_id: str) -> None:
        """Clear counters for a task (e.g., after task completion)."""
        self._usage.pop(task_id, None)
        self._primary_model.pop(task_id, None)
        reset = getattr(self._cost_tracker, "reset", None)
        if reset is not None:
            reset(task_id)
