"""Loop type contracts for the agent-loop engine."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal, Protocol, TypedDict, cast, runtime_checkable

from agent_core.messages import Message

logger = logging.getLogger(__name__)

# Absolute monotonic soft deadline stored in execution-scope metadata.
WALL_DEADLINE_MONOTONIC_KEY = "wall_deadline_monotonic"


class UsageMetadataExtras(TypedDict, total=False):
    """Host-specific usage aliases and provenance.

    ``extract_usage`` never emits these; they exist so a host that stamps
    its own aliases onto a usage mapping can still describe the result as
    a :class:`UsageMetadata`. ApodexHarness' budget observer reads the
    ``input_tokens`` / ``output_tokens`` aliases, and its native clients
    carry ``total_tokens`` on ``LLMResponse.usage``.

    Optional keys are NOT directly indexable under a type checker
    (``reportTypedDictNotRequiredAccess``) — read them with ``.get(key, 0)``.
    """

    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated: bool


class UsageMetadata(UsageMetadataExtras):
    """What :func:`agent_core.runtime.loop.extract_usage` returns.

    The normalized fields below are required: every non-``None`` return
    from ``extract_usage`` carries all of them, zero-filled when the
    provider reported nothing, so a consumer can index them without
    probing which response shape it was handed.
    ``tests/test_llm_runtime_extract_usage.py`` pins that at runtime
    against ``__required_keys__``.

    This is a *producer* type, deliberately narrower than the
    ``usage`` fields on :class:`TurnContext` / :class:`LLMAttemptContext`.
    Those stay ``Mapping[str, Any] | None`` because hosts build usage
    mappings of their own — partial dicts in test doubles, alias-only
    shapes, ``dict(event["usage"])`` re-wraps out of a ``dict[str, Any]``
    attempt event — and neither ``dict[str, int]`` nor ``dict[str, Any]``
    is assignable to a TypedDict. A consumer that wants the precise shape
    annotates its own parameter as ``UsageMetadata``; the loop does not
    force it on producers. Extra keys beyond those declared here are
    likewise a host's business: a TypedDict cannot express "open" on
    Python 3.12 (PEP 728 lands later), which is the other reason the
    dataclass fields stay a plain mapping.
    """

    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cached_tokens: int
    cache_creation_tokens: int
    reasoning_tokens: int


def deadline_remaining_s(metadata: Mapping[str, Any] | None) -> float | None:
    """Return seconds to a structural lease or absolute soft deadline.

    Execution-context storage belongs to each host product. Passing only its
    metadata keeps this frozen contract module independent of either host.

    ``None`` means no deadline is stamped (no wall observer — plain swarm,
    tests, HTTP API): callers keep their configured timeouts unchanged. The
    result may be negative once the deadline has passed.

    Two stamp shapes are accepted, and the check is **structural, not
    nominal**:

    * any object exposing ``remaining_s() -> float`` — a renewable lease view.
      The concrete renewable-lease implementation is a product concern and
      must stay out of this module: importing it here to run an ``isinstance``
      check would put a product-side dependency inside the frozen contract
      module and break the ``agent_core`` import closure. Do not "clean this
      up" into an ``isinstance`` check.
    * a bare ``int``/``float`` absolute ``time.monotonic()`` instant.

    Anything else yields ``None``. ``int``/``float`` have no ``remaining_s``,
    so the ordering below is unambiguous.
    """
    deadline = (metadata or {}).get(WALL_DEADLINE_MONOTONIC_KEY)
    remaining_s = cast(Callable[[], Any] | None, getattr(deadline, "remaining_s", None))
    if callable(remaining_s):
        try:
            return float(remaining_s())
        except Exception:
            return None
    # ``bool`` is an ``int`` subclass, but it is never a meaningful monotonic
    # instant. Treat it as a bad metadata shape instead of expiring the lease at
    # process time 0 or 1.
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)):
        return None
    return float(deadline) - time.monotonic()


def wall_deadline_remaining_s() -> float | None:
    """Return the active execution scope's remaining wall-time budget."""
    from agent_core.execution_context import get_current_execution_scope

    scope = get_current_execution_scope()
    return deadline_remaining_s(scope.metadata if scope is not None else None)


@dataclass(frozen=True)
class LoopPolicy:
    """Workflow-specific behavior injected into the generic loop."""

    phase_id: str = ""
    no_tool_behavior: Literal["stop", "nudge"] = "nudge"
    no_tool_nudge_message: str = ""
    terminal_tool_names: tuple[str, ...] = ()
    # Tool calls recovered from text on a tool-free landing turn are denied by
    # default. Workflows may explicitly allow bounded, end-of-run actions such
    # as ``collect_reports`` or ``write_file``. ``None`` preserves the legacy
    # terminal-tool allowlist; an empty tuple deliberately allows no tools.
    #
    # CONTRACT-ONLY IN THIS REPO (loop-v1 freeze, 2026-08-27): the landing-turn
    # recovery path that reads this lives in the other consumer. Frozen here so
    # both repos type-check against one ``LoopPolicy``; leaving it ``None``
    # keeps this repo's behavior unchanged.
    landing_tool_names: tuple[str, ...] | None = None


@dataclass
class LoopConfig:
    max_turns: int = 50
    max_tool_calls_per_turn: int = 5
    tool_timeout: int = 120
    llm_timeout: int = 180
    # First streamed chunk timeout; None defers to environment configuration.
    first_chunk_timeout: float | None = None
    # Abort reasoning-only streams after either enabled bound.
    reasoning_only_timeout_s: float | None = None
    reasoning_only_max_tokens: int | None = None
    # Total budget across admission, attempts, backoff, and recovery.
    logical_call_timeout_s: float | None = None
    context_token_limit: int = 120_000
    compact_after_turns: int = 12
    keep_recent: int = 16
    no_tool_max_retries: int = 2
    # Continuations offered to a reply the output cap cut off mid-sentence.
    # Separate from ``no_tool_max_retries`` because the two are opposite signals:
    # a tool-less turn is the model choosing to stop, a truncated one is the
    # model being stopped, so a truncation must not spend the nudge budget.
    truncation_max_continuations: int = 2
    max_llm_retries: int = 5
    # Fixed retry delay; None uses exponential backoff.
    retry_wait_fixed: int | None = None
    task_id: str = ""
    # Optional gateway affinity key; task_id remains the runtime scope.
    llm_session_id: str = field(default="", kw_only=True)
    role_id: str = ""
    loop_policy: LoopPolicy = field(default_factory=LoopPolicy)
    # ToolMessage character cap; None preserves full output.
    tool_result_max_chars: int | None = None
    # Any avoids importing runtime compaction interfaces into this type layer.
    compactor: Any = None
    compaction_policy: Any = None
    tool_result_post_processor: Any = None

    # Stop before tool output makes the next LLM plus summary request overflow.
    context_overflow_guard: bool = False
    max_context_length: int = 262_144
    max_completion_tokens: int = 32_768
    summary_prompt: str = ""

    # Per-call reminder added to a copy of history, never persisted.
    system_addendum_per_call: str | Callable[[], str] = ""
    system_addendum_min_turn: int = 0
    system_addendum_per_call_role: str = "system"


@dataclass
class TurnContext:
    turn: int
    max_turns: int
    task_id: str
    role_id: str
    ai_text: str
    thinking: str
    tool_calls: list[dict[str, Any]]
    messages: list[Message]
    # Read-only mapping, not ``UsageMetadata``, on purpose: producers are
    # hosts, and a TypedDict rejects the ``dict[str, int]`` / ``dict[str, Any]``
    # shapes they build. ``UsageMetadata`` documents what ``extract_usage``
    # puts here; annotate a consumer's own parameter with it for precision.
    usage: Mapping[str, Any] | None
    metadata: dict[str, Any]
    # Reasoning recovered from tags leaked into visible content.
    leaked_reasoning: str = ""
    # Native content blocks retained for signed/encrypted replay.
    thinking_blocks: list[Any] = field(default_factory=list[Any])
    # Calls parsed on a tool-schema-free landing turn but denied by the
    # workflow's landing allowlist. Keeping these separate from ``tool_calls``
    # lets observers distinguish "the model answered in plain text" from "the
    # runtime blocked an attempted action" instead of inferring from an empty
    # list and accidentally finalising or retrying the leaked call.
    #
    # CONTRACT-ONLY IN THIS REPO (loop-v1 freeze, 2026-08-27): no producer here
    # yet, so this stays empty and ``tool_schemas_stripped`` stays ``False``.
    # An observer may read them today; it will simply always see the defaults.
    blocked_tool_calls: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    # Whether tool schemas were withheld from the request that produced this
    # turn — the condition under which a leaked call can appear in
    # ``blocked_tool_calls``.
    tool_schemas_stripped: bool = False


@dataclass
class LLMDeltaContext:
    turn: int
    max_turns: int
    task_id: str
    role_id: str
    delta: str
    accumulated_text: str
    delta_index: int
    metadata: dict[str, Any]
    # Provider-native reasoning, kept separate from visible content.
    thinking_delta: str = ""
    # Partial JSON args keyed by call id, or index before an id arrives.
    tool_call_args_chunks: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    # Identifies deltas from attempts that may later be discarded.
    attempt_id: str = ""
    attempt_index: int = 1
    call_id: str = ""


# Attempt outcome describes delivery; health details live in reason fields.
ATTEMPT_ACCEPTED = "accepted"
ATTEMPT_ACCEPTED_DEGRADED = "accepted_degraded"
ATTEMPT_DISCARDED = "discarded"
ATTEMPT_FAILED = "failed"

# Both outcomes deliver bytes to the loop and must retain streamed state.
DELIVERED_ATTEMPT_OUTCOMES = frozenset(
    {
        ATTEMPT_ACCEPTED,
        ATTEMPT_ACCEPTED_DEGRADED,
    }
)


@dataclass
class LLMAttemptContext:
    """Summary-only lifecycle snapshot for one provider attempt."""

    turn: int
    max_turns: int
    task_id: str
    role_id: str
    call_id: str
    attempt_id: str
    attempt_index: int
    phase: str
    outcome: str = ""
    reason: str = ""
    recovery_action: str = ""
    duration_ms: int = 0
    ttft_ms: int | None = None
    # See ``TurnContext.usage`` for why this is a plain mapping.
    usage: Mapping[str, Any] | None = None
    finish_reason: str = ""
    visible_chars: int = 0
    reasoning_chars: int = 0
    tool_calls_count: int = 0
    max_tokens: int | None = None
    # Sampling actually used for this attempt. Distinct from the provider-level
    # ``anthropic_thinking_budget`` / ``thinking_budget_tokens`` config in
    # ``infra/``: those say what was *requested* for the run, these report what
    # this one attempt ran with — which a runaway ladder or a reduced-cap retry
    # changes mid-call.
    #
    # CONTRACT-ONLY IN THIS REPO (loop-v1 freeze, 2026-08-27): the attempt
    # emitter here does not populate them yet, so they stay at their defaults.
    thinking_mode: str = "profile_default"
    thinking_budget: int | None = None
    error_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass
class ToolResult:
    name: str
    args: dict[str, Any]
    result: str
    duration_ms: int
    tool_call_id: str
    is_error: bool
    # Interrupted results remain in history to preserve tool-call pairing.
    interrupted: bool = False
    # Stable machine-readable failure category supplied by the execution
    # engine or a host classifier. Empty for successful legacy tools.
    error_kind: str = ""
    # Opaque host-owned handle for a result body shed from model context.
    result_id: str = ""
    # Host-provided repeated-invocation metadata. Execution is never skipped.
    repeat_count: int = 1
    repeat_recovery_id: str = ""


@dataclass
class ContextCompactionContext:
    """Pre/post snapshot for any loop operation that discards history."""

    turn: int
    task_id: str
    role_id: str
    reason: str
    compactor: str
    policy: str
    messages_before: list[Message]
    messages_after: list[Message]
    tokens_before: int
    metadata: dict[str, Any]


@dataclass
class Intervention:
    inject_messages: list[str] | None = None
    stop_reason: str | None = None
    skip_tool_execution: bool = False
    # Applied after message injection and before continuing the turn.
    pop_last_message: bool = False
    continue_to_next_turn: bool = False


@dataclass
class ToolCallIntervention:
    """Tool-call rewrite, short-circuit result, and metadata updates."""

    rewrite_args: dict[str, Any] | None = None
    skip_with_result: str | None = None
    metadata_updates: dict[str, Any] | None = None


@dataclass
class AgentLoopResult:
    messages: list[Message]
    final_content: str = ""
    turns_used: int = 0
    tool_calls_count: int = 0
    stopped_by: str = ""
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass
class CompactionEvent:
    """What one compaction did, for the durable record.

    Compaction is the one history rewrite that leaves no trace: it replaces
    messages in place, so a trajectory reading only the post-compaction history
    shows the rollup with nothing to compare it against, and the replaced turns
    are simply gone. ``selected`` and the token pair say how much was freed;
    ``summary`` is the only field that says what survived.

    Three outcomes have to stay distinguishable, because two of them produce an
    empty ``summary``:

    * a tier that does not summarise at all (Tier 1 blanking, the
      ``tool_compression_*`` fallbacks) — ``summary`` and ``rollback_reason``
      both empty;
    * a summariser that ran and produced text — ``summary`` set;
    * a summariser that ran and **failed**, whose deterministic slice can still
      win — ``summary`` empty but ``rollback_reason`` set.

    Without the third, a failed summariser is indistinguishable from one that
    never ran, which is precisely the confusion this record exists to remove.
    """

    turn: int
    seq: int
    selected: str
    tokens_before: int
    tokens_after: int
    relief_met: bool
    spill_refs: int
    #: Number of summariser calls made by the selected tier. Zero means the
    #: selected compaction path did not run the summariser.
    attempts: int = 0
    summary: str = ""
    #: Why the summariser rolled back (``llm_error`` /
    #: ``llm_error_permanent`` / ``empty_summary``), or empty when it did not
    #: run or did not fail.
    rollback_reason: str = ""


@runtime_checkable
class LoopObserver(Protocol):
    """Required structural contract implemented by agent-loop observers.

    Compaction and cancellation notifications are optional so legacy observers
    remain valid. Hosts can narrow those hooks with :class:`CompactionObserver`
    and :class:`CancellationObserver`; :class:`BaseObserver` implements both.
    """

    critical: bool

    async def on_loop_start(self, config: LoopConfig) -> None: ...

    async def on_llm_delta(self, ctx: LLMDeltaContext) -> Intervention | None: ...

    async def on_llm_attempt(
        self,
        ctx: LLMAttemptContext,
    ) -> Intervention | None: ...

    async def on_llm_response(self, ctx: TurnContext) -> Intervention | None: ...

    async def on_tool_call(
        self,
        ctx: TurnContext,
        tool_call: dict[str, Any],
    ) -> ToolCallIntervention | None: ...

    async def on_tool_result(
        self,
        ctx: TurnContext,
        result: ToolResult,
    ) -> ToolResult | None: ...

    async def on_turn_end(self, ctx: TurnContext) -> Intervention | None: ...

    async def on_loop_end(self, result: AgentLoopResult) -> None: ...


@runtime_checkable
class CompactionObserver(Protocol):
    """Optional observer hook for completed history compactions."""

    async def on_compaction(self, event: CompactionEvent) -> None: ...


@runtime_checkable
class CancellationObserver(Protocol):
    """Optional observer hook for cancellation cleanup."""

    async def on_loop_cancelled(self) -> None: ...


class BaseObserver:
    """No-op observer base; override only required hooks."""

    critical: bool = False

    async def on_loop_start(self, config: LoopConfig) -> None:
        pass

    async def on_llm_delta(self, ctx: LLMDeltaContext) -> Intervention | None:
        return None

    async def on_llm_attempt(
        self,
        ctx: LLMAttemptContext,
    ) -> Intervention | None:
        return None

    async def on_llm_response(self, ctx: TurnContext) -> Intervention | None:
        return None

    async def on_tool_call(
        self,
        ctx: TurnContext,
        tool_call: dict[str, Any],
    ) -> ToolCallIntervention | None:
        return None

    async def on_tool_result(
        self,
        ctx: TurnContext,
        result: ToolResult,
    ) -> ToolResult | None:
        return None

    async def on_turn_end(self, ctx: TurnContext) -> Intervention | None:
        return None

    async def on_compaction(self, event: CompactionEvent) -> None:
        """History was rewritten. Passive: compaction has already happened by
        the time this runs, so there is no intervention to return."""

    async def on_context_compacted(self, ctx: ContextCompactionContext) -> None:
        """History was discarded while its pre-change form was still reachable."""

    async def on_loop_end(self, result: AgentLoopResult) -> None:
        pass

    async def on_loop_cancelled(self) -> None:
        """Release resources when cancellation bypasses ``on_loop_end``."""


# Prevent GC of fire-and-forget observer tasks without coupling independent
# loops. Agent loops execute their dispatches from one owning asyncio task, so
# that task is the natural lifecycle boundary for the passive hooks it starts.
_background_tasks_by_owner: dict[asyncio.Task[Any], set[asyncio.Task[None]]] = {}


# Log each observer-hook failure once at warning level.
_warned_observer_errors: set[tuple[str, str]] = set()


def _handle_observer_error(
    observer: Any,
    method: str,
    exc: BaseException,
) -> None:
    """Log an observer crash without propagating it into the loop."""
    obs_class = type(observer).__name__
    key = (obs_class, method)
    if key in _warned_observer_errors:
        logger.debug(
            "Observer %s.%s raised (suppressed)",
            obs_class,
            method,
            exc_info=True,
        )
        return
    _warned_observer_errors.add(key)
    logger.warning(
        "Observer %s.%s raised: %s — subsequent failures DEBUG only",
        obs_class,
        method,
        exc,
        exc_info=True,
    )


def _discard_background_task(
    owner: asyncio.Task[Any],
    completed: asyncio.Task[None],
) -> None:
    """Release a completed passive hook and its empty owner bucket."""
    tasks = _background_tasks_by_owner.get(owner)
    if tasks is None:
        return
    tasks.discard(completed)
    if not tasks:
        _background_tasks_by_owner.pop(owner, None)


async def notify_observers(
    observers: list[Any],
    method: str,
    *args: Any,
    **kwargs: Any,
) -> list[Intervention]:
    """Run hooks, awaiting critical observers and isolating hook errors.

    Loop end and cancellation drain this loop's passive hooks so their side
    effects are visible on return.
    """
    interventions: list[Intervention] = []

    for obs in observers:
        fn = getattr(obs, method, None)
        if fn is None:
            continue

        if getattr(obs, "critical", False):
            try:
                rv = await fn(*args, **kwargs)
                if isinstance(rv, Intervention):
                    interventions.append(rv)
            except Exception as exc:
                _handle_observer_error(obs, method, exc)
        else:

            async def _run(
                observer: Any = obs,
                m: str = method,
                f: Callable[..., Awaitable[Any]] = fn,
                a: tuple[Any, ...] = args,
                kw: dict[str, Any] = kwargs,
            ) -> None:
                try:
                    await f(*a, **kw)
                except Exception as exc:
                    _handle_observer_error(observer, m, exc)

            task = asyncio.create_task(_run())
            owner = asyncio.current_task()
            if owner is None:  # pragma: no cover - create_task also needs a running loop
                raise RuntimeError("observer dispatch requires an owning asyncio task")
            _background_tasks_by_owner.setdefault(owner, set()).add(task)
            task.add_done_callback(partial(_discard_background_task, owner))

    if method in {"on_loop_end", "on_loop_cancelled"}:
        await drain_background_observers()

    return interventions


async def drain_background_observers() -> None:
    """Drain passive observer tasks owned by the current agent loop."""
    owner = asyncio.current_task()
    if owner is None:
        return
    pending = [task for task in _background_tasks_by_owner.get(owner, ()) if not task.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


def merge_interventions(interventions: list[Intervention]) -> Intervention:
    """Merge messages, take the first non-empty stop, and OR boolean controls."""
    all_messages: list[str] = []
    inject_messages_set = False
    stop_reason: str | None = None
    skip: bool = False
    pop_last: bool = False
    continue_turn: bool = False

    for iv in interventions:
        if iv.inject_messages is not None:
            inject_messages_set = True
            all_messages.extend(iv.inject_messages)
        if stop_reason is None and iv.stop_reason:
            stop_reason = iv.stop_reason
        if iv.skip_tool_execution:
            skip = True
        if iv.pop_last_message:
            pop_last = True
        if iv.continue_to_next_turn:
            continue_turn = True

    return Intervention(
        inject_messages=all_messages if inject_messages_set else None,
        stop_reason=stop_reason,
        skip_tool_execution=skip,
        pop_last_message=pop_last,
        continue_to_next_turn=continue_turn,
    )


async def notify_tool_call(
    observers: list[Any],
    ctx: TurnContext,
    tool_call: dict[str, Any],
) -> ToolCallIntervention:
    """Merge tool-call hooks; last rewrite and first skip win."""
    rewrite: dict[str, Any] | None = None
    skip: str | None = None
    meta_updates: dict[str, Any] = {}

    for obs in observers:
        fn = getattr(obs, "on_tool_call", None)
        if fn is None:
            continue
        try:
            rv = await fn(ctx, tool_call)
        except Exception as exc:
            _handle_observer_error(obs, "on_tool_call", exc)
            continue
        if rv is None:
            continue
        if rv.rewrite_args is not None:
            rewrite = rv.rewrite_args
        if skip is None and rv.skip_with_result is not None:
            skip = rv.skip_with_result
        if rv.metadata_updates:
            meta_updates.update(rv.metadata_updates)

    return ToolCallIntervention(
        rewrite_args=rewrite,
        skip_with_result=skip,
        metadata_updates=meta_updates or None,
    )


async def notify_tool_result(
    observers: list[Any],
    ctx: TurnContext,
    result: ToolResult,
) -> ToolResult:
    """Chain tool-result hooks with last-mutation-wins semantics."""
    current = result
    for obs in observers:
        fn = getattr(obs, "on_tool_result", None)
        if fn is None:
            continue
        try:
            rv = await fn(ctx, current)
        except Exception as exc:
            _handle_observer_error(obs, "on_tool_result", exc)
            continue
        if rv is not None:
            current = rv
    return current


__all__ = [
    "ATTEMPT_ACCEPTED",
    "ATTEMPT_ACCEPTED_DEGRADED",
    "ATTEMPT_DISCARDED",
    "ATTEMPT_FAILED",
    "DELIVERED_ATTEMPT_OUTCOMES",
    "WALL_DEADLINE_MONOTONIC_KEY",
    "AgentLoopResult",
    "BaseObserver",
    "CancellationObserver",
    "CompactionEvent",
    "CompactionObserver",
    "ContextCompactionContext",
    "Intervention",
    "LLMAttemptContext",
    "LLMDeltaContext",
    "LoopConfig",
    "LoopObserver",
    "LoopPolicy",
    "ToolCallIntervention",
    "ToolResult",
    "TurnContext",
    "UsageMetadata",
    "UsageMetadataExtras",
    "deadline_remaining_s",
    "drain_background_observers",
    "merge_interventions",
    "notify_observers",
    "notify_tool_call",
    "notify_tool_result",
]
