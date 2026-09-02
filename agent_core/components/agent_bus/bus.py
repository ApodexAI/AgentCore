# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportMissingParameterType=false, reportMissingTypeArgument=false, reportUnnecessaryContains=false, reportUnnecessaryIsInstance=false
"""AgentBus — async job model for parallel sub-agent dispatch.

Phase A (Issue #24): submit/collect/abort async job model.
Phase B (Issue #24): SpawnGuard budget-aware concurrency/depth/token control.
dispatch_parallel is preserved as backward-compatible convenience wrapper.

Phase 0 thin-kernel guardrail:
- Do not add new research-specific observer defaults here.
- Do not add new domain-shaped result fields here.
- Do not add new workflow-specific prompt/result assembly here.

AgentBus is a migration target toward a generic subtask runtime. Research
observer stacks and result adapters should move to workflow-owned assembly.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from agent_core.components.agent_bus.models import (
    CollectResult,
    DepthLimitExceeded,
    JobEntry,
    PendingSessionTask,
    SessionWaitOutcome,
    SubAgentResult,
    SubAgentRuntimeSpec,
    SubAgentSession,
    SubTask,
)
from agent_core.components.agent_bus.runtime import SUBTASK_RESULT_MESSAGE_TYPE
from agent_core.components.agent_bus.runtime import (
    adapt_default_session_result as _adapt_default_session_result,
)
from agent_core.components.agent_bus.runtime import (
    adapt_default_subagent_result as _adapt_default_subagent_result,
)
from agent_core.components.agent_bus.runtime import (
    build_default_subagent_loop_config as _build_default_subagent_loop_config,
)
from agent_core.components.agent_bus.runtime import (
    build_default_subagent_observers as _build_default_subagent_observers,
)
from agent_core.components.agent_bus.runtime import (
    build_session_loop_config as _build_session_loop_config,
)
from agent_core.components.agent_bus.runtime import (
    close_session_boundary_aborted as _close_session_boundary_aborted,
)
from agent_core.components.agent_bus.runtime import (
    emit_session_task_completed as _emit_session_task_completed,
)
from agent_core.components.agent_bus.runtime import (
    emit_session_task_submitted as _emit_session_task_submitted,
)
from agent_core.components.agent_bus.runtime import (
    message_content_to_result as _message_content_to_result,
)
from agent_core.components.agent_bus.runtime import (
    rehydrate_orphaned_jobs as _rehydrate_orphaned_jobs,
)
from agent_core.components.agent_bus.runtime import (
    resolve_session_observers as _resolve_session_observers,
)
from agent_core.components.agent_bus.runtime import (
    safe_final_content as _safe_final_content,
)
from agent_core.components.agent_bus.runtime import (
    safe_metadata as _safe_metadata,
)
from agent_core.components.agent_bus.shared_pool import SharedArtifactPool
from agent_core.components.agent_bus.spawn_guard import SpawnGuard
from agent_core.components.observers.wall_clock_guard import WallClockGuard
from agent_core.events import EventType
from agent_core.loop_types import BaseObserver
from agent_core.messages import (
    Message,
    assistant_msg,
    system_msg,
    user_msg,
)
from agent_core.models.pipeline_spec import SubAgentProfile
from agent_core.protocols import EventSink
from agent_core.runtime.loop.agent_loop import run_agent_loop
from agent_core.runtime.loop.message_trimmer import (
    MessageTrimmer,
    NullTrimmer,
    find_final_assistant,
    trim_and_remap_boundaries,
)
from agent_core.runtime.registries import services as registry
from agent_core.runtime.registries.agents import AgentRegistry
from agent_core.tool import Tool

logger = logging.getLogger(__name__)


PauseCheckFn = Callable[[], Awaitable[bool]]
PauseCheckFactory = Callable[[str], PauseCheckFn | None]
_default_pause_check_factory: PauseCheckFactory | None = None
EventSinkResolver = Callable[[], EventSink | None]
_default_event_sink_resolver: EventSinkResolver | None = None
_default_session_activity = True


def configure_default_pause_check_factory(
    factory: PauseCheckFactory | None,
) -> None:
    """Configure the host pause integration used by zero-argument buses."""
    global _default_pause_check_factory
    _default_pause_check_factory = factory


def configure_default_event_sink_resolver(
    resolver: EventSinkResolver | None,
) -> None:
    """Configure host lookup for a product-specific EventSink protocol key."""
    global _default_event_sink_resolver
    _default_event_sink_resolver = resolver


def configure_default_session_activity(enabled: bool) -> None:
    """Enable or disable the portable live session activity observer."""
    global _default_session_activity
    _default_session_activity = bool(enabled)


def _with_wall_clock_guard(
    observers: list[Any] | None,
    guard: SpawnGuard | None,
) -> list[Any]:
    """Append a soft wall-clock deadline derived from ``guard.timeout_s``.

    Both spawn paths cap a sub-agent with a hard ``asyncio.wait_for(coro,
    guard.timeout_s)``. That is a cancel from OUTSIDE the loop, so the
    coroutine dies at an arbitrary ``await``: the loop never reaches an exit
    branch (``stopped_by`` stays ``""``), the trajectory ``end`` event is
    never written, and the runtime's pre-absorb ``force_final_answer`` hook —
    which sits *after* the ``wait_for`` — is skipped. The sub-agent's whole
    run is then discarded as ``(empty report)``. Measured at **52.8% of
    sub-agents** on bc200 × agent-team; see :class:`WallClockGuard`.

    Both the soft deadline here and the hard one below read the same
    ``guard.timeout_s``, which is the point of routing them through one
    helper — they cannot drift apart, and neither can the two call sites.
    Attached at this layer rather than inside each workflow's
    ``observers_builder`` so every fan-out workflow inherits it.
    """
    if guard is None or guard.timeout_s <= 0:
        return list(observers or [])
    return [
        *(observers or []),
        WallClockGuard(budget_s=float(guard.timeout_s)),
    ]


def _strip_job_suffix(task_id: str) -> str:
    """Strip ``.job.N`` suffixes iteratively to recover the root task id.

    Sub-agent spawn sites pass ``job_id`` (e.g. ``"root-1.job.2"``)
    rather than the root task id — that's the identifier threaded
    through LoopConfig. But pause_check must bind to the REAL task
    managed by ProcessManager, which exists only for the root. At
    depth >= 2 this is what makes grandsub-agents observe root pause
    instead of silently running their full max_turns.
    """
    root = task_id
    while ".job." in root:
        root = root.rsplit(".job.", 1)[0]
    return root


def _legacy_pause_check_factory(task_id: str) -> PauseCheckFn | None:
    """Product-neutral default: no process-manager pause integration."""
    if _default_pause_check_factory is None:
        return None
    try:
        return _default_pause_check_factory(task_id)
    except (ImportError, RuntimeError):
        return None


def _safe_session_filename(session_id: str) -> str:
    """Return a filesystem-safe representation of ``session_id``."""
    return session_id.replace("/", "_").replace(":", "_").replace(" ", "_")


class _SessionActivityObserver(BaseObserver):
    """Capture a bounded worker event trail for interactive team UIs."""

    critical = True
    _DETAIL_LIMIT = 8_000

    def __init__(self, session: SubAgentSession) -> None:
        self.session = session
        self._sequence = 0
        self._thinking_by_turn: dict[int, dict[str, Any]] = {}

    def _append(
        self, kind: str, title: str, detail: str, *, turn: int = 0,
        is_error: bool = False,
    ) -> dict[str, Any]:
        detail = str(detail or "").strip()
        if len(detail) > self._DETAIL_LIMIT:
            detail = detail[:self._DETAIL_LIMIT] + "\n… truncated in live view"
        self._sequence += 1
        event = {
            "id": f"{self.session.session_id}:{self._sequence}",
            "kind": kind,
            "title": title,
            "detail": detail,
            "turn": turn,
            "is_error": is_error,
            "at": time.monotonic(),
        }
        self.session.activity_events.append(event)
        return event

    async def on_llm_delta(self, ctx: Any) -> None:
        delta = str(getattr(ctx, "thinking_delta", "") or "")
        if not delta:
            return
        turn = int(getattr(ctx, "turn", 0) or 0)
        event = self._thinking_by_turn.get(turn)
        if event is None:
            event = self._append("thinking", "thinking", "", turn=turn)
            self._thinking_by_turn[turn] = event
        detail = str(event.get("detail") or "") + delta
        if len(detail) > self._DETAIL_LIMIT:
            detail = detail[:self._DETAIL_LIMIT] + "\n… truncated in live view"
        event["detail"] = detail

    async def on_llm_response(self, ctx: Any) -> None:
        thinking = str(getattr(ctx, "thinking", "") or "").strip()
        if thinking:
            turn = int(getattr(ctx, "turn", 0) or 0)
            event = self._thinking_by_turn.get(turn)
            if event is None:
                self._thinking_by_turn[turn] = self._append(
                    "thinking", "thinking", thinking, turn=turn,
                )
            else:
                event["detail"] = thinking[:self._DETAIL_LIMIT]
        text = str(getattr(ctx, "ai_text", "") or "").strip()
        if text:
            self._append(
                "message", "assistant", text,
                turn=int(getattr(ctx, "turn", 0) or 0),
            )

    async def on_tool_call(self, ctx: Any, tool_call: dict[str, Any]) -> None:
        name = str(tool_call.get("name") or "tool")
        args = tool_call.get("args") or {}
        try:
            detail = json.dumps(args, ensure_ascii=False, indent=2, default=str)
        except Exception:
            detail = str(args)
        self._append(
            "tool_call", name, detail,
            turn=int(getattr(ctx, "turn", 0) or 0),
        )

    async def on_tool_result(self, ctx: Any, result: Any) -> None:
        self._append(
            "tool_error" if bool(getattr(result, "is_error", False)) else "tool_result",
            str(getattr(result, "name", "tool") or "tool"),
            str(getattr(result, "result", "") or ""),
            turn=int(getattr(ctx, "turn", 0) or 0),
            is_error=bool(getattr(result, "is_error", False)),
        )




def _offload_dropped_messages(
    session: SubAgentSession,
    dropped: list[Message],
    out_dir: Path,
) -> None:
    """Write dropped messages to a per-session JSONL file, best-effort.

    Each message is serialised individually to avoid materialising the full
    history as a single dict — this keeps peak memory proportional to
    the largest single message rather than the entire dropped batch.
    Native ``Message`` is a plain ``TypedDict``, so it serialises directly
    with ``json.dumps`` (no langchain ``message_to_dict`` conversion needed).
    Failures are logged but never re-raised so the trim always proceeds.
    """
    import json

    try:
        session_dir = out_dir / _safe_session_filename(session.session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        out_path = session_dir / f"task_{session.total_task_count:03d}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for msg in dropped:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        session.offloaded_history_path = out_path
    except Exception:
        logger.exception(
            "eager_trim: failed to offload dropped messages for %s; trim still proceeds",
            session.session_id,
        )


# ── AgentBus ────────────────────────────────────────────────────────────


async def _run_within_context(setup, job_id, item, inner):
    """Await *inner* (the sub-agent ``run_agent_loop`` coroutine) inside the
    context manager returned by ``setup(job_id, item)``.

    Runs in the sub-agent's own asyncio task, so any contextvar the CM sets
    (e.g. a per-agent sandbox) is scoped to that sub-agent and reset on exit.
    """
    with setup(job_id, item):
        return await inner


class AgentBus:
    """Async job model for sub-agent dispatch.

    Core API: submit() / collect() / abort().
    dispatch_parallel() is a backward-compatible convenience wrapper.

    If a SpawnGuard is attached, submit() enforces concurrency/depth/token
    limits. Without a guard, only basic depth checking applies.
    """

    def __init__(
        self,
        *,
        event_sink: EventSink | None = None,
        pause_check_factory: PauseCheckFactory | None = None,
        agent_registry: AgentRegistry | None = None,
        resource_manager: Any = None,
        session_history_dir: Path | None = None,
        session_activity: bool | None = None,
    ) -> None:
        """Construct an AgentBus.

        Args:
            event_sink: Optional ``core.protocols.EventSink``. When set,
                the bus emits all telemetry through this sink directly
                instead of doing a global ``registry.get(EventSink)``
                lookup on every call.
            pause_check_factory: Optional ``(root_task_id) -> PauseCheckFn``
                builder.
            agent_registry: Optional ``AgentRegistry``. When set, the bus
                resolves agent prompts from this instance directly
                instead of doing ``registry.get(AgentRegistry)``. Phase 4
                D13 surgery 3 — lets the SDK assembly inject a
                workflow-scoped registry without polluting the global
                service container.
            resource_manager: Optional ``ResourceManager``. When set, the
                bus resolves per-role LLMs / tools through this instance
                instead of via ``registry.get(ResourceManager)``. Phase 4
                D13 surgery 3.
            session_history_dir: Optional base directory for offloading
                dropped messages to disk after eager trim. When ``None``,
                only the in-memory trim is performed (memory still reclaimed).

        ``None`` for any of the kwargs keeps the legacy registry fallback so
        existing zero-arg ``AgentBus()`` callers continue to work.
        """
        self._jobs: dict[str, JobEntry] = {}
        self._counter: int = 0
        self._spawn_guard: SpawnGuard | None = None
        self._sub_agent_profiles: dict[str, dict[str, SubAgentProfile]] = {}
        self._sessions: dict[str, SubAgentSession] = {}
        # Per-task evidence / assertion pool — populated when main agents
        # (or collect_reports) harvest completed SubAgentResults, drained
        # by the pipeline node at phase end for the report node to consume.
        self._task_aggregates: dict[str, dict[str, list[dict[str, Any]]]] = {}
        # L1-0 idempotency: job ids already handed to a fan-in caller, per
        # task. The in-memory queue pop is destructive, so without this the
        # message-based reconciliation below would re-deliver a result that
        # had already been drained and the main agent would see the same
        # ``<report>`` twice. This is also the dedupe table L2-4 needs.
        self._delivered_jobs: dict[str, set[str]] = {}
        # Agents that are addressed by ``subtask_result`` messages for a
        # task, learned from ``spawn_context`` at dispatch. Keeps the
        # reconciliation read cheap in-process; after a restart it is empty
        # and recipients are recovered from the log instead.
        self._result_recipients: dict[str, set[str]] = {}
        # Results pulled out of the message channel but not yet handed to a
        # caller. ``AgentComm.consume`` advances its cursor past the whole
        # batch it read, so returning only the first message would silently
        # drop the rest; they wait here instead — the message-channel
        # counterpart of ``session.pending_results``.
        self._recovered_results: dict[
            str, deque[tuple[str, SubAgentResult]]
        ] = {}
        self._event_sink_injected: EventSink | None = event_sink
        self._pause_check_factory: PauseCheckFactory | None = pause_check_factory
        self._agent_registry_injected: AgentRegistry | None = agent_registry
        self._resource_manager_injected: Any = resource_manager
        self._session_history_dir: Path | None = session_history_dir
        self._session_activity = (
            _default_session_activity
            if session_activity is None
            else bool(session_activity)
        )

    def _event_sink(self) -> Any:
        """Resolve the active event sink.

        Returns the constructor-injected sink when available; otherwise
        falls back to the global ``registry.get_optional(EventSink)``
        lookup, then the legacy concrete ``EventStore`` registration,
        so legacy callers keep working.
        """
        if self._event_sink_injected is not None:
            return self._event_sink_injected
        event_sink = registry.get_optional(EventSink)
        if event_sink is not None:
            return event_sink
        if _default_event_sink_resolver is not None:
            try:
                event_sink = _default_event_sink_resolver()
            except (ImportError, RuntimeError):
                event_sink = None
            if event_sink is not None:
                return event_sink
        return registry.get_optional_by_type_name("EventStore")

    def _agent_registry(self, *, required: bool) -> Any:
        """Resolve the active ``AgentRegistry``.

        Returns the constructor-injected registry when set; otherwise
        falls back to the global service registry. ``required=True``
        raises (via ``registry.get``) when no registry is available;
        ``required=False`` returns ``None`` instead.
        """
        if self._agent_registry_injected is not None:
            return self._agent_registry_injected
        if required:
            return registry.get(AgentRegistry)
        return registry.get_optional(AgentRegistry)

    def _resource_manager(self, *, required: bool) -> Any:
        """Resolve the active ``ResourceManager`` (lazy import to avoid
        a kernel→runtime.resources eager dependency at module load).


        """
        if self._resource_manager_injected is not None:
            return self._resource_manager_injected
        from agent_core.runtime.resources.manager import ResourceManager
        if required:
            direct = registry.get_optional(ResourceManager)
            if direct is not None:
                return direct
            legacy = registry.get_optional_by_type_name("ResourceManager")
            if legacy is None:
                return registry.get(ResourceManager)
            return legacy
        return registry.get_optional(ResourceManager) or registry.get_optional_by_type_name(
            "ResourceManager"
        )

    def _task_pause_check_or_none(self, task_id: str | None) -> PauseCheckFn | None:
        """Resolve a per-task pause check closure.

        Strips ``.job.N`` suffixes to recover the root task id, then:
        - if a ``pause_check_factory`` was injected, calls it directly;
        - otherwise falls back to the legacy lazy import of
          ``agent_core.runtime.pause_check.make_task_pause_check``.

        Returns ``None`` when ``task_id`` is empty (orphan/shadow runs)
        or when both paths come up empty.
        """
        if not task_id:
            return None
        root = _strip_job_suffix(task_id)
        if self._pause_check_factory is not None:
            try:
                return self._pause_check_factory(root)
            except Exception:
                logger.warning(
                    "pause_check_factory raised for task=%s; running without pause probe",
                    root, exc_info=True,
                )
                return None
        return _legacy_pause_check_factory(root)

    def register_sub_agent_profiles(
        self, node_id: str, profiles: dict[str, SubAgentProfile]
    ) -> None:
        """Register sub-agent profiles for a node (called at compile time)."""
        self._sub_agent_profiles[node_id] = profiles

    def get_sub_agent_profile(
        self, node_id: str, profile_name: str
    ) -> SubAgentProfile | None:
        """Look up a named sub-agent profile for a node."""
        node_profiles = self._sub_agent_profiles.get(node_id, {})
        return node_profiles.get(profile_name)

    def set_spawn_guard(self, guard: SpawnGuard) -> None:
        """Attach a SpawnGuard for budget-aware spawn control."""
        self._spawn_guard = guard

    @property
    def spawn_guard(self) -> SpawnGuard | None:
        return self._spawn_guard

    def _next_job_id(self, parent_task_id: str) -> str:
        self._counter += 1
        return f"{parent_task_id}.job.{self._counter}"

    # ── submit: non-blocking ────────────────────────────────────────────

    async def submit(
        self,
        parent_task_id: str,
        item: SubTask,
        *,
        shared_evidence: SharedArtifactPool | None = None,
        max_turns: int = 8,
        current_depth: int = 0,
        max_depth: int = 2,
        estimated_tokens: int = 0,
        runtime_spec: SubAgentRuntimeSpec | None = None,
        spawn_context: dict[str, Any] | None = None,
    ) -> str:
        """Submit a sub-agent job. Returns job_id immediately.

        If a SpawnGuard is attached, enforces concurrency/depth/token
        limits. Without a guard, only basic depth checking applies.

        Raises:
            DepthLimitExceeded: depth >= max_depth (or SpawnGuard depth)
            SpawnDepthExceeded: SpawnGuard depth limit
            BudgetExhausted: SpawnGuard token budget exceeded
        """
        guard = self._spawn_guard

        # Use SpawnGuard depth limit if stricter
        effective_max_depth = max_depth
        if guard and guard.max_depth < effective_max_depth:
            effective_max_depth = guard.max_depth

        if current_depth >= effective_max_depth:
            raise DepthLimitExceeded(
                f"Sub-agent depth limit exceeded: "
                f"current_depth={current_depth}, "
                f"max_depth={effective_max_depth}"
            )

        job_id = self._next_job_id(parent_task_id)
        agent_registry = self._agent_registry(required=True)
        dispatch_depth = current_depth + 1

        # Resolved before the guard reserves anything: an unregistered role
        # makes ``get_prompt_for`` raise, and a reservation taken first would
        # never come back — the RAII release lives in ``_run_and_finalize``,
        # which does not exist until the asyncio task below is created. A
        # leaked reservation is permanent, so it decays ``remaining_tokens``
        # and the concurrency count for the life of the bus.
        system_prompt = (
            item.system_prompt
            or agent_registry.get_prompt_for(item.role_id)
        )
        runtime = runtime_spec or SubAgentRuntimeSpec()

        # Layer 5+3: SpawnGuard pre-check (depth + budget — non-blocking)
        # Layer 4 (concurrency) is deferred to _run_and_finalize
        if guard:
            await guard.pre_check(
                job_id, dispatch_depth, estimated_tokens,
            )

        async def _run() -> SubAgentResult:
            # Declared outside try/except so exception handlers can
            # salvage partial metadata on failure (see the ``agent_result``
            # assignment inside the try block).
            agent_result: Any = None
            try:
                event_store = self._event_sink()
                resource_mgr = self._resource_manager(required=True)
                sub_llm = resource_mgr.get_llm(item.role_id)
                sub_tools = resource_mgr.get_tools_for_role(
                    item.role_id,
                )

                if runtime.config_builder is not None:
                    sub_config = runtime.config_builder(
                        job_id, item, max_turns,
                    )
                else:
                    sub_config = _build_default_subagent_loop_config(
                        job_id, item, max_turns,
                    )

                if runtime.observers_builder is not None:
                    # Non-session dispatch — no task_index ordinal; pass 0.
                    sub_observers = runtime.observers_builder(
                        job_id, item, 0,
                    )
                else:
                    sub_observers = _build_default_subagent_observers(
                        event_store=event_store,
                        job_id=job_id,
                    )

                # codex #6 / Log Schema v1 §2.4: stamp spawn_context onto
                # observers that surface it (WorkerTraceFileObserver
                # writes it as a top-level field on the sub trace doc).
                if spawn_context:
                    for _obs in sub_observers:
                        _setter = getattr(_obs, "set_extension_data", None)
                        if _setter is None:
                            continue
                        try:
                            _setter(spawn_context=dict(spawn_context))
                        except Exception as _exc:
                            logger.debug(
                                "spawn_context stamp failed on %s: %s",
                                type(_obs).__name__, _exc,
                            )

                sub_observers = _with_wall_clock_guard(sub_observers, guard)

                coro = run_agent_loop(
                    system_prompt=system_prompt,
                    user_message=item.question,
                    llm=sub_llm,
                    tools=sub_tools,
                    config=sub_config,
                    observers=sub_observers,
                    model_profile=runtime.model_profile,
                    history_policy=runtime.history_policy,
                    pause_check=self._task_pause_check_or_none(parent_task_id),
                )
                if runtime.context_setup is not None:
                    coro = _run_within_context(
                        runtime.context_setup, job_id, item, coro,
                    )
                # Layer 2: Timeout from SpawnGuard
                timeout = float(guard.timeout_s) if guard else None
                if timeout:
                    agent_result = await asyncio.wait_for(
                        coro, timeout=timeout,
                    )
                else:
                    agent_result = await coro
                if runtime.result_adapter is not None:
                    adapted = runtime.result_adapter(agent_result, job_id, item)
                    if inspect.isawaitable(adapted):
                        adapted = await adapted
                    return adapted
                return _adapt_default_subagent_result(
                    agent_result, job_id, item,
                )
            except asyncio.CancelledError:
                return SubAgentResult(
                    question=item.question,
                    role_id=item.role_id,
                    final_content=_safe_final_content(agent_result),
                    success=False,
                    error="aborted",
                    error_class="CancelledError",
                    job_id=job_id,
                    metadata=_safe_metadata(agent_result),
                )
            except TimeoutError:
                logger.warning(
                    "Sub-agent job %s timed out", job_id,
                )
                return SubAgentResult(
                    question=item.question,
                    role_id=item.role_id,
                    final_content=_safe_final_content(agent_result),
                    success=False,
                    error="timeout",
                    error_class="TimeoutError",
                    job_id=job_id,
                    metadata=_safe_metadata(agent_result),
                )
            except Exception as exc:
                logger.warning(
                    "Sub-agent job %s failed: %s", job_id, exc
                )
                return SubAgentResult(
                    question=item.question,
                    role_id=item.role_id,
                    final_content=_safe_final_content(agent_result),
                    success=False,
                    error=str(exc),
                    error_class=type(exc).__name__,
                    job_id=job_id,
                    metadata=_safe_metadata(agent_result),
                )

        async def _run_and_finalize() -> SubAgentResult:
            entry = self._jobs[job_id]
            # Layer 4: Concurrency gate (may queue here)
            if guard:
                await guard.acquire_slot(job_id)
            entry.status = "running"
            try:
                result = await _run()
                entry.result = result
                entry.completed_at = time.monotonic()
                if result.error == "aborted":
                    entry.status = "aborted"
                elif result.success:
                    entry.status = "completed"
                else:
                    entry.status = "failed"
                return result
            finally:
                # Layer 1: RAII release — always release guard slot
                if guard:
                    guard.release(job_id)

        # Register the entry before spawning: ``_run_and_finalize`` looks
        # itself up in ``self._jobs``, and the reservation has no RAII owner
        # until that task exists — so anything that fails in between has to
        # hand the reservation back explicitly.
        entry = JobEntry(
            job_id=job_id,
            parent_task_id=parent_task_id,
            item=item,
            task=None,
            status="submitted",
            submitted_at=time.monotonic(),
        )
        self._jobs[job_id] = entry
        try:
            task = asyncio.create_task(_run_and_finalize(), name=job_id)
        except BaseException:
            self._jobs.pop(job_id, None)
            if guard:
                guard.release(job_id)
            raise
        entry.task = task

        event_store = self._event_sink()
        if event_store:
            await event_store.append(
                task_id=parent_task_id,
                event_type=EventType.AGENT_ACTION,
                payload={
                    "trace_type": "agent_submitted",
                    "parent_task_id": parent_task_id,
                    "job_id": job_id,
                    "question": item.question,
                    "role_id": item.role_id,
                    "depth": dispatch_depth,
                },
                agent_role="system",
            )

        logger.info(
            "Submitted job %s (question=%s, role=%s, depth=%d)",
            job_id, item.question[:60], item.role_id, dispatch_depth,
        )
        return job_id

    # ── collect: wait with timeout ──────────────────────────────────────

    async def collect(
        self,
        job_ids: list[str],
        *,
        timeout: float = 1800,
    ) -> CollectResult:
        """Wait for jobs to complete. Returns partial results on timeout.

        Default 1800s matches the ``collect_results`` tool's user-facing
        default — sub-agents on hard BrowseComp / FrontierScience runs
        regularly need 10–20 min, and a shorter primitive default would
        silently strand work for any caller that forgets to override.
        """
        tasks_to_wait: dict[asyncio.Task, str] = {}
        result = CollectResult()

        for jid in job_ids:
            entry = self._jobs.get(jid)
            if entry is None:
                continue
            if entry.result is not None:
                # Already finished
                if entry.status == "failed" or not entry.result.success:
                    result.failed.append(entry.result)
                else:
                    result.completed.append(entry.result)
            else:
                if entry.status == "queued" or entry.task is None:
                    result.pending.append(jid)
                else:
                    tasks_to_wait[entry.task] = jid

        if tasks_to_wait:
            done, pending = await asyncio.wait(
                tasks_to_wait.keys(),
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED,
            )

            for task in done:
                jid = tasks_to_wait[task]
                entry = self._jobs.get(jid)
                if entry and entry.result:
                    if entry.result.success:
                        result.completed.append(entry.result)
                    else:
                        result.failed.append(entry.result)

            for task in pending:
                jid = tasks_to_wait[task]
                result.pending.append(jid)

        event_store = self._event_sink()
        if event_store and job_ids:
            parent = self._jobs.get(job_ids[0])
            parent_task_id = parent.parent_task_id if parent else ""
            await event_store.append(
                task_id=parent_task_id,
                event_type=EventType.AGENT_ACTION,
                payload={
                    "trace_type": "agent_collected",
                    "job_ids": job_ids,
                    "completed": len(result.completed),
                    "failed": len(result.failed),
                    "pending": len(result.pending),
                },
                agent_role="system",
            )

        return result

    # ── abort: graceful cancel ──────────────────────────────────────────

    async def abort(self, job_id: str) -> str:
        """Abort a job. Returns final status string.

        Idempotent: aborting a completed job returns its actual status.
        Aborting a queued (not-yet-dispatched) job removes it from its
        session's pending queue, releases its guard reservation, and
        marks it aborted without touching the running task.
        """
        entry = self._jobs.get(job_id)
        if entry is None:
            return "not_found"

        # Already finished — don't override
        if entry.status in ("completed", "failed", "aborted"):
            return entry.status

        if entry.status == "queued" or entry.task is None:
            # Queued task: scrub from the session's pending deque,
            # release any guard reservation, and finalise as aborted.
            self._purge_queued_job(job_id)
            self._mark_job_aborted(job_id, entry)
        else:
            # Running task: cancel the asyncio task and let
            # _run_and_finalize observe the CancelledError.
            entry.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await entry.task

            # If the task finished between cancel() and await (race),
            # respect the actual result
            if entry.result and entry.result.success:
                entry.status = "completed"
            elif entry.status not in ("completed", "failed"):
                self._mark_job_aborted(job_id, entry)

        event_store = self._event_sink()
        if event_store:
            await event_store.append(
                task_id=entry.parent_task_id,
                event_type=EventType.AGENT_ACTION,
                payload={
                    "trace_type": "agent_aborted",
                    "job_id": job_id,
                    "final_status": entry.status,
                },
                agent_role="system",
            )

        logger.info("Aborted job %s → %s", job_id, entry.status)
        return entry.status

    async def cancel_for_parent(self, parent_task_id: str) -> int:
        """Cancel all pending/running jobs for a parent task.

        Called between agent mode rounds to prevent orphaned sub-agents
        from holding semaphore slots and blocking the next round. Both
        running tasks (asyncio cancel) and queued tasks (purge + status
        flip) are handled — neither survives the round boundary.

        Queued jobs are cleared first, in a separate pass. Cancelling a
        running job runs its ``finally``, which drains that session's queue —
        so a single interleaved pass would dispatch the very tasks this call
        is tearing down, let them reach the LLM, and cancel them again a few
        iterations later.

        Returns the number of jobs this call cancelled.
        """
        cancelled = 0
        mine = [
            (jid, entry) for jid, entry in self._jobs.items()
            if entry.parent_task_id == parent_task_id
            and entry.status not in ("completed", "failed", "aborted")
        ]

        for jid, entry in mine:
            if entry.status == "queued" or entry.task is None:
                self._purge_queued_job(jid)
                self._mark_job_aborted(jid, entry)
                cancelled += 1

        for jid, entry in mine:
            if entry.status in ("completed", "failed", "aborted"):
                continue
            task = entry.task
            if task is None:
                continue
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            # Counted either way. A job that had actually started absorbs its
            # own cancellation — ``_run_and_finalize`` reaches the handler that
            # flips the status itself — and only one that never got that far
            # still needs the stand-in below. Both were cancelled by this call,
            # so gating the count on which of the two happened reported 0 for a
            # round that really did tear down every running sub-agent.
            if entry.status not in ("completed", "failed", "aborted"):
                self._mark_job_aborted(jid, entry)
            cancelled += 1
        if cancelled:
            logger.info(
                "Cancelled %d orphaned sub-agent jobs for %s",
                cancelled, parent_task_id,
            )
        return cancelled

    def _purge_queued_job(self, job_id: str) -> None:
        """Remove ``job_id`` from whichever session has it queued.

        Releases the guard's pre-check reservation if a guard is
        attached — queued tasks reserved budget at submit time and
        won't reach the ``_run_and_finalize`` finally that normally
        releases.
        """
        for session in self._sessions.values():
            queue = session.pending_tasks
            if not queue:
                continue
            kept: deque[PendingSessionTask] = deque()
            removed = False
            for pending in queue:
                if pending.job_id == job_id:
                    removed = True
                    continue
                kept.append(pending)
            if removed:
                session.pending_tasks = kept
                if self._spawn_guard is not None:
                    self._spawn_guard.release(job_id)
                return

    def _mark_job_aborted(
        self, job_id: str, entry: JobEntry | None = None,
    ) -> None:
        """Mark a job aborted and release any SpawnGuard reservation.

        This is intentionally idempotent. A task can be cancelled before
        its coroutine reaches the ``finally`` block that normally releases
        the guard reservation; queued tasks never reach that block at all.
        Calling ``release`` here covers both without double-freeing slots.
        """
        if self._spawn_guard is not None:
            self._spawn_guard.release(job_id)
        entry = entry or self._jobs.get(job_id)
        if entry is None or entry.status in ("completed", "failed", "aborted"):
            return
        entry.status = "aborted"
        entry.completed_at = time.monotonic()
        if entry.result is not None:
            return
        item = entry.item
        entry.result = SubAgentResult(
            question=str(getattr(item, "question", "") or ""),
            role_id=str(getattr(item, "role_id", "") or ""),
            final_content="",
            success=False,
            error="aborted",
            error_class="CancelledError",
            job_id=job_id,
        )

    # ── get_job_status: query job state ─────────────────────────────────

    def get_job_status(self, job_id: str) -> str | None:
        """Return current status of a job, or None if not found."""
        entry = self._jobs.get(job_id)
        return entry.status if entry else None

    # ── Persistent sessions (AgentTeam-style multi-task sub-agents) ─────

    def _make_session_id(self, task_id: str, name: str) -> str:
        return f"{task_id}::{name}"

    async def create_session(
        self,
        *,
        task_id: str,
        name: str,
        role_id: str,
        system_prompt: str | None = None,
        tools_override: list[Tool] | None = None,
        llm_override: Any = None,
        trimmer: MessageTrimmer | None = None,
        max_turns: int = 100,
        tool_result_max_chars: int | None = None,
        runtime_spec: SubAgentRuntimeSpec | None = None,
        llm_timeout: int | None = None,
    ) -> str:
        """Create (or return existing) persistent sub-agent session.

        Idempotent ONLY when the caller asks for exactly the same session:
        same ``(task_id, name, role_id, system_prompt)``. If a session with
        the same ``(task_id, name)`` exists but with a different role or
        system prompt, raise ``ValueError`` rather than silently reusing
        the old one — agent names are often LLM-generated and a clashing
        specialization is almost always a bug (e.g. a fresh verifier
        getting turned into a reused researcher).

        ``system_prompt`` falls back to ``AgentRegistry.get_prompt_for(role_id)``.
        ``tools_override`` / ``llm_override`` fall back to ResourceManager.
        ``trimmer`` defaults to ``NullTrimmer``.

        ``llm_timeout`` (seconds) sets the per-LLM-call timeout for every
        task submitted to this session. ``None`` (default) defers to
        ``LoopConfig.llm_timeout`` (180s). Set this when the session's
        model has high response latency variance (e.g. a strong reasoning
        model used for review). Identity check (idempotent reuse) ignores
        this field — it is treated as a runtime knob, not part of the
        session's identity.
        """
        prompt = system_prompt
        if prompt is None:
            agent_registry = self._agent_registry(required=False)
            if agent_registry is not None:
                prompt = agent_registry.get_prompt_for(role_id)
        if not prompt:
            prompt = ""

        session_id = self._make_session_id(task_id, name)
        existing = self._sessions.get(session_id)
        if existing is not None:
            if existing.role_id != role_id or existing.system_prompt != prompt:
                raise ValueError(
                    f"Session name {name!r} already exists for task "
                    f"{task_id!r} with a different role or system_prompt "
                    f"(existing role={existing.role_id!r}, requested "
                    f"role={role_id!r}). Pick a unique name."
                )
            logger.debug(
                "create_session: returning existing session %s", session_id,
            )
            return session_id

        llm = llm_override
        tools = tools_override
        if llm is None or tools is None:
            resource_mgr = self._resource_manager(required=False)
            if llm is None and resource_mgr is not None:
                llm = resource_mgr.get_llm(role_id)
            if tools is None and resource_mgr is not None:
                tools = resource_mgr.get_tools_for_role(role_id)
        if tools is None:
            tools = []

        session = SubAgentSession(
            session_id=session_id,
            task_id=task_id,
            name=name,
            role_id=role_id,
            system_prompt=prompt,
            tools=tools,
            llm=llm,
            trimmer=trimmer or NullTrimmer(),
            max_turns=max_turns,
            tool_result_max_chars=tool_result_max_chars,
            llm_timeout=llm_timeout,
            runtime_spec=runtime_spec,
        )
        self._sessions[session_id] = session

        event_store = self._event_sink()
        if event_store is not None:
            await event_store.append(
                task_id=task_id,
                event_type=EventType.AGENT_ACTION,
                payload={
                    "trace_type": "session_created",
                    "session_id": session_id,
                    "name": name,
                    "role_id": role_id,
                    "agent": name,
                    "action": "session_created",
                    "detail": (
                        f"Created sub-agent '{name}' (role={role_id})"
                    ),
                },
                agent_role="system",
            )

        logger.info(
            "Created session %s (role=%s, max_turns=%d)",
            session_id, role_id, max_turns,
        )
        return session_id

    async def submit_task_to_session(
        self,
        session_id: str,
        task_prompt: str,
        *,
        max_turns: int | None = None,
        observers: list[Any] | None = None,
        estimated_tokens: int = 0,
        runtime_spec: SubAgentRuntimeSpec | None = None,
        spawn_context: dict[str, Any] | None = None,
        task_metadata: dict[str, Any] | None = None,
    ) -> str:
        """Queue a task on an existing session. Non-blocking: returns job_id.

        Sessions execute tasks **strictly serially** — the session's
        message list / boundary list cannot be safely mutated by two
        concurrent runs. When the session is already busy, the new task
        joins ``session.pending_tasks`` (FIFO) and runs as soon as the
        current task finalises. This is transparent to the caller: the
        returned ``job_id`` is valid immediately and ``collect_reports``
        sees queued tasks once they start running.

        When a ``SpawnGuard`` is attached to the bus, session tasks flow
        through the same five-layer protection as ``submit()``: depth +
        token-budget pre-check (immediate), concurrency semaphore (per-
        run, gated inside the task), per-job wall-time timeout, and RAII
        slot release. This prevents a bad decomposition from fanning out
        unbounded background work on the session path.

        Raises ``KeyError`` if the session doesn't exist.
        Raises ``SpawnDepthExceeded`` / ``BudgetExhausted`` via the
        guard's pre-check (queued tasks reserve budget at submit time
        too — aborting a queued task releases the reservation).
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session {session_id!r} not found")

        guard = self._spawn_guard
        # Sessions are spawned by the main agent (depth 0) → session tasks
        # are depth 1. The swarm_sub role doesn't itself carry
        # delegate_subtask / create_subagent so there's no further
        # nesting in the current workflow, but we still respect the guard
        # so future recursive workflows don't silently bypass it.
        dispatch_depth = 1

        job_id = self._next_job_id(session.task_id)
        pending = PendingSessionTask(
            job_id=job_id,
            task_prompt=task_prompt,
            max_turns=max_turns,
            observers=observers,
            estimated_tokens=estimated_tokens,
            runtime_spec=runtime_spec,
            spawn_context=dict(spawn_context) if spawn_context else None,
            task_metadata=dict(task_metadata or {}),
        )

        # Layer 5+3: SpawnGuard pre-check at submit time (not at dispatch)
        # so queued tasks still raise BudgetExhausted / SpawnDepthExceeded
        # synchronously at the API boundary — preserves the legacy
        # behaviour that callers see budget errors immediately rather
        # than only via ``collect_reports`` after the queue drains.
        if guard is not None:
            await guard.pre_check(job_id, dispatch_depth, estimated_tokens)

        # Past the pre-check the reservation has no RAII owner yet: it passes
        # to ``session.pending_tasks`` (released when the queue drains and the
        # task runs) or to ``_dispatch_session_task`` (which releases on its
        # own failures). Anything that raises in between must hand it back
        # here, or it leaks for the life of the bus — permanently shrinking
        # ``remaining_tokens`` and the concurrency count.
        try:
            return await self._queue_or_dispatch_session_task(session, pending)
        except BaseException:
            if guard is not None:
                guard.release(job_id)
            raise

    async def _queue_or_dispatch_session_task(
        self,
        session: SubAgentSession,
        pending: PendingSessionTask,
    ) -> str:
        """Park ``pending`` behind the running task, or dispatch it now.

        Split out of :meth:`submit_task_to_session` so one wrapper there can
        release the guard reservation on any failure along this path.
        """
        job_id = pending.job_id
        session_id = session.session_id
        task_prompt = pending.task_prompt
        task_metadata = pending.task_metadata

        # Busy = a current task is in submitted/running state. ``queued``
        # current_job_id can't happen (we only set current_job_id at
        # dispatch time below), but check defensively in case future code
        # paths set the field early.
        busy = False
        if session.current_job_id is not None:
            current_entry = self._jobs.get(session.current_job_id)
            if current_entry is not None and current_entry.status in (
                "submitted", "running",
            ):
                busy = True

        if busy:
            # Park in the queue with a placeholder JobEntry so consumers
            # that look up by ``job_id`` (status queries, abort paths)
            # see the task before it actually starts running.
            entry = JobEntry(
                job_id=job_id,
                parent_task_id=session.task_id,
                item=SubTask(
                    question=task_prompt, role_id=session.role_id,
                    system_prompt=session.system_prompt,
                    metadata=dict(task_metadata or {}),
                ),
                task=None,
                status="queued",
                submitted_at=time.monotonic(),
            )
            self._jobs[job_id] = entry
            session.pending_tasks.append(pending)
            logger.debug(
                "Session %s busy → queued job %s (%d ahead)",
                session_id, job_id, len(session.pending_tasks) - 1,
            )
            return job_id

        # Free path: dispatch immediately.
        await self._dispatch_session_task(session, pending)
        return job_id

    async def _dispatch_session_task(
        self,
        session: SubAgentSession,
        pending: PendingSessionTask,
    ) -> None:
        """Spin up the asyncio task for ``pending`` on this session.

        Opens the new task boundary (so the trimmer sees only closed
        boundaries when seeding ``initial_messages``), registers /
        upgrades the ``JobEntry``, and emits the
        ``session_task_submitted`` SSE event. Called from the free
        branch of ``submit_task_to_session`` for fresh tasks and from
        ``_drain_session_queue`` when the previous task finalises.

        Releases the guard reservation on failure. Until ``_run_and_finalize``
        exists as an asyncio task, no ``finally`` owns that reservation — and
        this method can raise well before then (``trimmer.trim`` on a
        pathological history, the ``SubTask`` build, a cancelled dispatch).
        ``_drain_session_queue`` swallows dispatch failures by design, so
        without this the queued task's reservation would leak silently.
        """
        try:
            await self._dispatch_session_task_inner(session, pending)
        except BaseException:
            guard = self._spawn_guard
            if guard is not None:
                guard.release(pending.job_id)
            raise

    async def _dispatch_session_task_inner(
        self,
        session: SubAgentSession,
        pending: PendingSessionTask,
    ) -> None:
        """Body of :meth:`_dispatch_session_task`; see its docstring."""
        guard = self._spawn_guard
        job_id = pending.job_id
        task_prompt = pending.task_prompt
        observers = pending.observers
        max_turns = pending.max_turns
        runtime_spec = pending.runtime_spec
        session_id = session.session_id

        # Build initial_messages from trimmed prior history, before we
        # mutate session.messages so the trimmer sees the clean state.
        trimmed = session.trimmer.trim(
            session.messages, session.task_boundaries,
        )
        initial_messages: list[Message] = [
            system_msg(session.system_prompt), *trimmed,
        ]

        # Open a new boundary — the task prompt index is the current tail.
        session.total_task_count += 1
        boundary_start = len(session.messages)
        session.messages.append(user_msg(task_prompt))
        session.task_boundaries.append((boundary_start, None))

        effective_max_turns = (
            max_turns if max_turns is not None else session.max_turns
        )
        active_runtime = (
            runtime_spec or session.runtime_spec or SubAgentRuntimeSpec()
        )
        session_item = SubTask(
            question=task_prompt,
            role_id=session.role_id,
            system_prompt=session.system_prompt,
            metadata=dict(pending.task_metadata),
        )

        # Snapshot for _run closure so later mutations don't leak.
        # ``run_agent_loop`` only appends a HumanMessage when ``user_message``
        # is non-empty (resume path re-enters with ``user_message=""``). We
        # always pass ``task_prompt`` here which is non-empty by contract,
        # but the explicit branch keeps the accounting correct if a future
        # caller ever invokes this path with an empty prompt.
        prefix_count = len(initial_messages) + (1 if task_prompt else 0)

        async def _run() -> SubAgentResult:
            # ``raw_result`` lets the exception handlers salvage any
            # partial metadata (evidence, assertions, react_steps) when
            # the failure came AFTER the agent loop completed — e.g.
            # the result_adapter raised, or the boundary bookkeeping
            # hit an assertion. Without this, sub-agents that did real
            # web_search work but tripped on post-loop bookkeeping
            # would leak their evidence silently.
            raw_result: Any = None
            try:
                if active_runtime.config_builder is not None:
                    loop_config = active_runtime.config_builder(
                        job_id, session_item, effective_max_turns,
                    )
                else:
                    loop_config = _build_session_loop_config(
                        session, job_id, effective_max_turns,
                    )

                if active_runtime.observers_builder is not None:
                    # Pass task_index as the 1-based ordinal within the session
                    # (session.total_task_count was already incremented before
                    # this _run closure was constructed).
                    loop_observers = active_runtime.observers_builder(
                        job_id, session_item, session.total_task_count,
                    )
                else:
                    loop_observers = _resolve_session_observers(
                        observers, session,
                    )

                # codex #6 / Log Schema v1 §2.4: stamp the spawn_context
                # onto any observer with ``set_extension_data`` (today:
                # WorkerTraceFileObserver) so the sub-agent's trace file
                # carries the delegation lineage at top-level. SFT reflow
                # reads ``doc["spawn_context"]`` when merging per-sub
                # docs into the canonical ``runs[i].sub_agents[]`` shape.
                if pending.spawn_context:
                    for _obs in loop_observers:
                        _setter = getattr(_obs, "set_extension_data", None)
                        if _setter is None:
                            continue
                        try:
                            _setter(spawn_context=dict(pending.spawn_context))
                        except Exception as _exc:
                            logger.debug(
                                "spawn_context stamp failed on %s: %s",
                                type(_obs).__name__, _exc,
                            )

                if self._session_activity:
                    loop_observers.append(_SessionActivityObserver(session))
                loop_observers = _with_wall_clock_guard(loop_observers, guard)

                coro = run_agent_loop(
                    system_prompt=session.system_prompt,
                    user_message=task_prompt,
                    initial_messages=initial_messages,
                    llm=session.llm,
                    tools=session.tools,
                    config=loop_config,
                    observers=loop_observers,
                    model_profile=active_runtime.model_profile,
                    history_policy=active_runtime.history_policy,
                    pause_check=self._task_pause_check_or_none(session.task_id),
                )
                if active_runtime.context_setup is not None:
                    coro = _run_within_context(
                        active_runtime.context_setup, job_id, session_item, coro,
                    )
                # Layer 2: per-job wall-time timeout (from guard).
                timeout = float(guard.timeout_s) if guard else None
                if timeout and timeout > 0:
                    raw_result = await asyncio.wait_for(coro, timeout=timeout)
                else:
                    raw_result = await coro

                # Pre-absorb hook: let the runtime spec inject a forced
                # final turn (e.g. swarm's ``force_final_answer``)
                # BEFORE we copy raw_result.messages into the session.
                # Anything appended here lands inside the closing
                # boundary and feeds session.last_report below.
                if active_runtime.force_finalizer is not None:
                    try:
                        forced = active_runtime.force_finalizer(
                            raw_result, session_item,
                        )
                        if asyncio.iscoroutine(forced):
                            forced = await forced
                        if forced is not None:
                            raw_result = forced
                    except Exception as exc:
                        logger.warning(
                            "force_finalizer failed for session %s "
                            "task #%d: %s — continuing with un-forced "
                            "result",
                            session.session_id,
                            session.total_task_count,
                            exc,
                        )

                # Absorb new turns back into session.messages.
                new_turns = list(raw_result.messages[prefix_count:])
                session.messages.extend(new_turns)

                # Boundary closure invariant: the slice (start..end) must
                # contain at least one clean AIMessage so the trimmer can
                # surface a "final report" on reuse. When the loop exits
                # without one (max_turns / budget / llm_error and the
                # spec didn't supply / didn't successfully run a
                # force_finalizer), synthesise a stub from
                # raw_result.final_content or a static reason marker.
                final_text = (raw_result.final_content or "").strip()
                start_idx, _ = session.task_boundaries[-1]
                provisional_end = len(session.messages) - 1
                if find_final_assistant(
                    session.messages, start_idx + 1, provisional_end,
                ) is None:
                    stopped = (
                        getattr(raw_result, "stopped_by", "")
                        or "unknown"
                    )
                    stub_text = (
                        final_text
                        or f"[task ended without a clean final answer "
                           f"(stopped_by={stopped})]"
                    )
                    session.messages.append(assistant_msg(stub_text))

                # Close the boundary for this task.
                end_idx = len(session.messages) - 1
                session.task_boundaries[-1] = (start_idx, end_idx)
                # Fall back to the previous last_report when the new
                # run produced nothing — keeps cross-agent <attach/>
                # references stable mid-session.
                session.last_report = (
                    final_text
                    or session.last_report
                )

                # Bubble evidence + assertions harvested by observers up
                # to the caller so collect_reports / main_agent_node
                # can surface them to the report node + frontend.
                if active_runtime.result_adapter is not None:
                    adapted = active_runtime.result_adapter(
                        raw_result, job_id, session_item,
                    )
                    if inspect.isawaitable(adapted):
                        adapted = await adapted
                    return adapted
                return _adapt_default_session_result(
                    raw_result, job_id, session, task_prompt,
                )
            except asyncio.CancelledError:
                _close_session_boundary_aborted(session)
                return SubAgentResult(
                    question=task_prompt,
                    role_id=session.role_id,
                    final_content=_safe_final_content(raw_result),
                    success=False,
                    error="aborted",
                    error_class="CancelledError",
                    job_id=job_id,
                    metadata=_safe_metadata(raw_result),
                )
            except TimeoutError:
                logger.warning("Session task %s timed out", job_id)
                _close_session_boundary_aborted(session)
                return SubAgentResult(
                    question=task_prompt,
                    role_id=session.role_id,
                    final_content=_safe_final_content(raw_result),
                    success=False,
                    error="timeout",
                    error_class="TimeoutError",
                    job_id=job_id,
                    metadata=_safe_metadata(raw_result),
                )
            except Exception as exc:
                logger.warning(
                    "Session task %s failed: %s", job_id, exc,
                )
                _close_session_boundary_aborted(session)
                return SubAgentResult(
                    question=task_prompt,
                    role_id=session.role_id,
                    final_content=_safe_final_content(raw_result),
                    success=False,
                    error=str(exc),
                    error_class=type(exc).__name__,
                    job_id=job_id,
                    metadata=_safe_metadata(raw_result),
                )

        async def _run_and_finalize() -> SubAgentResult:
            entry = self._jobs[job_id]
            # Layer 4: Concurrency gate (may queue when max_parallel reached).
            if guard is not None:
                await guard.acquire_slot(job_id)
            entry.status = "running"
            try:
                result = await _run()
                entry.result = result
                entry.completed_at = time.monotonic()
                if result.error == "aborted":
                    entry.status = "aborted"
                elif result.success:
                    entry.status = "completed"
                else:
                    entry.status = "failed"
                if pending.task_metadata.get("can_publish") is True:
                    stopped_by = str(result.metadata.get("stopped_by") or "")
                    tool_calls = int(result.metadata.get("tool_calls_count", 0) or 0)
                    # Any ``no_tool`` ending earns a credit, not just a session
                    # with zero tool calls. The common degenerate shape is a
                    # publisher that runs a few commands and THEN emits its
                    # write as a fenced block — tool_calls_count > 0, yet the
                    # deliverable was never written, which is exactly the
                    # sticky-cap deadlock the credit exists to break.
                    if stopped_by == "no_tool":
                        session.publisher_no_tool_failures += 1
                    elif tool_calls > 0:
                        # Session demonstrated working tool use: drop the
                        # accumulated credit so it does not permanently inflate
                        # this session's task cap for the rest of the run.
                        session.publisher_no_tool_failures = 0
                # Enqueue for wait_any_session regardless of success/failure
                # so callers can observe errors (matching AgentTeam behavior
                # where failed agents also emit a <report>).
                session.pending_results.append(result)
                await _emit_session_task_completed(
                    session, job_id, result,
                    event_sink=self._event_sink(),
                    spawn_context=pending.spawn_context,
                )
                return result
            finally:
                # Layer 1: RAII release — always free the guard slot, even
                # if the inner loop crashed before entry.status was set.
                if guard is not None:
                    guard.release(job_id)
                if entry.submitted_at:
                    completed_at = entry.completed_at or time.monotonic()
                    session.last_task_elapsed_s = max(
                        0.0, completed_at - entry.submitted_at,
                    )
                if session.current_job_id == job_id:
                    session.current_job_id = None
                # Eager trim: compress completed boundaries before the next
                # task starts reading session.messages. Must be awaited here
                # (not fire-and-forget) to avoid a race with _drain_session_queue
                # reading stale messages for initial_messages construction.
                try:
                    await self._eager_trim_and_offload(session)
                except Exception:
                    logger.exception(
                        "eager_trim failed for session %s; continuing",
                        session.session_id,
                    )
                # Hand off to the next queued task (if any) so the
                # session continues draining without the main agent
                # having to re-call ``assign_task``. Failures are logged
                # but never re-raised — a broken successor must not
                # corrupt the just-finalised result.
                try:
                    await self._drain_session_queue(session)
                except Exception as exc:
                    logger.warning(
                        "Session %s: queue drain after %s failed: %s",
                        session.session_id, job_id, exc,
                    )

        # Either upgrade an already-registered queued JobEntry or create
        # a fresh one (free path).
        existing = self._jobs.get(job_id)
        if existing is not None:
            existing.status = "submitted"
            existing.submitted_at = existing.submitted_at or time.monotonic()
            entry = existing
        else:
            entry = JobEntry(
                job_id=job_id,
                parent_task_id=session.task_id,
                item=SubTask(
                    question=task_prompt, role_id=session.role_id,
                    system_prompt=session.system_prompt,
                    metadata=dict(pending.task_metadata),
                ),
                task=None,
                status="submitted",
                submitted_at=time.monotonic(),
            )
            self._jobs[job_id] = entry
        session.current_job_id = job_id
        try:
            task = asyncio.create_task(
                _run_and_finalize(), name=f"session:{session_id}:{job_id}",
            )
        except BaseException:
            if session.current_job_id == job_id:
                session.current_job_id = None
            if existing is None:
                self._jobs.pop(job_id, None)
            raise
        entry.task = task
        # L1-0: remember who this job's result message will address, so the
        # reconciliation read stays a cursor consume rather than a log scan.
        self._note_result_recipient(session.task_id, pending.spawn_context)

        try:
            await _emit_session_task_submitted(
                session, job_id, task_prompt,
                event_sink=self._event_sink(),
            )

            # Dispatch is fire-and-return: a host tool such as ``assign_task``
            # hands back a job id and its invocation scope closes. Nothing
            # above is guaranteed to suspend — the event helper returns
            # without awaiting when no sink is installed, and a sink whose
            # ``append`` never awaits does not yield either — so explicitly
            # give the child task one turn before returning its id.
            await asyncio.sleep(0)
        except BaseException:
            # Ownership has already transferred to ``task``. If dispatch is
            # cancelled or event publication fails now, the caller never gets
            # the job id, so leaving the child alive would create an
            # unreachable sub-agent. It would also let the outer failure path
            # release a SpawnGuard reservation while the child still runs.
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            if entry.status not in ("completed", "failed", "aborted"):
                _close_session_boundary_aborted(session)
                self._mark_job_aborted(job_id, entry)
            if session.current_job_id == job_id:
                session.current_job_id = None
            raise

    async def _eager_trim_and_offload(self, session: SubAgentSession) -> None:
        """Compress completed boundaries immediately after a task finishes.

        Must be awaited before ``_drain_session_queue`` to prevent a race
        where the next queued task reads stale (untrimmed) messages as its
        initial_messages. For single-task specialist sessions, this reclaims
        the full intermediate history (tool calls + results) right after the
        task completes, leaving only [task_prompt, final_ai].
        """
        new_messages, new_boundaries = trim_and_remap_boundaries(
            session.messages, session.task_boundaries,
        )
        if new_messages is session.messages:
            # trim_and_remap_boundaries returns the same object when there are
            # no completed boundaries — nothing to do.
            return

        old_len = len(session.messages)
        if self._session_history_dir is not None:
            kept_ids = {id(m) for m in new_messages}
            dropped = [m for m in session.messages if id(m) not in kept_ids]
            await asyncio.to_thread(
                _offload_dropped_messages,
                session,
                dropped,
                self._session_history_dir,
            )

        session.messages = new_messages
        session.task_boundaries = new_boundaries
        logger.debug(
            "eager_trim: session %s messages %d → %d (dropped %d)",
            session.session_id,
            old_len,
            len(new_messages),
            old_len - len(new_messages),
        )

    async def _drain_session_queue(
        self,
        session: SubAgentSession,
    ) -> None:
        """Pop and dispatch the next queued task on this session, if any.

        Called from ``_run_and_finalize`` after the current task clears
        ``current_job_id``. Only dispatches one task at a time — the
        next queued task will trigger the same drain when *it* finishes,
        keeping execution strictly serial.
        """
        if not session.pending_tasks:
            return
        next_pending = session.pending_tasks.popleft()
        # If the queued JobEntry was aborted while parked, skip it and
        # try the next one. ``abort_job`` removes the queued PendingTask
        # before flipping status, but cancel_for_parent doesn't touch
        # the queue first; defensive check covers both.
        entry = self._jobs.get(next_pending.job_id)
        if entry is not None and entry.status == "aborted":
            await self._drain_session_queue(session)
            return
        await self._dispatch_session_task(session, next_pending)

    async def _reconcile_terminal_session_jobs(self, task_id: str) -> None:
        """Turn terminal tasks that never published a result into failures.

        A task can be cancelled before ``_run_and_finalize`` enters its
        ``try/finally`` block (or fail in wrapper setup). In that case the
        asyncio task is done, but ``current_job_id`` and the job status remain
        ``submitted``/``running`` forever. The old wait path treated that
        stale state as a timeout and returned immediately, making callers say
        they had waited 30 minutes when only milliseconds elapsed.
        """
        sessions_to_drain: list[SubAgentSession] = []
        for session in self._sessions.values():
            if session.task_id != task_id or session.current_job_id is None:
                continue
            job_id = session.current_job_id
            entry = self._jobs.get(job_id)
            task = entry.task if entry is not None else None
            if entry is None or task is None or not task.done():
                continue

            result = entry.result
            if result is None and not task.cancelled():
                try:
                    task_result = task.result()
                except (asyncio.CancelledError, Exception) as exc:
                    error = str(exc) or type(exc).__name__
                    error_class = type(exc).__name__
                else:
                    if isinstance(task_result, SubAgentResult):
                        result = task_result
                        error = ""
                        error_class = ""
                    else:
                        error = "sub-agent task ended without a report"
                        error_class = "MissingSubAgentResult"
            elif result is None:
                error = "sub-agent task was cancelled before publishing a report"
                error_class = "CancelledError"
            else:
                error = ""
                error_class = ""

            if result is None:
                result = SubAgentResult(
                    question=str(getattr(entry.item, "question", "") or ""),
                    role_id=str(getattr(entry.item, "role_id", "") or ""),
                    final_content="",
                    success=False,
                    error=error,
                    error_class=error_class,
                    job_id=job_id,
                )
                _close_session_boundary_aborted(session)

            entry.result = result
            entry.completed_at = entry.completed_at or time.monotonic()
            entry.status = (
                "completed" if result.success
                else "aborted" if result.error_class == "CancelledError"
                else "failed"
            )
            if not any(item.job_id == job_id for item in session.pending_results):
                session.pending_results.append(result)
            if entry.submitted_at:
                session.last_task_elapsed_s = max(
                    0.0, entry.completed_at - entry.submitted_at,
                )
            if session.current_job_id == job_id:
                session.current_job_id = None
            if self._spawn_guard is not None:
                self._spawn_guard.release(job_id)
            sessions_to_drain.append(session)
            logger.warning(
                "Reconciled terminal sub-agent job without a published report: "
                "%s (%s)", job_id, result.error_class or entry.status,
            )

        for session in sessions_to_drain:
            await self._drain_session_queue(session)


    async def wait_any_session_detailed(
        self, task_id: str, *, timeout: float = 1800.0,
    ) -> SessionWaitOutcome:
        """Wait for one result and preserve why the wait ended."""
        started = time.monotonic()
        ready = self._pop_ready_result(task_id)
        if ready is not None:
            return SessionWaitOutcome(ready, "ready", time.monotonic() - started)

        await self._reconcile_terminal_session_jobs(task_id)
        ready = self._pop_ready_result(task_id)
        if ready is not None:
            return SessionWaitOutcome(ready, "ready", time.monotonic() - started)

        pending: set[asyncio.Task[SubAgentResult]] = set()
        for session in self._sessions.values():
            if session.task_id != task_id or session.current_job_id is None:
                continue
            entry = self._jobs.get(session.current_job_id)
            if (
                entry is not None
                and entry.status in ("submitted", "running")
                and entry.task is not None
                and not entry.task.done()
            ):
                pending.add(entry.task)

        if not pending:
            recovered = await self._reconcile_from_messages(task_id)
            reason = "ready" if recovered is not None else "no_pending"
            return SessionWaitOutcome(
                recovered, reason, time.monotonic() - started,
            )

        done, _ = await asyncio.wait(
            pending,
            timeout=max(0.0, float(timeout)),
            return_when=asyncio.FIRST_COMPLETED,
        )
        if done:
            await self._reconcile_terminal_session_jobs(task_id)
            ready = self._pop_ready_result(task_id)
            if ready is None:
                ready = await self._reconcile_from_messages(task_id)
            if ready is not None:
                return SessionWaitOutcome(
                    ready, "ready", time.monotonic() - started,
                )
        return SessionWaitOutcome(
            None,
            "unpublished" if done else "timeout",
            time.monotonic() - started,
        )

    async def wait_any_session(
        self, task_id: str, *, timeout: float = 1800.0,
    ) -> tuple[str, SubAgentResult] | None:
        """Return the next completed task, or ``None`` when no result arrives."""
        outcome = await self.wait_any_session_detailed(task_id, timeout=timeout)
        return outcome.result

    def _pop_ready_result(
        self, task_id: str,
    ) -> tuple[str, SubAgentResult] | None:
        for session in self._sessions.values():
            if session.task_id != task_id:
                continue
            if session.pending_results:
                result = session.pending_results.pop(0)
                self._mark_delivered(task_id, result.job_id)
                return session.session_id, result
        return None

    # ── L1-0: message-channel reconciliation ────────────────────────────

    def _mark_delivered(self, task_id: str, job_id: str) -> None:
        """Record that ``job_id``'s result has been handed to a caller."""
        if job_id:
            self._delivered_jobs.setdefault(task_id, set()).add(job_id)

    def _note_result_recipient(
        self, task_id: str, spawn_context: dict[str, Any] | None,
    ) -> None:
        """Remember who ``subtask_result`` messages for this task address."""
        if not spawn_context:
            return
        parent = str(spawn_context.get("parent_agent_id") or "").strip()
        if parent:
            self._result_recipients.setdefault(task_id, set()).add(parent)

    async def _discover_result_recipients(self, task_id: str) -> set[str]:
        """Recipients for ``task_id``, from memory or else from the log.

        In-process the dispatch path has already recorded them. After a
        restart it has not, and the only surviving record is the messages
        themselves — their ``to_agent`` is the recipient, so one scan of
        the task's message events recovers the set.
        """
        known = self._result_recipients.get(task_id)
        if known:
            return set(known)
        sink = self._event_sink()
        if sink is None or not hasattr(sink, "get_events"):
            return set()
        try:
            events = await sink.get_events(
                task_id, event_type=EventType.AGENT_MESSAGE,
            )
        except Exception:
            logger.debug("recipient discovery failed for %s", task_id, exc_info=True)
            return set()
        found = {
            str(event.to_agent)
            for event in events
            if getattr(event, "message_type", None) == SUBTASK_RESULT_MESSAGE_TYPE
            and getattr(event, "to_agent", None)
        }
        if found:
            self._result_recipients.setdefault(task_id, set()).update(found)
        return found

    async def _reconcile_from_messages(
        self, task_id: str,
    ) -> tuple[str, SubAgentResult] | None:
        """Recover one undelivered result from the message channel.

        Reads through ``AgentComm``'s type-isolated
        ``(agent, task, subtask_result)`` cursor, so repeated calls do not
        re-read the same results and unrelated status/control messages remain
        available to their own consumers. Skips job ids the in-memory path
        already delivered. Returns the same
        ``(session_id, result)`` shape as the in-memory path — callers and
        the ``<report>`` rendering downstream cannot tell the difference.

        One consume call drains a whole batch and moves the cursor past all
        of it, so every usable message is buffered and one is returned per
        call; otherwise a batch of N would deliver 1 and lose N-1.

        Entirely soft: no ``AgentComm``, no messages, or any read failure
        all fall back to today's ``None``.
        """
        buffered = self._recovered_results.get(task_id)
        if buffered:
            return buffered.popleft()

        # Lazy import: ``agent_comm`` is a sibling module and this keeps the
        # bus importable without it, matching how ``runtime.py`` reaches for
        # the same class.
        from agent_core.components.agent_bus.agent_comm import (
            AgentComm,
            EventStoreContractError,
        )

        comm = registry.get_optional(AgentComm)
        if comm is None:
            return None
        recipients = await self._discover_result_recipients(task_id)
        if not recipients:
            return None
        delivered = self._delivered_jobs.get(task_id, set())
        pending_recovered = self._recovered_results.setdefault(task_id, deque())
        for agent_id in sorted(recipients):
            try:
                events = await comm.consume(
                    agent_id,
                    task_id=task_id,
                    message_type=SUBTASK_RESULT_MESSAGE_TYPE,
                )
            except EventStoreContractError:
                # Not a transient read failure: every future read fails the
                # same way, so durable recovery is off until the store is
                # fixed. Debug level would hide that permanently.
                logger.error(
                    "durable result recovery is unavailable for %s/%s: the "
                    "configured event store does not stamp a total-order "
                    "ordinal on appended events",
                    task_id, agent_id, exc_info=True,
                )
                continue
            except Exception:
                logger.debug(
                    "message reconciliation failed for %s/%s",
                    task_id, agent_id, exc_info=True,
                )
                continue
            for event in events:
                payload = event.payload or {}
                if payload.get("message_type") != SUBTASK_RESULT_MESSAGE_TYPE:
                    continue
                content = payload.get("content") or {}
                job_id = str(content.get("correlation_id") or "")
                if not job_id or job_id in delivered:
                    continue
                raw = content.get("result")
                if not isinstance(raw, dict):
                    continue
                self._mark_delivered(task_id, job_id)
                delivered = self._delivered_jobs.get(task_id, set())
                pending_recovered.append((
                    str(content.get("session_id") or ""),
                    _message_content_to_result(raw),
                ))
                logger.info(
                    "Recovered sub-agent result %s from the message channel",
                    job_id,
                )
        return pending_recovered.popleft() if pending_recovered else None

    def get_session(self, session_id: str) -> SubAgentSession | None:
        return self._sessions.get(session_id)

    def list_sessions_for_task(self, task_id: str) -> list[SubAgentSession]:
        return [s for s in self._sessions.values() if s.task_id == task_id]

    def current_job_metadata(self, session_id: str) -> dict[str, Any]:
        """Task metadata of the job this session is running now.

        Empty when the session is unknown, idle, or its job entry has already
        been reaped. Callers use it to tell *what kind* of work a stop or a
        cancellation would interrupt — ``can_publish``, for one, marks the job
        that produces the run's deliverable. A copy, so a caller inspecting a
        live job cannot mutate the dispatched task's own metadata.
        """
        session = self._sessions.get(session_id)
        if session is None or session.current_job_id is None:
            return {}
        entry = self._jobs.get(session.current_job_id)
        if entry is None:
            return {}
        return dict(getattr(entry.item, "metadata", None) or {})

    def describe_sessions_for_task(self, task_id: str) -> list[dict[str, Any]]:
        """Return a small, UI-safe snapshot of every sub-agent session.

        The snapshot deliberately excludes prompts, reports, and model data;
        it is suitable for frequent progress rendering while collect_reports
        blocks. Durations use the job's monotonic submission clock.
        """
        now = time.monotonic()
        snapshots: list[dict[str, Any]] = []
        for session in self.list_sessions_for_task(task_id):
            entry = (
                self._jobs.get(session.current_job_id)
                if session.current_job_id is not None else None
            )
            if entry is not None:
                status = entry.status
                active = entry.status in ("queued", "submitted", "running")
                # A terminal entry can still be the session's ``current_job_id``
                # for the moment ``_run_and_finalize`` spends in its finally
                # block; measure it to ``completed_at`` so that window does not
                # publish a duration that keeps growing.
                until = now if active else (entry.completed_at or now)
                elapsed = (
                    max(0.0, until - entry.submitted_at)
                    if entry.submitted_at else 0.0
                )
            else:
                active = False
                # ``last_task_elapsed_s`` describes work that finished, so it
                # only belongs to the states that mean "finished". A session
                # merely holding a queue has not started that task yet.
                elapsed = 0.0
                if session.pending_results:
                    status = "ready"
                    elapsed = session.last_task_elapsed_s
                elif session.pending_tasks:
                    status = "queued"
                elif session.total_task_count:
                    status = "idle"
                    elapsed = session.last_task_elapsed_s
                else:
                    status = "unassigned"
            snapshots.append({
                "session_id": session.session_id,
                "name": session.name,
                "role_id": session.role_id,
                "status": status,
                # ``active`` says whether ``elapsed_s`` is still counting.
                # Without it a consumer cannot tell "ran for 12s and stopped"
                # from "has been running 12s", and re-deriving the duration
                # from its own clock makes finished workers tick forever.
                "active": active,
                "elapsed_s": elapsed,
                "queued": len(session.pending_tasks),
                "completed": len(session.pending_results),
                "events": [dict(event) for event in session.activity_events],
            })
        return snapshots


    def accumulate_task_metadata(
        self,
        task_id: str,
        **payload: list[dict[str, Any]] | None,
    ) -> None:
        """Merge named list-payloads into the task's metadata pool.

        Each keyword argument names a bucket (e.g. ``evidence_cards=[...],
        assertions=[...]``). Empty or ``None`` payloads are skipped so
        callers can pass optional harvests without guarding each one.

        The pool is drained via ``drain_task_metadata`` when the caller
        (typically a pipeline node) wants to hand the corpus downstream.
        """
        merged = {k: v for k, v in payload.items() if v}
        if not merged:
            return
        bucket = self._task_aggregates.setdefault(task_id, {})
        for key, items in merged.items():
            bucket.setdefault(key, []).extend(items)

    def drain_task_metadata(
        self, task_id: str,
    ) -> dict[str, list[dict[str, Any]]]:
        """Return + clear accumulated metadata lists for a task.

        Returns an empty dict if nothing was accumulated; callers that
        expect specific keys should use ``.get(key, [])``.
        """
        return self._task_aggregates.pop(task_id, {})

    async def cleanup_task(
        self, task_id: str, *, cancel_timeout_s: float = 10.0,
    ) -> int:
        """Cancel all running session tasks under ``task_id`` and drop sessions.

        If a sub-agent's asyncio.Task cannot be cancelled within
        ``cancel_timeout_s`` (e.g. blocked in a non-cancellable subprocess
        or a long socket read), we detach it, log a warning, and move on.
        That prevents the pipeline's post-loop cleanup from hanging
        indefinitely on stuck sub-agents, which previously caused the
        ``main_agent → report`` transition to stall until the eval
        driver killed the task.

        Returns the number of sessions cleaned up. Also clears any residual
        aggregate pool so re-runs don't carry over state.

        Rescue: ``session.pending_results`` are merged into
        ``_task_aggregates`` via ``accumulate_task_metadata`` before
        sessions are dropped, so a post-cleanup ``drain_task_metadata``
        still surfaces sub-agent reports the main agent never claimed
        (e.g. ``force_final_answer`` short-circuited a slow sub-agent).
        """
        self._task_aggregates.pop(task_id, None)

        rescued_evidence: list[dict[str, Any]] = []
        rescued_assertions: list[dict[str, Any]] = []

        def rescue_pending_results(session: SubAgentSession) -> None:
            while session.pending_results:
                result = session.pending_results.pop(0)
                md = result.metadata or {}
                rescued_evidence.extend(md.get("evidence_cards") or [])
                rescued_assertions.extend(md.get("assertions") or [])

        def abort_pending_tasks(session: SubAgentSession) -> None:
            while session.pending_tasks:
                pending = session.pending_tasks.popleft()
                self._mark_job_aborted(
                    pending.job_id,
                    self._jobs.get(pending.job_id),
                )

        cleaned = 0
        stuck: list[str] = []
        for sid in list(self._sessions):
            session = self._sessions.get(sid)
            if session is None or session.task_id != task_id:
                continue
            rescue_pending_results(session)
            # Prevent the running task's ``finally`` from draining queued
            # successors while cleanup is tearing this session down.
            abort_pending_tasks(session)
            if session.current_job_id is not None:
                current_job_id = session.current_job_id
                entry = self._jobs.get(session.current_job_id)
                if entry is not None and entry.status in (
                    "submitted", "running",
                ) and entry.task is not None:
                    entry.task.cancel()
                    try:
                        await asyncio.wait_for(
                            entry.task, timeout=cancel_timeout_s,
                        )
                    except (TimeoutError, asyncio.CancelledError):
                        if not entry.task.done():
                            stuck.append(current_job_id)
                    except Exception:
                        pass
                    if entry.status not in ("completed", "failed", "aborted"):
                        self._mark_job_aborted(current_job_id, entry)
                elif entry is not None and entry.status not in (
                    "completed", "failed", "aborted",
                ):
                    self._mark_job_aborted(current_job_id, entry)
            # A cancelled job can catch CancelledError, build a
            # SubAgentResult, and append it to pending_results while
            # cleanup is awaiting entry.task. Rescue again immediately
            # before dropping the session so that metadata is not lost.
            rescue_pending_results(session)
            del self._sessions[sid]
            cleaned += 1
        if rescued_evidence or rescued_assertions:
            self.accumulate_task_metadata(
                task_id,
                evidence_cards=rescued_evidence,
                assertions=rescued_assertions,
            )
            logger.info(
                "cleanup_task: rescued %d evidence + %d assertions from "
                "unconsumed pending_results (task=%s)",
                len(rescued_evidence), len(rescued_assertions), task_id,
            )
        if cleaned:
            logger.info(
                "cleanup_task: dropped %d sessions for %s", cleaned, task_id,
            )
        if stuck:
            logger.warning(
                "cleanup_task: %d session job(s) did not cancel within %ss — "
                "detaching so pipeline can progress: %s",
                len(stuck), cancel_timeout_s, stuck,
            )
        # These maps are task-scoped runtime acceleration only. cleanup_task
        # is the terminal lifecycle boundary used by the SDK/API workflows;
        # retaining their keys would make a long-lived AgentBus grow once per
        # completed task. Durable messages remain in EventStore and can still
        # be replayed by a fresh runtime after a process restart.
        self._delivered_jobs.pop(task_id, None)
        self._result_recipients.pop(task_id, None)
        self._recovered_results.pop(task_id, None)
        return cleaned

    # ── dispatch_parallel: backward-compatible convenience ──────────────

    async def dispatch_parallel(
        self,
        parent_task_id: str,
        items: list[SubTask],
        shared_evidence: SharedArtifactPool | None = None,
        max_turns: int = 8,
        current_depth: int = 0,
        max_depth: int = 2,
        runtime_spec: SubAgentRuntimeSpec | None = None,
    ) -> list[SubAgentResult]:
        """Submit all items and collect all results.

        Backward-compatible wrapper around submit + collect.
        Existing callers (fan_out_node, tests) work unchanged.
        """
        if current_depth >= max_depth:
            raise DepthLimitExceeded(
                f"Sub-agent depth limit exceeded: "
                f"current_depth={current_depth}, "
                f"max_depth={max_depth}"
            )

        if not items:
            return []

        evidence_pool = shared_evidence or SharedArtifactPool()

        event_store = self._event_sink()
        t0 = time.monotonic()

        if event_store:
            await event_store.append(
                task_id=parent_task_id,
                event_type=EventType.AGENT_ACTION,
                payload={
                    "trace_type": "agent_delegated",
                    "parent_task_id": parent_task_id,
                    "subtasks": [
                        {"question": item.question, "role": item.role_id}
                        for item in items
                    ],
                    "depth": current_depth + 1,
                },
                agent_role="system",
            )

        # Submit all
        job_ids: list[str] = []
        for item in items:
            jid = await self.submit(
                parent_task_id,
                item,
                shared_evidence=evidence_pool,
                max_turns=max_turns,
                current_depth=current_depth,
                max_depth=max_depth,
                runtime_spec=runtime_spec,
            )
            job_ids.append(jid)

        collect_result = await self.collect(job_ids, timeout=1800)

        if event_store:
            all_results = collect_result.completed + collect_result.failed
            await event_store.append(
                task_id=parent_task_id,
                event_type=EventType.AGENT_ACTION,
                payload={
                    "trace_type": "agent_completed",
                    "parent_task_id": parent_task_id,
                    "results_count": len(all_results),
                    "successful_results": sum(
                        1 for r in all_results if r.success
                    ),
                    "total_evidence": evidence_pool.count(),
                    "elapsed_ms": int(
                        (time.monotonic() - t0) * 1000
                    ),
                },
                agent_role="system",
            )

        return collect_result.completed + collect_result.failed

    # ── rehydration: recover from restart ───────────────────────────────

    async def rehydrate_active_jobs(self) -> int:
        """Mark jobs whose asyncio.Task was lost across a restart as orphaned."""
        return await _rehydrate_orphaned_jobs(
            self._jobs, event_sink=self._event_sink(),
        )
