# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportAttributeAccessIssue=false
"""Generic runtime helpers for AgentBus.

This module keeps the generic default config / observer / result-adaptation
helpers out of ``agent_bus.py`` so the bus focuses on job and session
lifecycle management. Telemetry emission and post-restart job rehydration
also live here — they are mostly glue to ``EventStore`` and share no
state with the ``AgentBus`` class itself.
"""

from __future__ import annotations

import logging
from typing import Any

from agent_core.components.agent_bus.models import JobEntry, SubAgentResult, SubAgentSession
from agent_core.events import EventType
from agent_core.messages import assistant_msg
from agent_core.protocols import EventSink
from agent_core.runtime.loop.message_trimmer import find_final_assistant
from agent_core.runtime.registries import services as registry

logger = logging.getLogger(__name__)


def build_default_subagent_loop_config(
    job_id: str,
    item: Any,
    max_turns: int,
) -> Any:
    """Build the generic default LoopConfig for async sub-agent jobs."""
    from agent_core.loop_types import LoopConfig

    return LoopConfig(
        max_turns=max_turns,
        task_id=job_id,
        role_id=item.role_id,
    )


def build_default_subagent_observers(
    *,
    event_store: Any | None,
    job_id: str,
) -> list[Any]:
    """Build the generic default observer stack for async sub-agent jobs."""
    del event_store, job_id
    return []


def adapt_default_subagent_result(
    agent_result: Any,
    job_id: str,
    item: Any,
) -> Any:
    """Adapt an AgentLoopResult into a generic SubAgentResult.

    The kernel default forwards the full loop metadata bag untouched —
    any domain-specific fields (evidence_cards, assertions, …) live
    inside ``metadata`` and only appear when workflow observers
    populated them.
    """
    metadata = dict(getattr(agent_result, "metadata", {}) or {})
    return SubAgentResult(
        question=item.question,
        role_id=item.role_id,
        final_content=getattr(agent_result, "final_content", "") or "",
        success=True,
        job_id=job_id,
        metadata=metadata,
    )


def adapt_default_session_result(
    agent_result: Any,
    job_id: str,
    session: Any,
    task_prompt: str,
) -> Any:
    """Adapt a session loop result into a generic SubAgentResult."""
    metadata = dict(getattr(agent_result, "metadata", {}) or {})
    return SubAgentResult(
        question=task_prompt,
        role_id=session.role_id,
        final_content=getattr(agent_result, "final_content", "") or "",
        success=True,
        job_id=job_id,
        metadata=metadata,
    )


def build_session_loop_config(
    session: Any,
    job_id: str,
    max_turns: int,
) -> Any:
    """Build LoopConfig for a session task."""
    from agent_core.loop_types import LoopConfig

    kwargs: dict[str, Any] = {
        "max_turns": max_turns,
        "task_id": session.task_id,
        "role_id": session.role_id,
        "tool_result_max_chars": session.tool_result_max_chars,
    }
    if getattr(session, "llm_timeout", None) is not None:
        kwargs["llm_timeout"] = session.llm_timeout
    return LoopConfig(**kwargs)


def resolve_session_observers(
    observers: list[Any] | None,
    session: Any | None = None,
) -> list[Any]:
    """Resolve observers for a session task.

    The kernel default is intentionally empty. Workflow-specific observer
    stacks should be injected through ``runtime_spec`` or explicit callers.
    """
    if observers is not None:
        return observers
    del session
    return []


def close_session_boundary_aborted(session: Any) -> None:
    """Close the current session task boundary when cancelled/errored.

    Maintains the SubAgentSession boundary invariant: when no clean
    AIMessage exists in the in-flight slice, append an abort stub so
    the trimmer can still surface "task #N happened, here's what we
    have" when the agent is reused.
    """
    if not session.task_boundaries:
        return
    start, end = session.task_boundaries[-1]
    if end is not None:
        return

    if find_final_assistant(
        session.messages, start + 1, len(session.messages) - 1,
    ) is None:
        session.messages.append(
            assistant_msg("[task aborted before producing a final answer]"),
        )

    tail = max(start, len(session.messages) - 1)
    session.task_boundaries[-1] = (start, tail)


async def emit_session_task_submitted(
    session: SubAgentSession,
    job_id: str,
    task_prompt: str,
    *,
    event_sink: Any = None,
) -> None:
    """Record the submit side of a session-task lifecycle to EventStore.

    ``event_sink`` is an optional ``core.protocols.EventSink`` injected
    by ``AgentBus``. Empty falls back to the
    global registry lookup so existing direct callers keep working.
    """
    event_store = event_sink if event_sink is not None else registry.get_optional(EventSink)
    if event_store is None:
        return
    await event_store.append(
        task_id=session.task_id,
        event_type=EventType.AGENT_ACTION,
        payload={
            "trace_type": "session_task_submitted",
            "session_id": session.session_id,
            "job_id": job_id,
            "task_count": session.total_task_count,
            "is_reuse": session.total_task_count > 1,
            "role_id": session.role_id,
            "agent": session.name,
            "action": "assign_task",
            "detail": (
                f"{'Reusing' if session.total_task_count > 1 else 'Starting'} "
                f"sub-agent '{session.name}' "
                f"(task #{session.total_task_count}): "
                f"{task_prompt[:140]}"
            ),
        },
        agent_role="system",
    )


def safe_metadata(raw_result: Any) -> dict[str, Any]:
    """Best-effort extraction of ``metadata`` from a partial AgentLoopResult.

    Called from exception handlers where ``raw_result`` may be ``None``
    (agent loop never completed) or a completed result whose downstream
    adapter raised. Never raises — returns ``{}`` on any oddity.

    The evidence harvested inside an agent loop is the most expensive
    side-effect of a research run (it cost real web_search/web_fetch
    API calls). Losing it because a post-loop bookkeeping step
    crashed is the worst possible UX: the user still paid for the
    calls but sees nothing in the evidence graph.
    """
    if raw_result is None:
        return {}
    try:
        meta = getattr(raw_result, "metadata", None)
        if isinstance(meta, dict):
            return dict(meta)
    except Exception:
        pass
    return {}


def safe_final_content(raw_result: Any) -> str:
    """Safely pull ``final_content`` off a possibly-None result."""
    if raw_result is None:
        return ""
    try:
        value = getattr(raw_result, "final_content", "") or ""
        return str(value)
    except Exception:
        return ""


SUBTASK_RESULT_MESSAGE_TYPE = "subtask_result"
"""``AgentMessage.message_type`` carrying a sub-agent's result home.

Open string by design — ``models/agent_message.py`` documents that any
agent may define its own types, and L1-3 will add a registry that
validates them rather than a hardcoded enum.
"""


def result_to_message_content(result: SubAgentResult) -> dict[str, Any]:
    """Serialise a ``SubAgentResult`` for transport, losslessly.

    All eight fields (``models.py``'s dataclass) travel, because the
    receiving side rebuilds the object and hands it to the *same*
    ``fan_in.process_collected``. Dropping any of them would change the
    rendered ``<report>``: ``final_content`` is the body, ``success`` and
    ``error_class`` drive ``classify_completion``, and ``metadata`` holds
    both ``stopped_by`` (the status label) and the harvested evidence.
    """
    return {
        "question": result.question,
        "role_id": result.role_id,
        "final_content": result.final_content,
        "success": result.success,
        "error": result.error,
        "error_class": result.error_class,
        "job_id": result.job_id,
        "metadata": dict(result.metadata or {}),
    }


def message_content_to_result(content: dict[str, Any]) -> SubAgentResult:
    """Rebuild a ``SubAgentResult`` from :func:`result_to_message_content`.

    Defensive about every field: a message may have been written by an
    older build, and a partially-shaped payload should still produce a
    renderable report rather than raising inside the collect path.
    """
    return SubAgentResult(
        question=str(content.get("question") or ""),
        role_id=str(content.get("role_id") or ""),
        final_content=str(content.get("final_content") or ""),
        success=bool(content.get("success")),
        error=content.get("error") or None,
        error_class=content.get("error_class") or None,
        job_id=str(content.get("job_id") or ""),
        metadata=dict(content.get("metadata") or {}),
    )


async def send_subtask_result_message(
    session: SubAgentSession,
    job_id: str,
    result: SubAgentResult,
    spawn_context: dict[str, Any] | None,
) -> None:
    """Hand a sub-agent's result back to its parent as a message.

    Why this exists (L1-0): the result used to travel only through
    ``AgentBus`` in-process state — ``entry.result`` plus a destructive
    ``session.pending_results`` pop. The event stream recorded the
    *fact* of completion (``session_task_completed``) but never the
    content, so a hand-back could not be replayed, and a double drain
    had no idempotency key. Sending it as a real message makes the
    hand-back addressable (``to_agent``), correlated
    (``correlation_id == job_id``), causally linked (``parent_id`` is the
    ``delegate_subtask`` tool call that spawned it), and durable.

    The join keys are **reused, not invented**: ``spawn_context`` already
    carries the parent→sub lineage that Log Schema v1 §2.4 requires, so
    trace-side and message-side reconstruction stay on one set of keys.

    ``spawn_context`` values are passed through **verbatim**, including an
    empty ``parent_run_id`` — agent_team's coordinator carries no
    ``run_id`` in scope metadata, so ``assign_task`` stamps ``""`` (v1-doc
    P7). The trace layer normalises that to ``run_1`` at its v1-translation
    boundary (``worker_trace._sub_run_to_v1_sub_agent``); doing it here too
    would put the same rule in two places, and inventing a value in the
    kernel would leak a trace-layer convention into the bus. Join on
    ``to_agent`` (the parent agent id), which is always populated —
    ``parent_run_id`` is only reliable after that normalisation.

    Soft-dependency and fully swallowed: with no ``AgentComm``
    registered, or no ``spawn_context`` to address, behaviour is exactly
    what it was before this function existed.
    """
    if not spawn_context:
        return
    parent_agent_id = str(spawn_context.get("parent_agent_id") or "").strip()
    if not parent_agent_id:
        return
    try:
        from agent_core.components.agent_bus.agent_comm import AgentComm
        from agent_core.models.agent_message import AgentMessage

        comm = registry.get_optional(AgentComm)
        if comm is None:
            return
        await comm.send(
            AgentMessage(
                task_id=session.task_id,
                from_agent=session.name,
                to_agent=parent_agent_id,
                message_type=SUBTASK_RESULT_MESSAGE_TYPE,
                # The spawning tool call — the causal parent of this reply.
                parent_id=(
                    str(spawn_context.get("spawned_by_tool_call_id") or "") or None
                ),
                content={
                    # AgentComm lifts this onto the KernelEvent column, which
                    # is what makes the message queryable by job.
                    "correlation_id": job_id,
                    "session_id": session.session_id,
                    "result": result_to_message_content(result),
                    "spawn_context": {
                        "parent_run_id": spawn_context.get("parent_run_id") or "",
                        "parent_turn": spawn_context.get("parent_turn") or 0,
                        "spawned_by_llm_call_id": (
                            spawn_context.get("spawned_by_llm_call_id") or ""
                        ),
                    },
                },
            ),
        )
    except Exception:
        # Same discipline as the telemetry below: the delivery channel is
        # not allowed to break the session lifecycle.
        logger.debug("subtask_result message not sent for %s", job_id, exc_info=True)


async def emit_session_task_completed(
    session: SubAgentSession,
    job_id: str,
    result: SubAgentResult,
    *,
    event_sink: Any = None,
    spawn_context: dict[str, Any] | None = None,
) -> None:
    """Record the completion side of a session-task lifecycle.

    Swallows failures — telemetry must not break the session lifecycle.
    List-valued metadata entries are summarised as ``{key}_count`` to
    keep the event payload compact regardless of workflow-specific
    metadata shape.

    ``event_sink`` accepts a ``core.protocols.EventSink`` injected by
    ``AgentBus``; falls back to the global
    registry when omitted.

    ``spawn_context`` is the delegation lineage recorded at dispatch
    (Log Schema v1 §2.4). When present, and when an ``AgentComm`` is
    registered, the result is **also** sent as a ``subtask_result``
    message so the hand-back is addressable, correlated and replayable
    — see :func:`send_subtask_result_message`.
    """
    # Message first: it has its own store via AgentComm, so an absent
    # telemetry sink must not suppress it.
    await send_subtask_result_message(session, job_id, result, spawn_context)
    ev_store = event_sink if event_sink is not None else registry.get_optional(EventSink)
    if ev_store is None:
        return
    try:
        metadata_counts = {
            f"{k}_count": len(v)
            for k, v in result.metadata.items()
            if isinstance(v, list)
        }
        detail_parts = [
            f"{len(v)} {k}"
            for k, v in result.metadata.items()
            if isinstance(v, list) and v
        ]
        detail = f"'{session.name}' returned"
        if detail_parts:
            detail += " " + ", ".join(detail_parts)
        if not result.success and result.error:
            # Make the failure legible in both the event stream and
            # any downstream log summarizer — the previous shape left
            # "success: false" with no reason and every sub-agent
            # failure looked identical.
            cls = (result.error_class or "").strip()
            err_short = str(result.error)[:200]
            detail += f" (failed: {cls}: {err_short})" if cls else f" (failed: {err_short})"
        payload: dict[str, Any] = {
            "trace_type": "session_task_completed",
            "session_id": session.session_id,
            "job_id": job_id,
            "success": result.success,
            "agent": session.name,
            "action": "report_returned",
            "detail": detail,
            **metadata_counts,
        }
        if result.error:
            payload["error"] = str(result.error)[:500]
        if result.error_class:
            payload["error_class"] = result.error_class
        await ev_store.append(
            task_id=session.task_id,
            event_type=EventType.AGENT_ACTION,
            payload=payload,
            agent_role="system",
        )
    except Exception:
        pass


async def rehydrate_orphaned_jobs(
    jobs: dict[str, JobEntry], *, event_sink: Any = None,
) -> int:
    """Recover job registry after process restart.

    Reads ``EventStore`` for ``agent_submitted`` events that have no
    matching ``agent_completed`` / ``agent_aborted`` / ``agent_collected``
    event. Marks those orphaned jobs as failed (their asyncio.Tasks are
    gone) by appending an ``agent_orphaned`` event for each.

    Returns the number of orphaned jobs recorded. No-op if no EventStore
    is registered. ``event_sink`` lets ``AgentBus`` thread its
    constructor-injected sink through; empty
    falls back to the global registry.
    """
    event_store = event_sink if event_sink is not None else registry.get_optional(EventSink)
    if event_store is None:
        return 0

    all_events = await event_store.get_all_events(
        event_type=EventType.AGENT_ACTION,
    )

    submitted: dict[str, dict[str, Any]] = {}
    completed_ids: set[str] = set()

    for evt in all_events:
        payload = evt.payload or {}
        trace_type = payload.get("trace_type", "")
        job_id = payload.get("job_id", "")

        if trace_type == "agent_submitted" and job_id:
            submitted[job_id] = payload
        elif trace_type in (
            "agent_completed", "agent_aborted", "agent_collected",
        ):
            if job_id:
                completed_ids.add(job_id)
            for jid in payload.get("job_ids", []):
                completed_ids.add(jid)

    orphaned = 0
    for job_id, payload in submitted.items():
        if job_id in completed_ids:
            continue
        if job_id in jobs:
            continue  # already tracked (shouldn't happen post-restart)

        orphaned += 1
        parent_task_id = payload.get("parent_task_id", "")
        logger.warning(
            "Rehydration: marking orphaned job %s as failed (parent=%s)",
            job_id, parent_task_id,
        )
        await event_store.append(
            task_id=parent_task_id,
            event_type=EventType.AGENT_ACTION,
            payload={
                "trace_type": "agent_orphaned",
                "job_id": job_id,
                "reason": "process_restart",
            },
            agent_role="system",
        )

    if orphaned:
        logger.info(
            "Rehydration: %d orphaned jobs marked as failed", orphaned,
        )
    return orphaned
