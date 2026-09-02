"""Session-task queue: behaviour around assign-while-busy.

Sessions execute serially (the message list / boundary list cannot be
mutated by two concurrent runs), but a second submit must NOT raise —
it queues FIFO and runs as soon as the predecessor finalises. Without
this, BrowseComp main agents see "already has a running task" errors
and fall back to spawning near-duplicate sessions, killing the reuse
benefit. Trial trace evidence: 38/112 trials hit the old error and ~50%
of those rejections led to a fresh-session spawn instead of reuse.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from agent_core.components.agent_bus import AgentBus
from agent_core.components.agent_bus.spawn_guard import SpawnGuard
from agent_core.loop_types import AgentLoopResult
from agent_core.messages import assistant_msg, user_msg
from agent_core.models.task_budget import TaskBudget
from agent_core.runtime.registries import services as registry


@pytest.fixture(autouse=True)
def _clear_registry():
    registry.clear()
    yield
    registry.clear()


async def _bus_session() -> tuple[AgentBus, str]:
    bus = AgentBus()
    sid = await bus.create_session(
        task_id="t-q",
        name="lit",
        role_id="researcher",
        system_prompt="You are a researcher.",
        llm_override=MagicMock(),
        tools_override=[],
    )
    return bus, sid


def _gated_loop_factory(gate: asyncio.Event):
    async def loop(**kwargs):
        await gate.wait()
        im = list(kwargs.get("initial_messages") or [])
        um = kwargs.get("user_message", "")
        return AgentLoopResult(
            messages=[*im, user_msg(um), assistant_msg(f"r:{um}")],
            final_content=f"r:{um}",
            metadata={},
        )
    return loop


# ──────────────────────────────────────────────────────────────────────────
# FIFO + multi-queue
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_three_queued_drain_in_fifo_order(monkeypatch):
    gate = asyncio.Event()
    monkeypatch.setattr(
        "agent_core.components.agent_bus.bus.run_agent_loop",
        _gated_loop_factory(gate),
    )
    bus, sid = await _bus_session()

    j1 = await bus.submit_task_to_session(sid, "T1")
    j2 = await bus.submit_task_to_session(sid, "T2")
    j3 = await bus.submit_task_to_session(sid, "T3")

    session = bus.get_session(sid)
    assert session.current_job_id == j1
    assert [p.job_id for p in session.pending_tasks] == [j2, j3]
    assert bus.get_job_status(j2) == "queued"
    assert bus.get_job_status(j3) == "queued"

    gate.set()
    results = []
    for _ in range(3):
        r = await bus.wait_any_session("t-q", timeout=2)
        assert r is not None
        results.append(r[1].final_content)

    # FIFO order preserved.
    assert results == ["r:T1", "r:T2", "r:T3"]
    assert not session.pending_tasks
    assert session.current_job_id is None


# ──────────────────────────────────────────────────────────────────────────
# Drain still happens after current task FAILS
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_queue_drains_when_current_fails(monkeypatch):
    """A failed run still triggers queue drain — otherwise a single
    flaky task would strand all queued follow-ups."""
    fail_gate = asyncio.Event()
    success_gate = asyncio.Event()
    seen: list[str] = []

    async def loop(**kwargs):
        um = kwargs.get("user_message", "")
        seen.append(um)
        if um == "T1":
            await fail_gate.wait()
            raise RuntimeError("inner loop boom")
        await success_gate.wait()
        im = list(kwargs.get("initial_messages") or [])
        return AgentLoopResult(
            messages=[*im, user_msg(um), assistant_msg(f"r:{um}")],
            final_content=f"r:{um}",
            metadata={},
        )

    monkeypatch.setattr(
        "agent_core.components.agent_bus.bus.run_agent_loop", loop,
    )
    bus, sid = await _bus_session()

    j1 = await bus.submit_task_to_session(sid, "T1")
    j2 = await bus.submit_task_to_session(sid, "T2")

    fail_gate.set()
    success_gate.set()

    # Drain via wait_any_session twice.
    r1 = await bus.wait_any_session("t-q", timeout=2)
    r2 = await bus.wait_any_session("t-q", timeout=2)
    by_job = {r[1].job_id: r[1] for r in (r1, r2)}

    assert by_job[j1].success is False
    assert "inner loop boom" in (by_job[j1].error or "")
    assert by_job[j2].success is True
    assert by_job[j2].final_content == "r:T2"
    assert seen == ["T1", "T2"]


# ──────────────────────────────────────────────────────────────────────────
# Aborting a queued (not-yet-running) job
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_abort_queued_job_purges_from_queue(monkeypatch):
    gate = asyncio.Event()
    monkeypatch.setattr(
        "agent_core.components.agent_bus.bus.run_agent_loop",
        _gated_loop_factory(gate),
    )
    bus, sid = await _bus_session()

    await bus.submit_task_to_session(sid, "T1")
    j2 = await bus.submit_task_to_session(sid, "T2")
    j3 = await bus.submit_task_to_session(sid, "T3")

    final = await bus.abort(j2)
    assert final == "aborted"
    session = bus.get_session(sid)
    assert [p.job_id for p in session.pending_tasks] == [j3]
    assert bus.get_job_status(j2) == "aborted"
    # Aborted entry should produce a synthetic "aborted" SubAgentResult.
    aborted_entry = bus._jobs[j2]
    assert aborted_entry.result is not None
    assert aborted_entry.result.success is False
    assert aborted_entry.result.error == "aborted"

    gate.set()
    # Only j1 + j3 should surface; j2 must not.
    r1 = await bus.wait_any_session("t-q", timeout=2)
    r2 = await bus.wait_any_session("t-q", timeout=2)
    surfaced = {r[1].final_content for r in (r1, r2)}
    assert surfaced == {"r:T1", "r:T3"}
    # Third call returns None (everything drained).
    assert await bus.wait_any_session("t-q", timeout=0.1) is None


@pytest.mark.asyncio
async def test_cancel_for_parent_clears_running_and_queued(monkeypatch):
    gate = asyncio.Event()
    monkeypatch.setattr(
        "agent_core.components.agent_bus.bus.run_agent_loop",
        _gated_loop_factory(gate),
    )
    bus, sid = await _bus_session()

    j1 = await bus.submit_task_to_session(sid, "T1")
    j2 = await bus.submit_task_to_session(sid, "T2")
    j3 = await bus.submit_task_to_session(sid, "T3")

    cancelled = await bus.cancel_for_parent("t-q")
    assert cancelled == 3
    assert bus.get_job_status(j1) == "aborted"
    assert bus.get_job_status(j2) == "aborted"
    assert bus.get_job_status(j3) == "aborted"
    assert not bus.get_session(sid).pending_tasks
    # No surviving asyncio tasks expected — gate-set should be a no-op.
    gate.set()
    await asyncio.sleep(0)


# ──────────────────────────────────────────────────────────────────────────
# Status reporting + spawn guard
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_collect_treats_queued_session_job_as_pending(monkeypatch):
    """Queued session tasks have no asyncio.Task yet, so collect() must
    report them as pending rather than passing None into asyncio.wait."""
    gate = asyncio.Event()
    monkeypatch.setattr(
        "agent_core.components.agent_bus.bus.run_agent_loop",
        _gated_loop_factory(gate),
    )
    bus, sid = await _bus_session()

    await bus.submit_task_to_session(sid, "T1")
    j2 = await bus.submit_task_to_session(sid, "T2")

    result = await bus.collect([j2], timeout=0.01)

    assert result.pending == [j2]
    assert result.completed == []
    assert result.failed == []

    gate.set()
    for _ in range(2):
        await bus.wait_any_session("t-q", timeout=2)


@pytest.mark.asyncio
async def test_session_state_running_when_only_queued(monkeypatch):
    """If a task is queued (no current_job_id yet), the swarm is not
    'all_collected' — collect_reports must keep waiting."""
    from agent_core.components.agent_bus.fan_in import session_state

    gate = asyncio.Event()
    monkeypatch.setattr(
        "agent_core.components.agent_bus.bus.run_agent_loop",
        _gated_loop_factory(gate),
    )
    bus, sid = await _bus_session()
    await bus.submit_task_to_session(sid, "T1")
    await bus.submit_task_to_session(sid, "T2")

    sessions = bus.list_sessions_for_task("t-q")
    # T1 running + T2 queued → state should be "running".
    assert session_state(sessions) == "running"

    gate.set()
    # Drain so pytest doesn't warn about pending tasks.
    for _ in range(2):
        await bus.wait_any_session("t-q", timeout=2)


@pytest.mark.asyncio
async def test_queued_jobs_release_guard_reservation_on_abort(monkeypatch):
    """Queued tasks reserve budget at submit time. Aborting must release
    it — otherwise repeated submit/abort cycles would silently exhaust
    the task budget."""
    gate = asyncio.Event()
    monkeypatch.setattr(
        "agent_core.components.agent_bus.bus.run_agent_loop",
        _gated_loop_factory(gate),
    )

    bus = AgentBus()
    budget = TaskBudget(max_tokens=10_000, max_depth=3, max_parallel=4)
    guard = SpawnGuard(budget=budget, sub_agent_timeout_s=5)
    bus.set_spawn_guard(guard)

    sid = await bus.create_session(
        task_id="t-budget",
        name="lit",
        role_id="researcher",
        system_prompt="You are a researcher.",
        llm_override=MagicMock(),
        tools_override=[],
    )

    j1 = await bus.submit_task_to_session(sid, "T1", estimated_tokens=3_000)
    j2 = await bus.submit_task_to_session(sid, "T2", estimated_tokens=3_000)
    reserved_before = guard.tokens_reserved
    assert reserved_before >= 6_000

    await bus.abort(j2)
    # Aborting the queued j2 must drop its reservation.
    assert guard.tokens_reserved < reserved_before

    gate.set()
    await bus.wait_any_session("t-budget", timeout=2)
    _ = j1  # silence unused


@pytest.mark.asyncio
async def test_cleanup_task_aborts_queued_jobs_and_releases_guard(monkeypatch):
    """cleanup_task drops sessions at task end. Queued jobs must be
    explicitly aborted first; otherwise their SpawnGuard reservations leak
    after the session object is deleted."""
    gate = asyncio.Event()
    monkeypatch.setattr(
        "agent_core.components.agent_bus.bus.run_agent_loop",
        _gated_loop_factory(gate),
    )

    bus = AgentBus()
    guard = SpawnGuard(
        budget=TaskBudget(max_tokens=10_000, max_depth=3, max_parallel=2),
        sub_agent_timeout_s=5,
    )
    bus.set_spawn_guard(guard)

    sid = await bus.create_session(
        task_id="t-clean",
        name="lit",
        role_id="researcher",
        system_prompt="You are a researcher.",
        llm_override=MagicMock(),
        tools_override=[],
    )

    j1 = await bus.submit_task_to_session(sid, "T1", estimated_tokens=3_000)
    j2 = await bus.submit_task_to_session(sid, "T2", estimated_tokens=3_000)
    assert bus.get_job_status(j2) == "queued"
    assert guard.tokens_reserved >= 6_000

    cleaned = await bus.cleanup_task("t-clean", cancel_timeout_s=0.1)

    assert cleaned == 1
    assert bus.get_session(sid) is None
    assert bus.get_job_status(j1) == "aborted"
    assert bus.get_job_status(j2) == "aborted"
    assert guard.tokens_reserved == 0
    assert guard.active_count == 0
    gate.set()


# ──────────────────────────────────────────────────────────────────────────
# codex #6: spawn_context plumbing
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_submit_task_threads_spawn_context_into_pending(monkeypatch):
    """``submit_task_to_session(spawn_context=...)`` must land the dict
    on ``PendingSessionTask.spawn_context`` so ``_dispatch_session_task``
    can stamp it onto the sub trace observer at dispatch time. This is
    the Log Schema v1 §2.4 wiring for SFT data attribution.
    """
    gate = asyncio.Event()
    monkeypatch.setattr(
        "agent_core.components.agent_bus.bus.run_agent_loop",
        _gated_loop_factory(gate),
    )
    bus, sid = await _bus_session()

    spawn_ctx = {
        "parent_run_id": "heavy_run_1",
        "parent_agent_id": "swarm_main",
        "delegation_prompt": "find NVIDIA Q3 datacenter revenue",
        "allowed_tools": ["web_search"],
        "depth": 1,
        "budget": {"max_turns": 8},
    }

    # First task runs immediately (free path); second queues. The
    # queued task is the easy assertion target since its ``pending``
    # field is still readable on the session.
    await bus.submit_task_to_session(sid, "T1")
    await bus.submit_task_to_session(sid, "T2", spawn_context=spawn_ctx)

    session = bus.get_session(sid)
    # The queued task carries the spawn_context verbatim.
    assert len(session.pending_tasks) == 1
    queued = session.pending_tasks[0]
    assert queued.spawn_context == spawn_ctx
    # Defensive copy: caller mutation must not leak into the bus.
    spawn_ctx["depth"] = 999
    assert queued.spawn_context["depth"] == 1

    gate.set()
    # Drain so the test doesn't leave dangling tasks.
    for _ in range(2):
        await bus.wait_any_session("t-q", timeout=2)


@pytest.mark.asyncio
async def test_submit_task_without_spawn_context_stays_none(monkeypatch):
    """No spawn_context arg → ``pending.spawn_context`` stays ``None``
    so callers that don't wire the field keep the legacy behaviour."""
    gate = asyncio.Event()
    monkeypatch.setattr(
        "agent_core.components.agent_bus.bus.run_agent_loop",
        _gated_loop_factory(gate),
    )
    bus, sid = await _bus_session()

    await bus.submit_task_to_session(sid, "T1")
    await bus.submit_task_to_session(sid, "T2")

    session = bus.get_session(sid)
    assert session.pending_tasks[0].spawn_context is None
    gate.set()
    for _ in range(2):
        await bus.wait_any_session("t-q", timeout=2)
