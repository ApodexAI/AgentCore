"""SpawnGuard reservations must not outlive a failed submit.

``pre_check`` reserves tokens *and* registers the job in ``SpawnGuard._active``
before anything owns the job. The RAII release lives in ``_run_and_finalize``,
which does not exist until ``asyncio.create_task`` has run — so every failure
between the pre-check and that point leaks the reservation permanently.

A leak is not a transient glitch: ``remaining_tokens`` and ``active_count``
never recover, so a bus that has rejected N bad submits behaves as if N
sub-agents were running forever, and eventually refuses legitimate work with
``BudgetExhausted`` or blocks on a semaphore nobody holds.

Three entry points reserve: ``submit`` (unregistered role), and both session
paths — the free dispatch and the queued dispatch that ``_drain_session_queue``
performs later (whose failures are swallowed by design, which is exactly why
the release has to happen inside ``_dispatch_session_task``).
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from agent_core.components.agent_bus import AgentBus, SubTask
from agent_core.components.agent_bus.spawn_guard import SpawnGuard
from agent_core.loop_types import AgentLoopResult
from agent_core.messages import assistant_msg, user_msg
from agent_core.models.agent_definition import AgentDefinition
from agent_core.models.task_budget import TaskBudget
from agent_core.runtime.registries import services as registry
from agent_core.runtime.registries.agents import AgentRegistry
from agent_core.runtime.resources.manager import ResourceManager

BUDGET = TaskBudget(max_parallel=2, max_depth=4, max_tokens=30_000)


@pytest.fixture(autouse=True)
def _clear_registry():
    registry.clear()
    yield
    registry.clear()


def _registry_with(*role_ids: str) -> AgentRegistry:
    reg = AgentRegistry()
    for role_id in role_ids:
        reg.register(AgentDefinition(
            role_id=role_id, display_name=role_id, system_prompt="p",
            allowed_tools=[], color="#000", icon="agent",
        ))
    registry.register(AgentRegistry, reg)
    resource_mgr = MagicMock(spec=ResourceManager)
    resource_mgr.get_llm.return_value = MagicMock()
    resource_mgr.get_tools_for_role.return_value = []
    registry.register(ResourceManager, resource_mgr)
    return reg


def _assert_guard_is_clean(guard: SpawnGuard) -> None:
    assert guard.active_count == 0
    assert guard.tokens_reserved == 0
    assert guard.remaining_tokens == BUDGET.max_tokens


def _loop_factory(gate: asyncio.Event):
    async def loop(**kwargs):
        await gate.wait()
        um = kwargs.get("user_message", "")
        return AgentLoopResult(
            messages=[*list(kwargs.get("initial_messages") or []),
                      user_msg(um), assistant_msg(f"r:{um}")],
            final_content=f"r:{um}",
            metadata={},
        )
    return loop


async def _session(bus: AgentBus, task_id: str = "t-res") -> str:
    return await bus.create_session(
        task_id=task_id, name="lit", role_id="researcher",
        system_prompt="You are a researcher.",
        llm_override=MagicMock(), tools_override=[],
    )


# ── submit() ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_unregistered_role_releases_the_reservation():
    _registry_with("researcher")
    bus = AgentBus()
    guard = SpawnGuard(BUDGET)
    bus.set_spawn_guard(guard)

    with pytest.raises(KeyError):
        await bus.submit(
            "t1", SubTask(question="q", role_id="nope"), estimated_tokens=10_000,
        )

    _assert_guard_is_clean(guard)


@pytest.mark.asyncio
async def test_repeated_failed_submits_do_not_decay_the_budget():
    """The shape that made this a real outage rather than a cosmetic leak."""
    _registry_with("researcher")
    bus = AgentBus()
    guard = SpawnGuard(BUDGET)
    bus.set_spawn_guard(guard)

    for _ in range(3):
        with pytest.raises(KeyError):
            await bus.submit(
                "t1", SubTask(question="q", role_id="nope"),
                estimated_tokens=10_000,
            )

    _assert_guard_is_clean(guard)
    # Budget still admits a job that needs the whole allowance.
    await bus.submit(
        "t1", SubTask(question="q", role_id="researcher"),
        estimated_tokens=30_000,
    )


@pytest.mark.asyncio
async def test_successful_submit_keeps_the_reservation_until_the_job_ends(
    monkeypatch,
):
    """Guard against over-releasing: the fix must not free a live job's slot."""
    gate = asyncio.Event()
    monkeypatch.setattr(
        "agent_core.components.agent_bus.bus.run_agent_loop", _loop_factory(gate),
    )
    _registry_with("researcher")
    bus = AgentBus()
    guard = SpawnGuard(BUDGET)
    bus.set_spawn_guard(guard)

    await bus.submit(
        "t1", SubTask(question="q", role_id="researcher"), estimated_tokens=10_000,
    )
    await asyncio.sleep(0)

    assert guard.active_count == 1
    assert guard.tokens_reserved == 10_000


# ── session paths ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_failed_free_dispatch_releases_the_reservation():
    _registry_with("researcher")
    bus = AgentBus()
    guard = SpawnGuard(BUDGET)
    bus.set_spawn_guard(guard)
    sid = await _session(bus)

    session = bus.get_session(sid)
    assert session is not None
    session.trimmer = MagicMock()
    session.trimmer.trim.side_effect = RuntimeError("trim blew up")

    with pytest.raises(RuntimeError, match="trim blew up"):
        await bus.submit_task_to_session(sid, "T1", estimated_tokens=10_000)

    _assert_guard_is_clean(guard)


@pytest.mark.asyncio
async def test_failed_queued_dispatch_releases_the_reservation(monkeypatch):
    """``_drain_session_queue`` logs dispatch failures and moves on.

    So the queued task's reservation can only come back from inside
    ``_dispatch_session_task`` itself — nothing upstream is still watching.
    """
    gate = asyncio.Event()
    monkeypatch.setattr(
        "agent_core.components.agent_bus.bus.run_agent_loop", _loop_factory(gate),
    )
    _registry_with("researcher")
    bus = AgentBus()
    guard = SpawnGuard(BUDGET)
    bus.set_spawn_guard(guard)
    sid = await _session(bus)

    await bus.submit_task_to_session(sid, "T1", estimated_tokens=10_000)
    j2 = await bus.submit_task_to_session(sid, "T2", estimated_tokens=10_000)

    session = bus.get_session(sid)
    assert session is not None
    assert [p.job_id for p in session.pending_tasks] == [j2]
    assert guard.tokens_reserved == 20_000

    # T2 dispatches only once T1 finalises — break it in between.
    session.trimmer = MagicMock()
    session.trimmer.trim.side_effect = RuntimeError("trim blew up")
    gate.set()
    for _ in range(20):
        await asyncio.sleep(0)
        if not session.pending_tasks and guard.active_count == 0:
            break

    _assert_guard_is_clean(guard)


@pytest.mark.asyncio
async def test_queued_task_keeps_its_reservation_while_it_waits(monkeypatch):
    """Queued tasks reserve at submit time on purpose — don't undo that."""
    gate = asyncio.Event()
    monkeypatch.setattr(
        "agent_core.components.agent_bus.bus.run_agent_loop", _loop_factory(gate),
    )
    _registry_with("researcher")
    bus = AgentBus()
    guard = SpawnGuard(BUDGET)
    bus.set_spawn_guard(guard)
    sid = await _session(bus)

    await bus.submit_task_to_session(sid, "T1", estimated_tokens=10_000)
    await bus.submit_task_to_session(sid, "T2", estimated_tokens=10_000)

    assert guard.tokens_reserved == 20_000
    assert guard.active_count == 2

    # Drain so the queued task is not garbage-collected mid-flight.
    gate.set()
    for _ in range(40):
        await asyncio.sleep(0)
        if guard.active_count == 0:
            break
    _assert_guard_is_clean(guard)
