"""Tests for SpawnGuard (Issue #24 Phase B).

Covers: acquire/release, concurrency, depth, budget, timeout,
RAII context manager, and AgentBus integration.
"""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_core.components.agent_bus import (
    AgentBus,
    DepthLimitExceeded,
    SubTask,
)
from agent_core.components.agent_bus.spawn_guard import (
    BudgetExhausted,
    SpawnDepthExceeded,
    SpawnGuard,
)
from agent_core.loop_types import AgentLoopResult
from agent_core.models.agent_definition import AgentDefinition
from agent_core.models.task_budget import TaskBudget
from agent_core.protocols import EventSink
from agent_core.runtime.registries import services as registry
from agent_core.runtime.registries.agents import AgentRegistry
from agent_core.runtime.resources.manager import ResourceManager


class _FakeEventStore:
    def __init__(self) -> None:
        self.append = AsyncMock()
        self.get_events = AsyncMock(return_value=[])


@pytest.fixture(autouse=True)
def clear_registry():
    registry.clear()
    yield
    registry.clear()


def _setup_registry():
    agent_reg = AgentRegistry()
    agent_reg.register(AgentDefinition(
        role_id="researcher",
        display_name="Researcher",
        system_prompt="test",
        allowed_tools=[],
        color="#000", icon="agent",
    ))
    registry.register(AgentRegistry, agent_reg)
    _es = _FakeEventStore()
    registry.register(EventSink, _es)
    resource_mgr = MagicMock(spec=ResourceManager)
    resource_mgr.get_llm.return_value = MagicMock()
    resource_mgr.get_tools_for_role.return_value = []
    registry.register(ResourceManager, resource_mgr)


# ── SpawnGuard standalone tests ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_acquire_release_basic():
    """Basic acquire + release cycle."""
    guard = SpawnGuard(TaskBudget(max_parallel=2, max_depth=3))

    res = await guard.acquire("job-1", depth=1, estimated_tokens=1000)
    assert res.job_id == "job-1"
    assert res.depth == 1
    assert guard.active_count == 1
    assert guard.tokens_reserved == 1000

    guard.release("job-1")
    assert guard.active_count == 0
    assert guard.tokens_reserved == 0


@pytest.mark.asyncio
async def test_acquire_depth_exceeded():
    """Should raise SpawnDepthExceeded at max depth."""
    guard = SpawnGuard(TaskBudget(max_depth=2))

    with pytest.raises(SpawnDepthExceeded):
        await guard.acquire("job-1", depth=2)


@pytest.mark.asyncio
async def test_acquire_budget_exhausted():
    """Should raise BudgetExhausted when tokens exceed remaining."""
    guard = SpawnGuard(TaskBudget(max_tokens=5000))

    # First acquire takes most of the budget
    await guard.acquire("job-1", depth=0, estimated_tokens=4000)

    # Second acquire exceeds remaining (1000)
    with pytest.raises(BudgetExhausted):
        await guard.acquire("job-2", depth=0, estimated_tokens=2000)

    guard.release("job-1")


@pytest.mark.asyncio
async def test_concurrency_limit_queues():
    """Should queue (not reject) when concurrency limit reached."""
    guard = SpawnGuard(TaskBudget(max_parallel=1, max_depth=5))

    await guard.acquire("job-1", depth=0)
    assert guard.active_count == 1

    # Second acquire should block (we test by using a timeout)
    acquired = asyncio.Event()

    async def try_acquire():
        await guard.acquire("job-2", depth=0)
        acquired.set()

    task = asyncio.create_task(try_acquire())
    await asyncio.sleep(0.05)
    assert not acquired.is_set()  # Still blocked

    # Release first slot — second should proceed
    guard.release("job-1")
    await asyncio.sleep(0.05)
    assert acquired.is_set()
    assert guard.active_count == 1

    guard.release("job-2")
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_release_idempotent():
    """release() called twice should not error."""
    guard = SpawnGuard(TaskBudget(max_parallel=2))

    await guard.acquire("job-1", depth=0, estimated_tokens=500)
    guard.release("job-1")
    guard.release("job-1")  # Second call — should be noop
    assert guard.active_count == 0


@pytest.mark.asyncio
async def test_release_corrects_token_estimate():
    """release() with actual_tokens should adjust accounting."""
    guard = SpawnGuard(TaskBudget(max_tokens=10000))

    await guard.acquire("job-1", depth=0, estimated_tokens=3000)
    assert guard.tokens_reserved == 3000

    guard.release("job-1", actual_tokens=2000)
    assert guard.tokens_reserved == 0  # Estimate removed
    assert guard.tokens_actual == 2000  # Actual recorded


@pytest.mark.asyncio
async def test_raii_context_manager():
    """reservation() should auto-release on exit."""
    guard = SpawnGuard(TaskBudget(max_parallel=2, max_depth=3))

    async with guard.reservation("job-1", depth=1, estimated_tokens=500) as res:
        assert res.job_id == "job-1"
        assert guard.active_count == 1

    assert guard.active_count == 0


@pytest.mark.asyncio
async def test_raii_releases_on_exception():
    """reservation() should release even if body raises."""
    guard = SpawnGuard(TaskBudget(max_parallel=2, max_depth=3))

    with pytest.raises(ValueError):
        async with guard.reservation("job-1", depth=1):
            raise ValueError("boom")

    assert guard.active_count == 0


@pytest.mark.asyncio
async def test_release_without_acquire_slot_does_not_leak_semaphore():
    """Regression: ``release`` after ``pre_check`` only must not return
    a slot we never owned.

    Reproduces the scenario where ``bus.py:_run_and_finalize`` ran
    ``pre_check`` (registering the reservation), then was cancelled
    before reaching ``acquire_slot`` so the semaphore was never taken,
    and finally hit its ``finally`` block which always calls ``release``.
    Pre-fix the unconditional ``self._semaphore.release()`` inflated the
    counter past ``max_parallel``, breaking the concurrency invariant.
    Post-fix the reservation's ``acquired`` flag is ``False`` so the
    semaphore is left intact.
    """
    guard = SpawnGuard(TaskBudget(max_parallel=1, max_depth=5))

    # Consume the only slot.
    await guard.acquire("job-1", depth=0)

    # Simulate "pre_check ran, acquire_slot was cancelled before being
    # awaited, finally block now calls release" — the exact path
    # bus.py walks when its outer task is cancelled mid-await.
    await guard.pre_check("job-2", depth=0, estimated_tokens=500)
    guard.release("job-2")  # finally-block fallback

    # job-1 still owns the only slot. A bystander acquire must still
    # block — pre-fix this resolved instantly because the bogus
    # ``self._semaphore.release()`` bumped the counter to 1 free.
    bystander = asyncio.create_task(guard.acquire("job-3", depth=0))
    await asyncio.sleep(0.05)
    assert not bystander.done(), (
        "release() leaked a semaphore slot — concurrency limit broken"
    )

    # Real release of job-1 must let the waiter through, proving the
    # semaphore counter is still consistent after the dance.
    guard.release("job-1")
    await asyncio.wait_for(bystander, timeout=0.5)
    guard.release("job-3")
    assert guard.active_count == 0


@pytest.mark.asyncio
async def test_release_after_acquire_slot_returns_semaphore_normally():
    """Sanity check: when ``acquire_slot`` did succeed, ``release`` must
    still hand the slot back. Guards against an over-eager ``acquired``
    check that would also mask normal releases."""
    guard = SpawnGuard(TaskBudget(max_parallel=1, max_depth=5))

    res = await guard.acquire("job-1", depth=0)
    assert res.acquired is True

    guard.release("job-1")

    # Slot is free again — bystander resolves without blocking.
    await asyncio.wait_for(
        guard.acquire("job-2", depth=0), timeout=0.5,
    )
    guard.release("job-2")


@pytest.mark.asyncio
async def test_stats():
    """stats() should return current state."""
    guard = SpawnGuard(TaskBudget(
        max_parallel=3, max_depth=2, max_tokens=50000,
    ))
    await guard.acquire("job-1", depth=0, estimated_tokens=1000)

    s = guard.stats()
    assert s["active"] == 1
    assert s["max_parallel"] == 3
    assert s["max_depth"] == 2
    assert s["tokens_reserved"] == 1000
    assert s["max_tokens"] == 50000
    assert s["total_spawns"] == 1

    guard.release("job-1")


# ── AgentBus + SpawnGuard integration ───────────────────────────────────


@pytest.mark.asyncio
async def test_agent_bus_with_spawn_guard_enforces_concurrency(monkeypatch):
    """AgentBus with SpawnGuard should respect max_parallel."""
    _setup_registry()

    started = []
    gate = asyncio.Event()

    async def gated_agent_loop(**kwargs):
        task_id = ""
        cfg = kwargs.get("config")
        if cfg:
            task_id = getattr(cfg, "task_id", "")
        started.append(task_id)
        await gate.wait()
        return AgentLoopResult(
            messages=[], final_content="ok", metadata={},
        )

    monkeypatch.setattr(
        "agent_core.components.agent_bus.bus.run_agent_loop",
        gated_agent_loop,
    )

    guard = SpawnGuard(TaskBudget(max_parallel=1, max_depth=3))
    bus = AgentBus()
    bus.set_spawn_guard(guard)

    # Submit 2 jobs — only 1 should start (concurrency=1)
    jid1 = await bus.submit("t", SubTask("q1"))
    jid2 = await bus.submit("t", SubTask("q2"))
    await asyncio.sleep(0.1)

    # Only 1 should have started (other queued at semaphore)
    assert len(started) == 1

    # Unblock — both should eventually complete
    gate.set()
    result = await bus.collect([jid1, jid2], timeout=5)
    assert len(result.completed) == 2


@pytest.mark.asyncio
async def test_agent_bus_with_spawn_guard_depth_limit(monkeypatch):
    """AgentBus should enforce SpawnGuard depth limit."""
    _setup_registry()

    guard = SpawnGuard(TaskBudget(max_depth=1))
    bus = AgentBus()
    bus.set_spawn_guard(guard)

    # depth=1 should be rejected (max_depth=1 means depth<1 only)
    with pytest.raises(DepthLimitExceeded):
        await bus.submit("t", SubTask("q"), current_depth=1, max_depth=5)


@pytest.mark.asyncio
async def test_agent_bus_with_spawn_guard_budget_exhausted(monkeypatch):
    """AgentBus should reject when SpawnGuard budget exhausted."""
    _setup_registry()

    guard = SpawnGuard(TaskBudget(max_tokens=1000, max_depth=3))
    bus = AgentBus()
    bus.set_spawn_guard(guard)

    with pytest.raises(BudgetExhausted):
        await bus.submit(
            "t", SubTask("q"), estimated_tokens=2000,
        )


@pytest.mark.asyncio
async def test_agent_bus_spawn_guard_releases_on_completion(monkeypatch):
    """SpawnGuard slot should be released when job finishes."""
    _setup_registry()

    async def fast_agent_loop(**kwargs):
        return AgentLoopResult(
            messages=[], final_content="ok", metadata={},
        )

    monkeypatch.setattr(
        "agent_core.components.agent_bus.bus.run_agent_loop",
        fast_agent_loop,
    )

    guard = SpawnGuard(TaskBudget(max_parallel=2, max_depth=3))
    bus = AgentBus()
    bus.set_spawn_guard(guard)

    jid = await bus.submit("t", SubTask("q"))
    await bus.collect([jid], timeout=5)

    # Guard slot should be released
    assert guard.active_count == 0
    assert guard.total_spawns == 1


@pytest.mark.asyncio
async def test_agent_bus_spawn_guard_releases_on_failure(monkeypatch):
    """SpawnGuard slot should be released when job fails."""
    _setup_registry()

    async def failing_agent_loop(**kwargs):
        raise RuntimeError("fail")

    monkeypatch.setattr(
        "agent_core.components.agent_bus.bus.run_agent_loop",
        failing_agent_loop,
    )

    guard = SpawnGuard(TaskBudget(max_parallel=2, max_depth=3))
    bus = AgentBus()
    bus.set_spawn_guard(guard)

    jid = await bus.submit("t", SubTask("q"))
    await bus.collect([jid], timeout=5)

    assert guard.active_count == 0


@pytest.mark.asyncio
async def test_agent_bus_spawn_guard_releases_on_abort(monkeypatch):
    """SpawnGuard slot should be released on abort."""
    _setup_registry()

    async def slow_agent_loop(**kwargs):
        await asyncio.sleep(100)
        return AgentLoopResult(messages=[], final_content="", metadata={})

    monkeypatch.setattr(
        "agent_core.components.agent_bus.bus.run_agent_loop",
        slow_agent_loop,
    )

    guard = SpawnGuard(TaskBudget(max_parallel=2, max_depth=3))
    bus = AgentBus()
    bus.set_spawn_guard(guard)

    jid = await bus.submit("t", SubTask("q"))
    await asyncio.sleep(0.05)
    assert guard.active_count == 1

    await bus.abort(jid)
    # Give finalize a moment to run
    await asyncio.sleep(0.05)
    assert guard.active_count == 0


@pytest.mark.asyncio
async def test_agent_bus_without_spawn_guard_still_works(monkeypatch):
    """AgentBus without SpawnGuard should work as before."""
    _setup_registry()

    async def fast_agent_loop(**kwargs):
        return AgentLoopResult(
            messages=[], final_content="ok", metadata={},
        )

    monkeypatch.setattr(
        "agent_core.components.agent_bus.bus.run_agent_loop",
        fast_agent_loop,
    )

    bus = AgentBus()  # No spawn guard
    jid = await bus.submit("t", SubTask("q"))
    result = await bus.collect([jid], timeout=5)
    assert len(result.completed) == 1
