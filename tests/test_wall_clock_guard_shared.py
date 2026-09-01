"""WallClockGuard regression tests.

The guard exists so a sub-agent's ``SpawnGuard`` slot expires *inside* the
loop (clean exit → ``force_final_answer`` salvages the report) instead of
being hard-cancelled from outside by ``asyncio.wait_for`` (whole run
discarded as ``(empty report)``). These tests pin the three things that
make that work: the stop reason, the reserve sizing, and fire-once —
plus the two timeout invariants the ``tool_timeout_s: 360`` profile change
depends on.
"""

from __future__ import annotations

import pytest

from agent_core.components.observers.wall_clock_guard import WallClockGuard
from agent_core.loop_types import LoopConfig, TurnContext


def _ctx(turn: int) -> TurnContext:
    return TurnContext(
        turn=turn,
        max_turns=100,
        task_id="t",
        role_id="swarm_sub",
        ai_text="",
        thinking="",
        tool_calls=[],
        messages=[],
        usage=None,
        metadata={},
    )


def _config(*, llm_timeout: int, tool_timeout: int) -> LoopConfig:
    return LoopConfig(
        max_turns=100,
        task_id="t",
        llm_timeout=llm_timeout,
        tool_timeout=tool_timeout,
    )


def test_rejects_non_positive_budget() -> None:
    with pytest.raises(ValueError):
        WallClockGuard(budget_s=0)


@pytest.mark.asyncio
async def test_does_not_trip_before_soft_deadline() -> None:
    guard = WallClockGuard(budget_s=5400, reserve_s=600)
    await guard.on_loop_start(_config(llm_timeout=600, tool_timeout=360))
    assert await guard.on_turn_end(_ctx(1)) is None


@pytest.mark.asyncio
async def test_trips_with_wall_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stop reason must be ``wall_deadline`` — the shared "ran out of
    clock" reason, and a member of ``INCOMPLETE_STOP_REASONS`` so
    ``force_final_answer`` salvages a report instead of the run being lost."""
    clock = {"t": 1000.0}
    monkeypatch.setattr(
        "agent_core.components.observers.wall_clock_observer.time.monotonic",
        lambda: clock["t"],
    )
    guard = WallClockGuard(budget_s=5400, reserve_s=600)
    await guard.on_loop_start(_config(llm_timeout=600, tool_timeout=360))

    # reserve = max(600, 600 + 360 + 60) = 1020 → soft deadline 4380s.
    assert guard.soft_deadline_s == pytest.approx(4380.0)

    # Inherited one-shot warn at warn_ratio (0.8 × 4380 = 3504s): a nudge, but
    # NOT a stop — the sub-agent gets a chance to start consolidating.
    clock["t"] += 4379
    warn = await guard.on_turn_end(_ctx(1))
    assert warn is not None and warn.stop_reason is None
    assert "consolidating" in warn.inject_messages[0]

    clock["t"] += 2
    iv = await guard.on_turn_end(_ctx(2))
    assert iv is not None
    assert iv.stop_reason == "wall_deadline"
    # Sub-agent wording: deliver the report, not "stop spawning sub-agents"
    # (the parent's coordinator-facing default).
    assert "deliver your report now" in iv.inject_messages[0]


@pytest.mark.asyncio
async def test_publishes_soft_deadline_into_execution_scope() -> None:
    """The stamp is load-bearing, not cosmetic: ``llm_client.call_llm`` reads
    it per RETRY ATTEMPT and clamps that attempt to the wall remaining, which
    is the only thing bounding a multi-retry turn — ``on_turn_end`` cannot,
    since it runs between turns."""
    from agent_core.execution_context import (
        ExecutionScope,
        reset_current_execution_scope,
        set_current_execution_scope,
    )
    from agent_core.loop_types import (
        WALL_DEADLINE_MONOTONIC_KEY,
        wall_deadline_remaining_s,
    )

    scope = ExecutionScope(task_id="t", role_id="swarm_sub", metadata={})
    token = set_current_execution_scope(scope)
    try:
        assert wall_deadline_remaining_s() is None  # nothing stamped yet
        guard = WallClockGuard(budget_s=5400, reserve_s=600)
        await guard.on_loop_start(_config(llm_timeout=600, tool_timeout=360))
        assert WALL_DEADLINE_MONOTONIC_KEY in scope.metadata
        # ~4380s of usable budget, minus the moment spent arming.
        assert wall_deadline_remaining_s() == pytest.approx(4380.0, abs=5.0)
    finally:
        reset_current_execution_scope(token)


@pytest.mark.asyncio
async def test_reserve_floor_covers_one_worst_case_turn() -> None:
    """The check runs in ``on_turn_end``, so a turn starting just under the
    soft deadline must still fit inside the hard one: reserve >= llm_timeout
    + tool_timeout + 60, even when the caller asked for less."""
    guard = WallClockGuard(budget_s=18000, reserve_s=60)
    await guard.on_loop_start(_config(llm_timeout=600, tool_timeout=360))
    assert guard.soft_deadline_s == pytest.approx(18000 - 1020)


@pytest.mark.asyncio
async def test_reserve_never_eats_more_than_half_the_budget() -> None:
    """A tight budget with generous timeouts must still leave the loop room
    to run, or the guard would trip on turn 1 and no work would happen."""
    guard = WallClockGuard(budget_s=1200, reserve_s=600)
    # floor would be 1800 + 900 + 60 = 2760 > budget.
    await guard.on_loop_start(_config(llm_timeout=1800, tool_timeout=900))
    assert guard.soft_deadline_s == pytest.approx(600.0)


@pytest.mark.asyncio
async def test_sub_tool_timeout_shrinks_the_reserve() -> None:
    """Why the agent-team profiles moved tool_timeout_s 1800 → 360: the value
    is subtracted from every sub-agent's usable wall clock."""
    at_1800 = WallClockGuard(budget_s=5400)
    await at_1800.on_loop_start(_config(llm_timeout=600, tool_timeout=1800))
    at_360 = WallClockGuard(budget_s=5400)
    await at_360.on_loop_start(_config(llm_timeout=600, tool_timeout=360))
    # 2460s of reserve vs 1020s — 24 extra minutes of research per sub-agent.
    assert at_360.soft_deadline_s - at_1800.soft_deadline_s == pytest.approx(
        1440.0
    )


def test_is_a_wall_clock_deadline_observer() -> None:
    """One implementation of the deadline arithmetic, not two — the whole
    point of subclassing, so a fix lands once."""
    from agent_core.components.observers.wall_clock_observer import (
        WallClockDeadlineObserver,
    )

    assert issubclass(WallClockGuard, WallClockDeadlineObserver)
    assert WallClockGuard(budget_s=60).critical is True

