"""Renewable wall-time semantics for live user intervention."""

from __future__ import annotations

import asyncio
import threading

import pytest

from agent_core.components.observers.wall_clock_observer import (
    WallClockDeadlineObserver,
)
from agent_core.execution_context import (
    ExecutionScope,
    reset_current_execution_scope,
    set_current_execution_scope,
)
from agent_core.loop_types import (
    WALL_DEADLINE_MONOTONIC_KEY,
    LoopConfig,
    TurnContext,
    wall_deadline_remaining_s,
)
from agent_core.runtime import wall_time_lease as lease_module
from agent_core.runtime.wall_time_lease import (
    WALL_TIME_LEASE_SCOPE_KEY,
    RenewableWallTimeDeadline,
    RenewableWallTimeExceeded,
    RenewableWallTimeLease,
    run_with_renewable_wall_time,
)


@pytest.fixture
def fake_clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    clock = [100.0]
    monkeypatch.setattr(lease_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(lease_module.time, "time", lambda: 1_800_000_000 + clock[0])
    return clock


def _ctx(metadata: dict | None = None) -> TurnContext:
    return TurnContext(
        turn=1,
        max_turns=10,
        task_id="task",
        role_id="main",
        ai_text="",
        thinking="",
        tool_calls=[],
        messages=[],
        usage=None,
        metadata=metadata or {},
    )


def test_renew_restores_full_bound_window(fake_clock: list[float]) -> None:
    lease = RenewableWallTimeLease()
    deadline = lease.bind_duration(20)
    assert deadline.remaining_s() == pytest.approx(20)

    fake_clock[0] += 17
    assert deadline.remaining_s() == pytest.approx(3)

    renewal = lease.renew()
    assert renewal.sequence == 1
    assert deadline.remaining_s() == pytest.approx(20)
    assert renewal.ack_fields()["walltime_reset"] is True
    assert renewal.ack_fields()["walltime_reset_seq"] == 1


def test_binding_soft_deadline_does_not_move_existing_hard_window(
    fake_clock: list[float],
) -> None:
    lease = RenewableWallTimeLease()
    hard_started = fake_clock[0]
    fake_clock[0] += 4

    soft_deadline = lease.bind_duration(20)

    assert soft_deadline.remaining_s() == pytest.approx(20)
    assert lease.remaining_s_for(
        20,
        started_monotonic=hard_started,
    ) == pytest.approx(16)


def test_concurrent_deadline_views_keep_independent_durations(
    fake_clock: list[float],
) -> None:
    lease = RenewableWallTimeLease()
    short = lease.bind_duration(10)
    long = lease.bind_duration(20)

    fake_clock[0] += 3
    assert short.remaining_s() == pytest.approx(7)
    assert long.remaining_s() == pytest.approx(17)

    lease.renew()
    assert short.remaining_s() == pytest.approx(10)
    assert long.remaining_s() == pytest.approx(20)


def test_concurrent_renewals_cannot_move_anchor_backwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sequence order and sampled anchors must share one critical section."""
    older_sampled = threading.Event()
    release_older = threading.Event()
    newer_sampled = threading.Event()
    results: dict[str, object] = {}

    def controlled_monotonic() -> float:
        if threading.current_thread().name == "older-renewal":
            older_sampled.set()
            assert release_older.wait(timeout=2)
            return 100.0
        if threading.current_thread().name == "newer-renewal":
            newer_sampled.set()
        return 200.0

    monkeypatch.setattr(lease_module.time, "monotonic", controlled_monotonic)
    monkeypatch.setattr(lease_module.time, "time", lambda: 1_800_000_000.0)
    lease = RenewableWallTimeLease()

    def renew(label: str) -> None:
        results[label] = lease.renew()

    older = threading.Thread(
        target=renew,
        args=("older",),
        name="older-renewal",
    )
    newer = threading.Thread(
        target=renew,
        args=("newer",),
        name="newer-renewal",
    )
    older.start()
    assert older_sampled.wait(timeout=2)
    newer.start()
    # With the broken pre-lock sampling, the newer caller reaches the clock
    # while the older caller is paused and commits first. With the fixed code
    # it remains blocked on the lease lock until the older caller is released.
    newer_sampled.wait(timeout=0.1)
    release_older.set()
    older.join(timeout=2)
    newer.join(timeout=2)

    assert not older.is_alive()
    assert not newer.is_alive()
    older_renewal = results["older"]
    newer_renewal = results["newer"]
    assert isinstance(older_renewal, lease_module.WallTimeRenewal)
    assert isinstance(newer_renewal, lease_module.WallTimeRenewal)
    assert (older_renewal.sequence, older_renewal.monotonic_s) == (1, 100.0)
    assert (newer_renewal.sequence, newer_renewal.monotonic_s) == (2, 200.0)
    assert lease.remaining_s_for(10) == pytest.approx(10)


@pytest.mark.asyncio
async def test_hard_guard_survives_original_deadline_after_renewal() -> None:
    lease = RenewableWallTimeLease()

    async def work() -> str:
        await asyncio.sleep(0.25)
        return "finished"

    async def renew() -> None:
        await asyncio.sleep(0.10)
        lease.renew()

    renewal_task = asyncio.create_task(renew())
    try:
        result = await run_with_renewable_wall_time(
            work(),
            lease=lease,
            duration_s=0.20,
        )
    finally:
        await renewal_task

    assert result == "finished"
    assert lease.sequence == 1


@pytest.mark.asyncio
async def test_hard_guard_cancels_when_lease_really_expires() -> None:
    lease = RenewableWallTimeLease()
    expired = False

    def on_expire() -> None:
        nonlocal expired
        expired = True

    with pytest.raises(RenewableWallTimeExceeded, match="reset_seq=0"):
        await run_with_renewable_wall_time(
            asyncio.sleep(60),
            lease=lease,
            duration_s=0.05,
            on_expire=on_expire,
            grace_s=0,
        )

    assert expired is True


@pytest.mark.asyncio
async def test_hard_guard_allows_graceful_stop_after_expiry() -> None:
    lease = RenewableWallTimeLease()
    stop_requested = asyncio.Event()

    async def work() -> str:
        await stop_requested.wait()
        return "partial answer"

    result = await run_with_renewable_wall_time(
        work(),
        lease=lease,
        duration_s=0.01,
        on_expire=stop_requested.set,
        grace_s=0.1,
    )

    assert result == "partial answer"


@pytest.mark.asyncio
async def test_absolute_session_cap_cannot_be_extended_by_renewals() -> None:
    lease = RenewableWallTimeLease()

    async def renew_repeatedly() -> None:
        while True:
            await asyncio.sleep(0.01)
            lease.renew()

    renew_task = asyncio.create_task(renew_repeatedly())
    try:
        with pytest.raises(
            RenewableWallTimeExceeded,
            match="maximum session wall-time exceeded",
        ):
            await run_with_renewable_wall_time(
                asyncio.sleep(60),
                lease=lease,
                duration_s=0.05,
                max_session_s=0.08,
                grace_s=0,
            )
    finally:
        renew_task.cancel()
        await asyncio.gather(renew_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_wall_observer_and_deadline_readers_follow_renewal(
    fake_clock: list[float],
) -> None:
    lease = RenewableWallTimeLease()
    scope = ExecutionScope(
        task_id="task",
        role_id="main",
        metadata={WALL_TIME_LEASE_SCOPE_KEY: lease},
    )
    token = set_current_execution_scope(scope)
    try:
        observer = WallClockDeadlineObserver(
            deadline_s=10,
            reserve_s=2,
            warn_ratio=0.5,
        )
        await observer.on_loop_start(LoopConfig(task_id="task"))

        assert isinstance(
            scope.metadata[WALL_DEADLINE_MONOTONIC_KEY],
            RenewableWallTimeDeadline,
        )
        assert wall_deadline_remaining_s() == pytest.approx(8)

        fake_clock[0] += 4.1
        first_warning = await observer.on_turn_end(_ctx())
        assert first_warning is not None
        assert first_warning.stop_reason is None

        renewal = lease.renew()
        assert renewal.sequence == 1
        assert wall_deadline_remaining_s() == pytest.approx(8)

        after_reset = _ctx()
        fake_clock[0] += 1
        assert await observer.on_turn_end(after_reset) is None
        assert after_reset.metadata["walltime_reset_seq"] == 1

        # The one-shot warning belongs to one lease epoch. A renewal must
        # re-arm it, otherwise the extended turn gets no wrap-up warning.
        fake_clock[0] += 3.1
        second_warning = await observer.on_turn_end(_ctx())
        assert second_warning is not None
        assert second_warning.stop_reason is None
    finally:
        reset_current_execution_scope(token)
