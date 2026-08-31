"""``LLMCallExhausted.last_exc`` must always agree with ``.reason``.

A chain wrapper classifies ``last_exc`` directly to decide whether to sleep,
rotate a provider leg, or give up. So a deadline-driven raise that carried an
unrelated earlier failure in ``last_exc`` told the wrapper the opposite of
what ``reason`` said: reason=``wall_deadline`` ("the run is out of budget")
paired with a stale 429 ("sleep and retry this key"), and the wrapper would
retry straight past the deadline the reason had just announced.

The superseded failure is still reachable as ``prior_exc`` — diagnostics,
deliberately outside the field classification reads.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

import agent_core.runtime.loop._call as call_module
from agent_core.llm import LLMResponse
from agent_core.messages import user_msg
from agent_core.runtime.loop.llm_client import (
    LLMCallExhausted,
    LLMDeadlineExceeded,
    call_llm,
)


class _RateLimit429(Exception):
    def __init__(self) -> None:
        super().__init__("rate limited")
        self.status_code = 429


@pytest.fixture
def _instant_sleep(monkeypatch):
    """Keep backoff schedules deterministic without burning wall time."""
    real_sleep = asyncio.sleep

    async def _record(_duration):
        await real_sleep(0)

    monkeypatch.setattr("asyncio.sleep", _record)


@pytest.mark.asyncio
async def test_pre_attempt_refusal_after_a_failed_attempt_reports_the_deadline(
    monkeypatch,
) -> None:
    """Attempt 1 fails 429, its backoff fits, then attempt 2 is refused.

    Covers the pre-attempt floor check specifically: the budget is ample when
    the retry sleep is decided and under the floor by the time the next
    attempt would start, so the refusal happens before the provider is
    touched a second time.
    """
    monkeypatch.setattr(call_module, "_WALL_DEADLINE_FLOOR_S", 20.0)
    calls = 0
    slept = False
    real_sleep = asyncio.sleep

    async def _sleep(_duration):
        nonlocal slept
        slept = True
        await real_sleep(0)

    monkeypatch.setattr("asyncio.sleep", _sleep)

    async def _chat(_messages, **_kw):
        nonlocal calls
        calls += 1
        raise _RateLimit429

    # Keyed on the retry sleep actually having happened, not on how often the
    # deadline is polled: ample while the backoff is being decided, under the
    # floor once it has been taken.
    def _remaining() -> float:
        return 5.0 if slept else 600.0

    with pytest.raises(LLMCallExhausted) as exc_info:
        await call_llm(
            SimpleNamespace(chat=_chat, model="fake"),
            [user_msg("hello")],
            timeout=120,
            max_retries=4,
            turn=1,
            retry_wait_fixed=1,  # 1s + 20s floor < 600s, so the sleep is taken
            wall_deadline_remaining=_remaining,
        )

    exhausted = exc_info.value
    assert slept, "expected the retry backoff to be taken, not abandoned"
    assert calls == 1, "the second attempt must be refused before the provider"
    assert exhausted.reason == "wall_deadline"
    # The reason is a deadline, so last_exc must be the deadline — not the 429
    # a chain wrapper would have read as "sleep and retry".
    assert isinstance(exhausted.last_exc, LLMDeadlineExceeded)
    assert exhausted.last_exc.reason == exhausted.reason
    assert "wall_deadline" in str(exhausted.last_exc)
    assert isinstance(exhausted.prior_exc, _RateLimit429)


@pytest.mark.asyncio
async def test_backoff_crossing_the_wall_deadline_keeps_the_deadline_reason(
    monkeypatch, _instant_sleep,
) -> None:
    """Abandoning the retry sleep must not relabel itself ``exhausted``.

    This path used to ``break`` into the generic exhausted-retries raise,
    discarding the wall-deadline signal even though the attempt event emitted
    alongside it already reported ``wall_deadline`` — so a caller could not
    tell "out of run budget, go salvage" from "this key is spent, try
    another leg".
    """
    monkeypatch.setattr(call_module, "_WALL_DEADLINE_FLOOR_S", 20.0)
    calls = 0

    async def _chat(_messages, **_kw):
        nonlocal calls
        calls += 1
        raise _RateLimit429

    # Above the floor (so the attempt runs) but too small for the 429 backoff
    # plus the floor, so the retry sleep is abandoned instead of taken.
    with pytest.raises(LLMCallExhausted) as exc_info:
        await call_llm(
            SimpleNamespace(chat=_chat, model="fake"),
            [user_msg("hello")],
            timeout=30,
            max_retries=4,
            turn=1,
            wall_deadline_remaining=lambda: 25.0,
        )

    exhausted = exc_info.value
    assert calls == 1
    assert exhausted.reason == "wall_deadline"
    assert isinstance(exhausted.last_exc, LLMDeadlineExceeded)
    assert exhausted.last_exc.reason == exhausted.reason
    assert isinstance(exhausted.prior_exc, _RateLimit429)


@pytest.mark.asyncio
async def test_logical_deadline_abandoning_backoff_reports_its_own_reason(
    monkeypatch, _instant_sleep,
) -> None:
    """The logical call deadline is labelled distinctly from the run wall."""
    monkeypatch.setattr(call_module, "_WALL_DEADLINE_FLOOR_S", 20.0)

    async def _chat(_messages, **_kw):
        raise _RateLimit429

    with pytest.raises(LLMCallExhausted) as exc_info:
        await call_llm(
            SimpleNamespace(chat=_chat, model="fake"),
            [user_msg("hello")],
            timeout=30,
            max_retries=4,
            turn=1,
            logical_call_timeout_s=25.0,
        )

    assert exc_info.value.reason == "logical_call_deadline"
    assert isinstance(exc_info.value.last_exc, LLMDeadlineExceeded)
    assert exc_info.value.last_exc.reason == exc_info.value.reason
    assert isinstance(exc_info.value.prior_exc, _RateLimit429)


@pytest.mark.asyncio
async def test_wall_clamped_final_attempt_timeout_reports_wall_deadline(
    monkeypatch,
) -> None:
    """A provider still running when the wall budget expires is not exhausted."""
    monkeypatch.setattr(call_module, "_WALL_DEADLINE_FLOOR_S", 0.0)
    deadline = time.monotonic() + 0.05

    async def _chat(_messages, **_kw):
        await asyncio.sleep(1)

    with pytest.raises(LLMCallExhausted) as exc_info:
        await call_llm(
            SimpleNamespace(chat=_chat, model="fake"),
            [user_msg("hello")],
            timeout=1,
            max_retries=1,
            turn=1,
            wall_deadline_remaining=lambda: deadline - time.monotonic(),
        )

    exhausted = exc_info.value
    assert exhausted.reason == "wall_deadline"
    assert isinstance(exhausted.last_exc, LLMDeadlineExceeded)
    assert exhausted.last_exc.reason == exhausted.reason
    assert exhausted.prior_exc is None


@pytest.mark.asyncio
async def test_in_flight_logical_deadline_preserves_reason(monkeypatch) -> None:
    """The logical deadline remains identifiable after callers unwrap it."""
    monkeypatch.setattr(call_module, "_WALL_DEADLINE_FLOOR_S", 0.0)

    async def _chat(_messages, **_kw):
        await asyncio.sleep(1)

    with pytest.raises(LLMCallExhausted) as exc_info:
        await call_llm(
            SimpleNamespace(chat=_chat, model="fake"),
            [user_msg("hello")],
            timeout=1,
            max_retries=1,
            turn=1,
            logical_call_timeout_s=0.05,
        )

    exhausted = exc_info.value
    assert exhausted.reason == "logical_call_deadline"
    assert isinstance(exhausted.last_exc, LLMDeadlineExceeded)
    assert exhausted.last_exc.reason == exhausted.reason
    assert exhausted.prior_exc is None


@pytest.mark.asyncio
async def test_real_concurrency_gate_wait_preserves_wall_deadline(
    monkeypatch,
) -> None:
    """E2E: a real semaphore holder makes a second call consume its wall budget."""
    monkeypatch.setenv("AGENT_CORE_LLM_MAX_CONCURRENT", "1")
    monkeypatch.setattr(call_module, "_llm_gate_state", None)
    monkeypatch.setattr(call_module, "_WALL_DEADLINE_FLOOR_S", 0.02)
    holder_started = asyncio.Event()
    release_holder = asyncio.Event()
    calls = 0

    async def _chat(_messages, **_kw):
        nonlocal calls
        calls += 1
        holder_started.set()
        await release_holder.wait()
        return LLMResponse(content="ok")

    llm = SimpleNamespace(chat=_chat, model="fake")
    holder = asyncio.create_task(call_llm(
        llm,
        [user_msg("holder")],
        timeout=1,
        max_retries=1,
        turn=1,
    ))
    await asyncio.wait_for(holder_started.wait(), timeout=1)
    deadline = time.monotonic() + 0.15

    try:
        with pytest.raises(LLMCallExhausted) as exc_info:
            await call_llm(
                llm,
                [user_msg("waiting")],
                timeout=1,
                max_retries=1,
                turn=1,
                wall_deadline_remaining=lambda: deadline - time.monotonic(),
            )
    finally:
        release_holder.set()
        await holder

    exhausted = exc_info.value
    assert calls == 1, "the waiting call must never reach the provider"
    assert exhausted.reason == "wall_deadline"
    assert isinstance(exhausted.last_exc, LLMDeadlineExceeded)
    assert exhausted.last_exc.reason == exhausted.reason
    assert "concurrency-gate" in str(exhausted.last_exc)


@pytest.mark.asyncio
async def test_ordinary_exhaustion_still_carries_the_real_failure() -> None:
    """With no deadline configured, ``exhausted`` keeps pointing at the 429.

    The fix must not push deadline semantics onto the ordinary path: here
    ``last_exc`` IS the cause of the raise, and a chain wrapper reading
    ``rate_limited`` off it is doing the right thing.
    """
    async def _chat(_messages, **_kw):
        raise _RateLimit429

    with pytest.raises(LLMCallExhausted) as exc_info:
        await call_llm(
            SimpleNamespace(chat=_chat, model="fake"),
            [user_msg("hello")],
            timeout=30,
            max_retries=1,
            turn=1,
        )

    assert exc_info.value.reason == "exhausted"
    assert isinstance(exc_info.value.last_exc, _RateLimit429)
    assert exc_info.value.prior_exc is None
