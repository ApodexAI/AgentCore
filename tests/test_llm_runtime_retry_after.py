"""Regression for P2-2 / Step 8: clamp ``Retry-After`` at the same 300s
ceiling as the exponential fallback so a misbehaving provider returning
``Retry-After: 86400`` cannot stall the loop with no output.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agent_core.llm import LLMResponse
from agent_core.messages import user_msg
from agent_core.runtime.loop.llm_client import LLMCallExhausted, call_llm


class _RateLimit429(Exception):
    """Mimics the shape ``_get_retry_after`` looks for: ``response.headers``."""

    def __init__(self, retry_after: float) -> None:
        super().__init__("rate limited")
        self.status_code = 429
        self.response = SimpleNamespace(
            headers={"retry-after": str(retry_after)},
        )


@pytest.mark.asyncio
async def test_retry_after_huge_value_is_clamped(monkeypatch):
    """``Retry-After: 86400`` must not sleep the loop for a day.

    Without the clamp, a single hostile or buggy upstream response would
    silently freeze the agent loop for ``Retry-After`` seconds — the
    exact "no error, no progress" pattern the swarm hang audit (P2-2)
    flagged.
    """
    sleeps: list[float] = []

    real_sleep = asyncio.sleep

    async def _record(duration):
        sleeps.append(duration)
        # Yield the loop without burning real time so the test stays
        # deterministic; ``asyncio.wait_for`` and friends only need a
        # zero-duration yield, not the requested duration.
        await real_sleep(0)

    monkeypatch.setattr("asyncio.sleep", _record)

    call_count = 0

    async def _chat(_messages, **_kw):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _RateLimit429(retry_after=86400)  # one full day
        return LLMResponse(content="ok")

    fake_llm = SimpleNamespace(chat=_chat)

    result = await call_llm(
        fake_llm,
        [user_msg("hi")],
        timeout=10,
        max_retries=3,
        turn=0,
    )

    assert result is not None and result.content == "ok"
    # The retry path slept once between attempt 1 (429) and attempt 2.
    # Filter out 0-duration yields the test injects via ``real_sleep(0)``.
    backoffs = [s for s in sleeps if s > 0]
    assert backoffs, f"expected a backoff sleep, got {sleeps!r}"
    assert all(s <= 300 for s in backoffs), (
        f"backoff exceeded 300s ceiling: {backoffs!r}"
    )


@pytest.mark.asyncio
async def test_retry_after_within_ceiling_unchanged(monkeypatch):
    """``Retry-After`` values ≤300 are honoured verbatim — clamp is a
    ceiling, not a floor.
    """
    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def _record(duration):
        sleeps.append(duration)
        await real_sleep(0)

    monkeypatch.setattr("asyncio.sleep", _record)

    call_count = 0

    async def _chat(_messages, **_kw):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _RateLimit429(retry_after=42)
        return LLMResponse(content="ok")

    fake_llm = SimpleNamespace(chat=_chat)

    result = await call_llm(
        fake_llm,
        [user_msg("hi")],
        timeout=10,
        max_retries=3,
        turn=0,
    )
    assert result is not None
    backoffs = [s for s in sleeps if s > 0]
    assert 42 in backoffs, (
        f"expected literal Retry-After=42 in backoffs, got {backoffs!r}"
    )


@pytest.mark.asyncio
async def test_default_backoff_is_exponential_on_timeout(monkeypatch):
    """``retry_wait_fixed=None`` (default) → timeouts back off on the
    2/4/8s exponential base with ±25% jitter (jitter de-synchronises
    retry stampedes across parallel runs — partial3 saw timed-out
    attempts re-collide 20 minutes later without it).

    Regression: a prior refactor flattened the default to 30s flat which
    over-paused short transient errors. Pin the schedule shape so other
    workflows keep mirothinker's non-flat behavior.
    """
    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def _record(duration):
        sleeps.append(duration)
        await real_sleep(0)

    monkeypatch.setattr("asyncio.sleep", _record)

    async def _chat_timeout(_messages, **_kw):
        raise TimeoutError("simulated stream timeout")

    fake_llm = SimpleNamespace(chat=_chat_timeout)

    with pytest.raises(LLMCallExhausted) as exc_info:
        await call_llm(
            fake_llm,
            [user_msg("hi")],
            timeout=10,
            max_retries=4,
            turn=0,
        )
    assert exc_info.value.reason == "exhausted"
    backoffs = [s for s in sleeps if s > 0]
    # Base min(2 * 2**attempt, 60) for attempts 0..2 (last attempt has
    # no sleep), ±25% jitter → assert each within its jitter band.
    assert len(backoffs) == 3, f"expected 3 backoffs, got {backoffs!r}"
    for got, base in zip(backoffs, [2, 4, 8], strict=False):
        assert base * 0.75 <= got <= base * 1.25, (
            f"backoff {got:.2f}s outside jitter band of base {base}s "
            f"(full schedule: {backoffs!r})"
        )


@pytest.mark.asyncio
async def test_retry_wait_fixed_overrides_default(monkeypatch):
    """Workflows that opt into ``retry_wait_fixed=N`` get N-second flat waits."""
    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def _record(duration):
        sleeps.append(duration)
        await real_sleep(0)

    monkeypatch.setattr("asyncio.sleep", _record)

    async def _chat_timeout(_messages, **_kw):
        raise TimeoutError()

    fake_llm = SimpleNamespace(chat=_chat_timeout)

    with pytest.raises(LLMCallExhausted):
        await call_llm(
            fake_llm,
            [user_msg("hi")],
            timeout=10,
            max_retries=3,
            turn=0,
            retry_wait_fixed=90,
        )
    backoffs = [int(s) for s in sleeps if s > 0]
    assert backoffs == [90, 90], (
        f"expected fixed 90s waits, got {backoffs!r}"
    )
