"""Stream-stall watchdog in ``call_llm`` / ``_stream_llm_response``.

2026-06-05 heavy-trace forensics (partial3.json): 12 streaming attempts
against the apodex gateway went silent mid-flight — no chunks, no
error — and each pinned its call for the full 1200 s timeout, driving
4/8 runs into their wall deadline. Decode throughput on *successful*
calls was a steady 110-120 tok/s (a full-cap generation finishes in
~155 s), so a silent stream is a dead stream, not a slow one.

The watchdog bounds the gap between consecutive stream chunks
(``MIROHARNESS_LLM_STREAM_STALL_S``, default 180 s, <= 0 disables) and
raises :class:`LLMStreamStalled` — a ``TimeoutError`` subclass — so the
existing transient retry/backoff budget handles the redo.
"""

from __future__ import annotations

import asyncio

import pytest

import agent_core.runtime.loop._streaming as llm_client
from agent_core.llm import StreamDelta
from agent_core.messages import user_msg
from agent_core.runtime.loop.llm_client import (
    LLMCallExhausted,
    LLMStreamStalled,
    call_llm,
)

MSGS = [user_msg("q")]


async def _sink_delta(*_a, **_kw) -> None:
    """Minimal on_delta — presence enables the streaming path."""


class _StallingLLM:
    """Streams ``preamble`` chunks then hangs forever (silent stream).

    ``hang_attempts`` controls how many attempts hang before a healthy
    one — models the partial3 pattern where a retry after the stall
    succeeds at full speed.
    """

    def __init__(self, hang_attempts: int = 99, preamble: int = 1) -> None:
        self.hang_attempts = hang_attempts
        self.preamble = preamble
        self.calls = 0

    async def stream(self, messages, **_kw):
        self.calls += 1
        for i in range(self.preamble):
            yield StreamDelta(content=f"chunk{i} ")
        if self.calls <= self.hang_attempts:
            await asyncio.sleep(3600)  # gateway black-hole: no chunks, no error
        else:
            yield StreamDelta(content="recovered answer")


class _HealthyGappyLLM:
    """Streams with inter-chunk gaps below the stall threshold."""

    async def stream(self, messages, **_kw):
        for i in range(3):
            await asyncio.sleep(0.02)
            yield StreamDelta(content=f"part{i} ")


@pytest.mark.asyncio
async def test_stall_aborts_attempt_and_exhausts(monkeypatch) -> None:
    """A permanently silent stream dies at the stall bound (not the full
    call timeout) on every attempt, then surfaces as ``exhausted`` with
    the stall as ``last_exc`` — the chain wrapper sees WHY it died."""
    monkeypatch.setenv("MIROHARNESS_LLM_STREAM_STALL_S", "0.1")
    llm = _StallingLLM()
    with pytest.raises(LLMCallExhausted) as exc_info:
        await call_llm(
            llm, MSGS, timeout=30, max_retries=2, turn=1,
            on_delta=_sink_delta, retry_wait_fixed=0,
        )
    assert exc_info.value.reason == "exhausted"
    assert isinstance(exc_info.value.last_exc, LLMStreamStalled)
    assert exc_info.value.last_exc.chunks_seen == 1
    assert llm.calls == 2  # both attempts ran and stalled


@pytest.mark.asyncio
async def test_stall_chain_advances_after_threshold(monkeypatch) -> None:
    """Under an active provider chain, repeated stalls stop same-key
    retrying and surface ``chain_advance`` at the threshold (default 2)
    instead of burning the whole retry budget on a black-holed endpoint.
    fail1: 56 stalls × ~180-330 s wasted on one dead 397b gateway."""
    monkeypatch.setenv("MIROHARNESS_LLM_STREAM_STALL_S", "0.1")
    llm = _StallingLLM()  # hangs forever
    with pytest.raises(LLMCallExhausted) as exc_info:
        await call_llm(
            llm, MSGS, timeout=30, max_retries=5, turn=1,
            on_delta=_sink_delta, retry_wait_fixed=0,
            chain_fallback_active=lambda: True,
        )
    assert exc_info.value.reason == "chain_advance"
    assert isinstance(exc_info.value.last_exc, LLMStreamStalled)
    # Advanced at the 2nd stall — did NOT consume the full 5-attempt budget.
    assert llm.calls == 2


@pytest.mark.asyncio
async def test_stall_no_chain_retries_to_budget(monkeypatch) -> None:
    """With no outer chain there's nothing to advance to, so stalls keep
    same-key retrying to the budget and surface ``exhausted`` (the
    single-endpoint swarm/benchmark floor is unchanged)."""
    monkeypatch.setenv("MIROHARNESS_LLM_STREAM_STALL_S", "0.1")
    llm = _StallingLLM()
    with pytest.raises(LLMCallExhausted) as exc_info:
        await call_llm(
            llm, MSGS, timeout=30, max_retries=4, turn=1,
            on_delta=_sink_delta, retry_wait_fixed=0,
        )
    assert exc_info.value.reason == "exhausted"
    assert llm.calls == 4  # all attempts ran, no early chain advance


@pytest.mark.asyncio
async def test_stall_chain_advance_disabled_by_env(monkeypatch) -> None:
    """``MIROHARNESS_LLM_STREAM_STALL_MAX=0`` disables the escape even
    under a chain: stalls retry to the budget then surface ``exhausted``."""
    monkeypatch.setenv("MIROHARNESS_LLM_STREAM_STALL_S", "0.1")
    monkeypatch.setenv("MIROHARNESS_LLM_STREAM_STALL_MAX", "0")
    llm = _StallingLLM()
    with pytest.raises(LLMCallExhausted) as exc_info:
        await call_llm(
            llm, MSGS, timeout=30, max_retries=3, turn=1,
            on_delta=_sink_delta, retry_wait_fixed=0,
            chain_fallback_active=lambda: True,
        )
    assert exc_info.value.reason == "exhausted"
    assert llm.calls == 3


@pytest.mark.asyncio
async def test_stall_then_retry_recovers(monkeypatch) -> None:
    """First attempt stalls, second streams normally — the retry budget
    converts a 1200 s black-hole into one stall-bound redo."""
    monkeypatch.setenv("MIROHARNESS_LLM_STREAM_STALL_S", "0.1")
    llm = _StallingLLM(hang_attempts=1)
    response = await call_llm(
        llm, MSGS, timeout=30, max_retries=3, turn=1,
        on_delta=_sink_delta, retry_wait_fixed=0,
    )
    assert response is not None
    assert "recovered answer" in response.content
    assert llm.calls == 2


@pytest.mark.asyncio
async def test_slow_but_alive_stream_not_flagged(monkeypatch) -> None:
    """Inter-chunk gaps below the bound never trip the watchdog — slow
    decode is healthy, only silence is pathological."""
    monkeypatch.setenv("MIROHARNESS_LLM_STREAM_STALL_S", "0.5")
    response = await call_llm(
        _HealthyGappyLLM(), MSGS, timeout=30, max_retries=1, turn=1,
        on_delta=_sink_delta, retry_wait_fixed=0,
    )
    assert response is not None
    assert response.content == "part0 part1 part2 "


@pytest.mark.asyncio
async def test_stall_disabled_falls_back_to_total_timeout(monkeypatch) -> None:
    """``MIROHARNESS_LLM_STREAM_STALL_S=0`` disables the watchdog: the
    silent stream then dies at the (plain) total call timeout."""
    monkeypatch.setenv("MIROHARNESS_LLM_STREAM_STALL_S", "0")
    llm = _StallingLLM()
    with pytest.raises(LLMCallExhausted) as exc_info:
        await call_llm(
            llm, MSGS, timeout=1, max_retries=1, turn=1,
            on_delta=_sink_delta, retry_wait_fixed=0,
        )
    assert isinstance(exc_info.value.last_exc, asyncio.TimeoutError)
    assert not isinstance(exc_info.value.last_exc, LLMStreamStalled)


@pytest.mark.asyncio
async def test_stall_default_and_invalid_env(monkeypatch) -> None:
    """No env → documented default; junk env → default (with a warning),
    never a crash in the hot path."""
    monkeypatch.delenv("MIROHARNESS_LLM_STREAM_STALL_S", raising=False)
    assert llm_client._stream_stall_timeout_s() == 180.0
    monkeypatch.setenv("MIROHARNESS_LLM_STREAM_STALL_S", "not-a-number")
    assert llm_client._stream_stall_timeout_s() == 180.0


@pytest.mark.asyncio
async def test_first_chunk_bound_fires_before_stall(monkeypatch) -> None:
    """With the TTFT knob set, a stream that never produces its FIRST
    chunk dies at the tight first-chunk bound — not the loose stall
    bound (healthy TTFT is seconds; zero chunks after the bound means
    black-holed, not thinking)."""
    monkeypatch.setenv("MIROHARNESS_LLM_STREAM_STALL_S", "30")
    monkeypatch.setenv("MIROHARNESS_LLM_FIRST_CHUNK_S", "0.1")
    llm = _StallingLLM(preamble=0)  # hangs before any chunk
    loop = asyncio.get_running_loop()
    started = loop.time()
    with pytest.raises(LLMCallExhausted) as exc_info:
        await call_llm(
            llm, MSGS, timeout=600, max_retries=1, turn=1,
            on_delta=_sink_delta, retry_wait_fixed=0,
        )
    assert isinstance(exc_info.value.last_exc, LLMStreamStalled)
    assert exc_info.value.last_exc.chunks_seen == 0
    assert exc_info.value.last_exc.stall_s == pytest.approx(0.1)
    assert loop.time() - started < 5, "must die at TTFT bound, not stall/timeout"


@pytest.mark.asyncio
async def test_first_chunk_bound_relaxes_after_first_chunk(monkeypatch) -> None:
    """Once the first chunk arrives, inter-chunk gaps are judged by the
    looser stall bound — gaps longer than the TTFT bound but below the
    stall bound are healthy (mid-generation pauses are legitimate)."""
    monkeypatch.setenv("MIROHARNESS_LLM_STREAM_STALL_S", "0.5")
    monkeypatch.setenv("MIROHARNESS_LLM_FIRST_CHUNK_S", "0.05")

    class _SlowAfterFirst:
        async def stream(self, messages, **_kw):
            yield StreamDelta(content="fast ")  # TTFT well under 0.05
            await asyncio.sleep(0.2)  # > first_s, < stall_s — must NOT flag
            yield StreamDelta(content="slow")

    response = await call_llm(
        _SlowAfterFirst(), MSGS, timeout=30, max_retries=1, turn=1,
        on_delta=_sink_delta, retry_wait_fixed=0,
    )
    assert response is not None
    assert response.content == "fast slow"


@pytest.mark.asyncio
async def test_first_chunk_only_mode_disarms_after_first(monkeypatch) -> None:
    """TTFT bound with the stall watchdog disabled: the scope disarms
    after the first chunk, so a later silence falls through to the
    plain total call timeout (not LLMStreamStalled)."""
    monkeypatch.setenv("MIROHARNESS_LLM_STREAM_STALL_S", "0")
    monkeypatch.setenv("MIROHARNESS_LLM_FIRST_CHUNK_S", "0.05")
    llm = _StallingLLM()  # 1 preamble chunk, then silent forever
    with pytest.raises(LLMCallExhausted) as exc_info:
        await call_llm(
            llm, MSGS, timeout=1, max_retries=1, turn=1,
            on_delta=_sink_delta, retry_wait_fixed=0,
        )
    assert isinstance(exc_info.value.last_exc, asyncio.TimeoutError)
    assert not isinstance(exc_info.value.last_exc, LLMStreamStalled)


@pytest.mark.asyncio
async def test_first_chunk_per_call_param_wins_over_env(monkeypatch) -> None:
    """``call_llm(first_chunk_s=...)`` (LoopConfig ← profile
    ``agent.first_chunk_s``) overrides the process-wide env knob; an
    explicit 0 disables even when the env arms it."""
    monkeypatch.setenv("MIROHARNESS_LLM_STREAM_STALL_S", "30")
    monkeypatch.setenv("MIROHARNESS_LLM_FIRST_CHUNK_S", "30")
    # Param tightens past the env: dies at 0.1, not 30.
    llm = _StallingLLM(preamble=0)
    loop = asyncio.get_running_loop()
    started = loop.time()
    with pytest.raises(LLMCallExhausted) as exc_info:
        await call_llm(
            llm, MSGS, timeout=600, max_retries=1, turn=1,
            on_delta=_sink_delta, retry_wait_fixed=0, first_chunk_s=0.1,
        )
    assert isinstance(exc_info.value.last_exc, LLMStreamStalled)
    assert loop.time() - started < 5

    # Param 0 disables the TTFT bound even though the env sets 30: the
    # first chunk is then judged by the stall bound (0.2 here).
    monkeypatch.setenv("MIROHARNESS_LLM_STREAM_STALL_S", "0.2")
    started = loop.time()
    with pytest.raises(LLMCallExhausted) as exc_info:
        await call_llm(
            _StallingLLM(preamble=0), MSGS, timeout=600, max_retries=1,
            turn=1, on_delta=_sink_delta, retry_wait_fixed=0,
            first_chunk_s=0,
        )
    assert isinstance(exc_info.value.last_exc, LLMStreamStalled)
    assert exc_info.value.last_exc.stall_s == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_first_chunk_default_off_and_invalid_env(monkeypatch) -> None:
    """No env → disabled (first chunk rides the stall bound); junk env
    → disabled, never a crash in the hot path."""
    monkeypatch.delenv("MIROHARNESS_LLM_FIRST_CHUNK_S", raising=False)
    assert llm_client._first_chunk_timeout_s() == 0.0
    monkeypatch.setenv("MIROHARNESS_LLM_FIRST_CHUNK_S", "not-a-number")
    assert llm_client._first_chunk_timeout_s() == 0.0


@pytest.mark.asyncio
async def test_stall_closes_stream_promptly(monkeypatch) -> None:
    """The stall path acloses the chunk generator before raising so the
    underlying HTTP stream is released immediately (the httpcore-leak
    family from the mm1 incident), and the whole failure takes ~stall_s,
    not ~timeout."""
    monkeypatch.setenv("MIROHARNESS_LLM_STREAM_STALL_S", "0.1")
    closed = asyncio.Event()

    class _TrackingLLM(_StallingLLM):
        async def stream(self, messages, **_kw):
            try:
                yield StreamDelta(content="x")
                await asyncio.sleep(3600)
            finally:
                closed.set()

    loop = asyncio.get_running_loop()
    started = loop.time()
    with pytest.raises(LLMCallExhausted):
        await call_llm(
            _TrackingLLM(), MSGS, timeout=600, max_retries=1, turn=1,
            on_delta=_sink_delta, retry_wait_fixed=0,
        )
    assert closed.is_set(), "generator finally must run (stream released)"
    assert loop.time() - started < 5, "must fail at stall bound, not timeout"
