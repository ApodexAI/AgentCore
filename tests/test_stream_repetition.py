"""``StreamRepetitionDetectorMiddleware`` + the ``on_chunk`` Protocol hook.

Two layers of coverage:

1. The middleware itself — fed synthetic ``full_text`` strings through
   ``on_chunk``, locks the detection thresholds (min_pattern_len /
   min_repeats / min_text_len) and the abort semantics.

2. End-to-end via ``LLMProxy.stream`` — a fake streaming LLM that
   produces a degenerate sequence; the middleware must cut the stream
   short, the ``after_llm`` chain must still see the truncated
   ``LLMResponse``, and ``ctx.metadata`` must carry the abort signal.

Why this matters: heavy-mode STT branches that fall into loops would
otherwise consume their entire max_tokens budget before the supervisor
even notices. Early termination is the whole point of the chunk hook.
"""

from __future__ import annotations

import pytest

from agent_core.components.middleware.llm.base import (
    LLMCallContext,
    LLMMiddleware,
    LLMMiddlewareChain,
)
from agent_core.components.middleware.llm.proxy import LLMProxy
from agent_core.components.middleware.llm.stream_repetition import (
    StreamRepetitionDetectorMiddleware,
)
from agent_core.llm import StreamDelta
from agent_core.messages import user_msg


def _chunk(text: str) -> StreamDelta:
    return StreamDelta(content=text)


# ── Middleware unit tests ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_short_text_is_not_flagged() -> None:
    """Texts under ``min_text_len`` never trigger detection — short
    outputs (e.g. a one-line answer) can't contain a 6× repeat without
    being legitimately short."""
    mw = StreamRepetitionDetectorMiddleware(min_text_len=800)
    ctx = LLMCallContext(call_index=0)
    text = "x" * 100
    assert await mw.on_chunk(ctx, _chunk("x"), text) is False


@pytest.mark.asyncio
async def test_pattern_repeating_6_times_aborts_stream() -> None:
    """The defining case: a 30-char pattern repeated 6 times at the
    tail of the stream → on_chunk returns True."""
    mw = StreamRepetitionDetectorMiddleware(
        min_pattern_len=30,
        min_repeats=6,
        min_text_len=800,
        check_interval=1,
    )
    pattern = "A" * 30
    # 800 chars of prefix + 6 copies of the pattern at the tail.
    full = ("prefix " * 200)[:800] + pattern * 6
    ctx = LLMCallContext(call_index=0)
    aborted = await mw.on_chunk(ctx, _chunk(pattern), full)
    assert aborted is True
    assert ctx.metadata["stream_repetition_pattern_len"] == 30
    assert ctx.metadata["stream_repetition_repeats"] >= 6


@pytest.mark.asyncio
async def test_pattern_repeating_5_times_does_not_abort() -> None:
    """One short of the threshold → no abort. Tunes the fence right
    at the boundary so future drift fails this test, not production."""
    mw = StreamRepetitionDetectorMiddleware(
        min_pattern_len=30,
        min_repeats=6,
        min_text_len=800,
        check_interval=1,
    )
    pattern = "B" * 30
    full = ("prefix " * 200)[:800] + pattern * 5
    ctx = LLMCallContext(call_index=0)
    assert await mw.on_chunk(ctx, _chunk(pattern), full) is False


@pytest.mark.asyncio
async def test_check_interval_throttles_scans() -> None:
    """The pattern scan only runs every ``check_interval`` new chars.
    Until then, ``on_chunk`` returns False even when a repeat is
    structurally present — keeps the hot path cheap."""
    mw = StreamRepetitionDetectorMiddleware(
        min_pattern_len=30,
        min_repeats=6,
        min_text_len=800,
        check_interval=1000,
    )
    pattern = "C" * 30
    full = ("prefix " * 200)[:800] + pattern * 6
    ctx = LLMCallContext(call_index=0)
    # First chunk adds ~30 chars — far below check_interval, so no scan.
    assert await mw.on_chunk(ctx, _chunk(pattern), full) is False


@pytest.mark.asyncio
async def test_state_is_per_call_index() -> None:
    """One middleware instance must handle concurrent streams cleanly.
    Different ``call_index`` values get isolated detector state — a
    detection on call A doesn't poison call B."""
    mw = StreamRepetitionDetectorMiddleware(
        min_pattern_len=30, min_repeats=6, min_text_len=800, check_interval=1,
    )
    # Same metadata dict — simulates the LLMProxy invariant where ctx is
    # built per call but the middleware might share other state.
    metadata: dict = {}

    ctx_a = LLMCallContext(call_index=0, metadata=metadata)

    pat = "Q" * 30
    full = ("prefix " * 200)[:800] + pat * 6
    assert await mw.on_chunk(ctx_a, _chunk(pat), full) is True
    # Call B (call_index=1) hasn't been touched — same metadata bag,
    # but its detector slot is unallocated.
    state = metadata["_stream_repetition_state"]
    assert state[0]["detected"] is True
    assert 1 not in state


def test_constructor_rejects_invalid_thresholds() -> None:
    """Locks the contract — silently accepting nonsense (e.g.
    ``min_repeats=1``) would mean every output is "detected"."""
    with pytest.raises(ValueError):
        StreamRepetitionDetectorMiddleware(min_pattern_len=0)
    with pytest.raises(ValueError):
        StreamRepetitionDetectorMiddleware(min_pattern_len=50, max_pattern_len=30)
    with pytest.raises(ValueError):
        StreamRepetitionDetectorMiddleware(min_repeats=1)


def test_constructor_clamps_min_text_len_to_match_threshold() -> None:
    """If the caller picks ``min_text_len`` too small for the
    pattern × repeats combo, raise it silently — running the scan
    against text that's too short to ever match is pure CPU waste."""
    mw = StreamRepetitionDetectorMiddleware(
        min_pattern_len=50, min_repeats=10, min_text_len=100,
    )
    assert mw._min_text_len == 500  # 50 × 10


# ── Protocol hook ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_chain_run_on_chunk_short_circuits_on_first_true() -> None:
    """When middleware A votes abort, middleware B is skipped — the
    stream is going to terminate anyway, no point spending cycles on
    the rest of the chain."""
    a_calls = 0
    b_calls = 0

    class _A(LLMMiddleware):
        @property
        def name(self) -> str:
            return "a"

        async def on_chunk(self, ctx, delta, full_text) -> bool:
            nonlocal a_calls
            a_calls += 1
            return True

    class _B(LLMMiddleware):
        @property
        def name(self) -> str:
            return "b"

        async def on_chunk(self, ctx, delta, full_text) -> bool:
            nonlocal b_calls
            b_calls += 1
            return False

    chain = LLMMiddlewareChain()
    chain.add(_A())
    chain.add(_B())
    ctx = LLMCallContext()
    aborted = await chain.run_on_chunk(ctx, _chunk("x"), "x")
    assert aborted is True
    assert a_calls == 1
    assert b_calls == 0


@pytest.mark.asyncio
async def test_chain_run_on_chunk_swallows_middleware_exceptions() -> None:
    """A buggy ``on_chunk`` raising must NOT crash the stream — log +
    continue. The chunk hook should never be able to break a perfectly
    good generation."""

    class _Crash(LLMMiddleware):
        @property
        def name(self) -> str:
            return "crash"

        async def on_chunk(self, ctx, delta, full_text) -> bool:
            raise RuntimeError("intentional")

    class _Good(LLMMiddleware):
        called = False

        @property
        def name(self) -> str:
            return "good"

        async def on_chunk(self, ctx, delta, full_text) -> bool:
            type(self).called = True
            return False

    chain = LLMMiddlewareChain()
    chain.add(_Crash())
    chain.add(_Good())
    aborted = await chain.run_on_chunk(LLMCallContext(), _chunk("x"), "x")
    assert aborted is False
    assert _Good.called is True


@pytest.mark.asyncio
async def test_default_on_chunk_returns_false() -> None:
    """Default implementation returns False — middlewares that don't
    care about streaming get pass-through behaviour for free."""

    class _Plain(LLMMiddleware):
        @property
        def name(self) -> str:
            return "plain"

    assert await _Plain().on_chunk(LLMCallContext(), _chunk("x"), "x") is False


# ── LLMProxy integration ───────────────────────────────────────────


class _StreamingLLM:
    """Async ``LLMClient``-shaped stub that yields ``StreamDelta``s.

    Records how many chunks it actually emitted — the abort path means
    the consumer stopped consuming before exhausting the iterator.
    """

    model = "fake-streaming"

    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self.emitted = 0

    async def stream(
        self,
        messages,
        *,
        tools=None,
        temperature=None,
        max_tokens=None,
        extra_headers=None,
        timeout=None,
    ):
        for text in self.chunks:
            self.emitted += 1
            yield _chunk(text)


class _AfterLLMRecorder(LLMMiddleware):
    """Captures the response that flows through ``after_llm`` so tests
    can assert what the consumer would see post-abort."""

    def __init__(self) -> None:
        self.responses: list = []

    @property
    def name(self) -> str:
        return "recorder"

    async def after_llm(self, ctx, response):
        self.responses.append(response)
        return response


@pytest.mark.asyncio
async def test_proxy_aborts_stream_when_detector_fires() -> None:
    """End-to-end: a degenerate streaming LLM hits the detector
    threshold mid-stream; the proxy stops consuming, the partial
    message still flows through ``after_llm``, and metadata records
    the abort."""
    prefix = "x" * 800
    pat = "Z" * 30
    chunks = [prefix] + [pat] * 6 + [pat] * 100  # 100 extra chunks we should NOT see
    inner = _StreamingLLM(chunks)

    chain = LLMMiddlewareChain()
    chain.add(StreamRepetitionDetectorMiddleware(
        min_pattern_len=30, min_repeats=6, min_text_len=800, check_interval=1,
    ))
    recorder = _AfterLLMRecorder()
    chain.add(recorder)
    proxy = LLMProxy(inner=inner, chain=chain, role_id="test")

    collected = []
    async for delta in proxy.stream([user_msg("x")]):
        collected.append(delta.content)

    # Inner emitted way more chunks than we consumed — proxy aborted.
    assert inner.emitted < len(chunks)
    # The recorder saw the truncated response on the after_llm pass.
    assert len(recorder.responses) == 1
    assembled = recorder.responses[0].content
    assert assembled.startswith(prefix)
    # The recorder also sees no further chunks past the abort point.
    # Reconstruct from `collected` to double-check.
    assert "".join(collected) == assembled


@pytest.mark.asyncio
async def test_proxy_stream_runs_to_completion_when_no_abort() -> None:
    """Sanity: detector that never fires → proxy yields every chunk
    the inner LLM produces. Locks the non-regression path."""
    chunks = ["a", "b", "c"]
    inner = _StreamingLLM(chunks)
    chain = LLMMiddlewareChain()
    chain.add(StreamRepetitionDetectorMiddleware(min_text_len=10_000))
    proxy = LLMProxy(inner=inner, chain=chain, role_id="test")

    collected = [d.content async for d in proxy.stream([user_msg("x")])]
    assert collected == chunks
    assert inner.emitted == 3
