"""``LLMProxy.stream`` must call ``after_llm`` exactly once per call.

The ``finally`` that emits ``run_after`` used to sit *inside* the retry
``while True``, so a stream that failed before its first chunk and was retried
handed every middleware an empty ``LLMResponse`` and then the real one. Any
middleware that accumulates — cost/usage accounting, history appending —
double-counted, and the phantom empty response looked like a real (empty)
answer to anything reading content.

Also pinned here: an execution scope is shared by concurrent ``chat`` /
``stream`` calls, so per-call trace ids must stay on the call's own context
instead of being written back onto the scope.
"""

from __future__ import annotations

import asyncio

import pytest

from agent_core.components.middleware.llm.base import (
    LLMCallContext,
    LLMMiddleware,
    LLMMiddlewareChain,
)
from agent_core.components.middleware.llm.proxy import LLMProxy
from agent_core.execution_context import (
    build_execution_scope,
    reset_current_execution_scope,
    set_current_execution_scope,
)
from agent_core.llm import LLMResponse, StreamDelta
from agent_core.messages import user_msg


class _Recorder(LLMMiddleware):
    """Records every after_llm, and can demand one retry."""

    name = "recorder"

    def __init__(self, *, retries: int = 0) -> None:
        super().__init__()
        self.after_calls: list[str] = []
        self.error_calls: int = 0
        self._retries_left = retries

    async def after_llm(
        self, ctx: LLMCallContext, response: LLMResponse,
    ) -> LLMResponse:
        self.after_calls.append(response.content)
        return response

    async def on_llm_error(
        self, ctx: LLMCallContext, error: Exception, attempt: int,
    ) -> bool:
        self.error_calls += 1
        if self._retries_left > 0:
            self._retries_left -= 1
            return True
        return False


class _Inner:
    """Fails the first ``fail_times`` streams before its first chunk."""

    model = "test-model"

    def __init__(self, fail_times: int = 0) -> None:
        self.fail_times = fail_times
        self.stream_calls = 0

    async def stream(self, messages, **kwargs):
        self.stream_calls += 1
        if self.stream_calls <= self.fail_times:
            raise RuntimeError("upstream reset")
        for piece in ("he", "llo"):
            yield StreamDelta(content=piece)


def _chain(mw: LLMMiddleware) -> LLMMiddlewareChain:
    chain = LLMMiddlewareChain()
    chain.add(mw)
    return chain


async def _drain(proxy: LLMProxy) -> str:
    return "".join([d.content or "" async for d in proxy.stream([user_msg("q")])])


@pytest.mark.asyncio
async def test_after_llm_fires_once_on_a_clean_stream():
    rec = _Recorder()
    proxy = LLMProxy(inner=_Inner(), chain=_chain(rec), role_id="r")

    assert await _drain(proxy) == "hello"
    assert rec.after_calls == ["hello"]


@pytest.mark.asyncio
async def test_after_llm_fires_once_across_a_retry():
    """The empty first attempt must NOT reach after_llm."""
    rec = _Recorder(retries=1)
    inner = _Inner(fail_times=1)
    proxy = LLMProxy(inner=inner, chain=_chain(rec), role_id="r")

    assert await _drain(proxy) == "hello"
    assert inner.stream_calls == 2
    assert rec.error_calls == 1
    assert rec.after_calls == ["hello"]


@pytest.mark.asyncio
async def test_after_llm_still_fires_once_when_every_attempt_fails():
    rec = _Recorder(retries=1)
    inner = _Inner(fail_times=99)
    proxy = LLMProxy(inner=inner, chain=_chain(rec), role_id="r")

    with pytest.raises(RuntimeError, match="upstream reset"):
        await _drain(proxy)

    assert inner.stream_calls == 2
    assert rec.after_calls == [""]


@pytest.mark.asyncio
async def test_failure_metadata_describes_the_last_attempt():
    rec = _Recorder()
    proxy = LLMProxy(inner=_Inner(fail_times=1), chain=_chain(rec), role_id="r")
    seen: dict[str, object] = {}

    class _Capture(LLMMiddleware):
        name = "capture"

        async def after_llm(self, ctx, response):
            seen.update(ctx.metadata)
            return response

    proxy.chain.add(_Capture())
    with pytest.raises(RuntimeError):
        await _drain(proxy)

    assert seen.get("error") == "upstream reset"
    assert "duration_ms" in seen


# ── per-call trace ids stay off the shared scope ──────────────────────────


@pytest.mark.asyncio
async def test_concurrent_calls_do_not_clobber_each_others_step_id():
    scope = build_execution_scope(
        task_id="t1", phase_id="research", role_id="r", state={},
    )
    token = set_current_execution_scope(scope)
    try:
        proxy = LLMProxy(inner=_Inner(), chain=LLMMiddlewareChain(), role_id="r")
        ctx_a = proxy._make_ctx(proxy._next_call_index())
        ctx_b = proxy._make_ctx(proxy._next_call_index())

        assert ctx_a.metadata["step_id"] != ctx_b.metadata["step_id"]
        assert ctx_a.metadata["prompt_id"] != ctx_b.metadata["prompt_id"]
        # The second call must not have rewritten the first one's identity.
        assert ctx_a.metadata["step_id"] == "research:llm:1"
        assert ctx_b.metadata["step_id"] == "research:llm:2"
        # ...and neither leaked a per-call id onto the shared scope.
        assert "step_id" not in scope.metadata
        assert "prompt_id" not in scope.metadata
        # The scope-stable identifier is still published once.
        assert scope.metadata["session_id"] == ctx_a.metadata["session_id"]
    finally:
        reset_current_execution_scope(token)


@pytest.mark.asyncio
async def test_interleaved_streams_keep_their_own_step_id():
    """The race the scope write created, exercised through the public API."""
    scope = build_execution_scope(
        task_id="t1", phase_id="research", role_id="r", state={},
    )
    token = set_current_execution_scope(scope)
    try:
        steps: list[str] = []

        class _Step(LLMMiddleware):
            name = "step"

            async def after_llm(self, ctx, response):
                steps.append(str(ctx.metadata.get("step_id")))
                return response

        proxy = LLMProxy(inner=_Inner(), chain=_chain(_Step()), role_id="r")
        await asyncio.gather(_drain(proxy), _drain(proxy))

        assert sorted(steps) == ["research:llm:1", "research:llm:2"]
    finally:
        reset_current_execution_scope(token)
