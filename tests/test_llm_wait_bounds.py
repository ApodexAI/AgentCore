"""Admission, tracing and cleanup must not turn a timed-out LLM into a hang."""
from __future__ import annotations

import asyncio
import contextlib
import time
from types import SimpleNamespace

import pytest

from agent_core.components.middleware.llm import proxy as proxy_module
from agent_core.components.middleware.llm import tracing as tracing_module
from agent_core.components.middleware.llm.base import LLMMiddlewareChain
from agent_core.components.middleware.llm.proxy import LLMProxy
from agent_core.components.middleware.llm.tracing import LLMTracingMiddleware
from agent_core.errors import LLMCallExhausted
from agent_core.llm import LLMResponse, StreamDelta
from agent_core.messages import user_msg
from agent_core.runtime.async_utils import await_bounded
from agent_core.runtime.loop import _call as call_module
from agent_core.runtime.loop import _streaming as streaming_module


async def _ignore(*args, **kwargs):
    pass


@pytest.mark.parametrize("external_cancel", [False, True])
async def test_bounded_io_returns_even_when_cancellation_handler_waits(external_cancel):
    started, cancelling, release, finished = (asyncio.Event() for _ in range(4))

    async def stubborn_io():
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelling.set()
            await release.wait()
            raise RuntimeError("late cleanup error") from None
        finally:
            finished.set()

    task = asyncio.create_task(await_bounded(stubborn_io(), 1 if external_cancel else 0.02))
    await started.wait()
    try:
        if external_cancel:
            task.cancel()
        done, _ = await asyncio.wait({task}, timeout=0.2)
        assert task in done
        with pytest.raises(asyncio.CancelledError if external_cancel else TimeoutError):
            await task
        await asyncio.wait_for(cancelling.wait(), 0.2)
        assert not finished.is_set()
    finally:
        release.set()
        await asyncio.wait_for(finished.wait(), 0.2)
        await asyncio.sleep(0)  # Drain the late exception callback.


@pytest.mark.parametrize("mode", ["success", "timeout", "cancel"])
async def test_builtin_tracing_cannot_hold_llm_completion_or_cancellation(monkeypatch, mode):
    monkeypatch.setattr(proxy_module, "_HOOK_TIMEOUT_S", 0.02)
    monkeypatch.setattr(tracing_module, "_TRACE_TIMEOUT_S", 0.02)
    started, trace_started, trace_closed = (asyncio.Event() for _ in range(3))

    class Trace:
        async def log_llm_call(self, **kwargs):
            trace_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                trace_closed.set()

    async def stream(messages, **kwargs):
        started.set()
        if mode != "success":
            await asyncio.Event().wait()
        yield StreamDelta(content="done")

    chain = LLMMiddlewareChain()
    chain.add(LLMTracingMiddleware(Trace()))
    proxy = LLMProxy(SimpleNamespace(model="test", stream=stream), chain)
    task = asyncio.create_task(streaming_module._stream_llm_response(
        proxy, [user_msg("hi")], 0.03 if mode == "timeout" else 1, _ignore,
    ))
    await started.wait()
    try:
        if mode == "cancel":
            task.cancel()
        done, _ = await asyncio.wait({task}, timeout=0.3)
        assert task in done
        if mode == "success":
            assert (await task).content == "done"
        else:
            with pytest.raises(TimeoutError if mode == "timeout" else asyncio.CancelledError):
                await task
        assert trace_started.is_set()
        await asyncio.wait_for(trace_closed.wait(), 0.2)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, TimeoutError):
            await task


async def test_gate_wait_without_optional_deadlines_is_bounded_and_reusable(monkeypatch):
    gate = asyncio.Semaphore(0)
    monkeypatch.setattr(call_module, "_llm_gate", lambda: gate)
    calls = []

    async def chat(messages, **kwargs):
        calls.append(kwargs["timeout"])
        return LLMResponse(content="ok")

    llm = SimpleNamespace(model="test", chat=chat)
    task = asyncio.create_task(call_module.call_llm(llm, [user_msg("hi")], 0.03, 1, 1))
    try:
        done, _ = await asyncio.wait({task}, timeout=0.2)
        assert task in done
        with pytest.raises(LLMCallExhausted) as exc:
            await task
        assert isinstance(exc.value.last_exc, TimeoutError)
        assert calls == []
        assert not gate._waiters  # Timed-out admission must not steal a later slot.
        gate.release()
        response = await call_module.call_llm(llm, [user_msg("hi")], 1, 1, 1)
        assert response.content == "ok" and len(calls) == 1
        assert gate._value == 1
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, LLMCallExhausted):
            await task


async def test_gate_wait_and_provider_share_one_attempt_budget(monkeypatch):
    gate = asyncio.Semaphore(0)
    monkeypatch.setattr(call_module, "_llm_gate", lambda: gate)
    provider_timeouts = []

    async def chat(messages, **kwargs):
        provider_timeouts.append(kwargs["timeout"])
        await asyncio.sleep(0.1)
        return LLMResponse(content="late")

    async def release_slot():
        await asyncio.sleep(0.04)
        gate.release()

    release = asyncio.create_task(release_slot())
    start = time.monotonic()
    with pytest.raises(LLMCallExhausted):
        await call_module.call_llm(SimpleNamespace(model="test", chat=chat),
                                   [user_msg("hi")], 0.09, 1, 1)
    await release
    assert 0 < provider_timeouts[0] < 0.07
    assert time.monotonic() - start < 0.2
    assert gate._value == 1


async def test_stalled_attempt_observer_does_not_prevent_provider_call(monkeypatch):
    monkeypatch.setattr(call_module, "_OBSERVATION_TIMEOUT_S", 0.02)
    cancelled = []

    async def observer(event):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.append(event["phase"])

    async def chat(messages, **kwargs):
        return LLMResponse(content="ok")

    response = await asyncio.wait_for(call_module.call_llm(
        SimpleNamespace(model="test", chat=chat), [user_msg("hi")], 1, 1, 1,
        on_attempt=observer,
    ), 0.3)
    assert response.content == "ok"
    await asyncio.sleep(0)
    assert cancelled == ["started", "finished"]


@pytest.mark.parametrize("protocol", ["anthropic", "openai"])
async def test_real_sdk_blocked_close_cannot_hold_total_timeout(monkeypatch, protocol):
    from agent_core.providers import _stream_activity

    if protocol == "anthropic":
        from test_anthropic_stream_activity import _client

        manager = _client("silent")
    else:
        from test_openai_stream_activity import _client

        manager = _client(("chat", "gpt-5.1"), ("silent",))

    monkeypatch.setattr(_stream_activity, "_CLEANUP_TIMEOUT_S", 0.02)
    monkeypatch.setattr(streaming_module, "_CLEANUP_TIMEOUT_S", 0.04)
    entered, release, finished = (asyncio.Event() for _ in range(3))
    async with manager as (client, body_or_bodies, _requests):
        # OpenAI creates its body lazily; patch the body once headers land.
        stream = client.stream([user_msg("hi")])
        assert (await anext(stream)).transport_activity
        body = body_or_bodies if protocol == "anthropic" else body_or_bodies[0]
        original_close = body.aclose

        async def blocked_close():
            entered.set()
            try:
                while not release.is_set():
                    with contextlib.suppress(asyncio.CancelledError):
                        await release.wait()
                await original_close()
            finally:
                finished.set()

        body.aclose = blocked_close

        class AlreadyOpened:
            def stream(self, *_args, **_kwargs):
                return stream

        task = asyncio.create_task(streaming_module._stream_llm_response(
            AlreadyOpened(), [user_msg("hi")], 0.03, _ignore,
        ))
        try:
            await asyncio.wait_for(entered.wait(), 0.2)
            done, _ = await asyncio.wait({task}, timeout=0.2)
            assert task in done
            with pytest.raises(TimeoutError):
                await task
            assert not finished.is_set()
        finally:
            release.set()
            await asyncio.wait_for(finished.wait(), 0.2)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, TimeoutError):
                await task
            await asyncio.sleep(0)
        assert body.closed and not body.reading


async def test_nonstream_timeout_returns_while_provider_cancellation_unwinds():
    cancelling, release, finished = (asyncio.Event() for _ in range(3))

    async def chat(messages, **kwargs):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelling.set()
            await release.wait()
            raise
        finally:
            finished.set()

    task = asyncio.create_task(call_module.call_llm(
        SimpleNamespace(model="test", chat=chat), [user_msg("hi")], 0.03, 1, 1,
    ))
    try:
        done, _ = await asyncio.wait({task}, timeout=0.2)
        assert task in done
        with pytest.raises(LLMCallExhausted) as exc:
            await task
        assert isinstance(exc.value.last_exc, TimeoutError)
        await asyncio.wait_for(cancelling.wait(), 0.2)
        assert not finished.is_set()
    finally:
        release.set()
        await asyncio.wait_for(finished.wait(), 0.2)
        await asyncio.sleep(0)


async def test_gate_time_is_not_subtracted_twice_from_tool_argument_replay(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(call_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    read_timeouts = []

    class Gate:
        async def acquire(self):
            clock[0] += 40

        def release(self):
            pass

    monkeypatch.setattr(call_module, "_llm_gate", Gate)
    monkeypatch.setattr(call_module, "stream_tool_calls_missing_required_arguments",
                        lambda response, llm: ["search"])

    async def stream_response(llm, messages, timeout, on_delta, **kwargs):
        read_timeouts.append(timeout)
        clock[0] += 30
        return LLMResponse(content="partial")

    async def chat(messages, **kwargs):
        read_timeouts.append(kwargs["timeout"])
        return LLMResponse(content="repaired")

    monkeypatch.setattr(call_module, "_stream_llm_response", stream_response)
    response = await call_module.call_llm(
        SimpleNamespace(model="test", chat=chat), [user_msg("hi")], 100, 1, 1,
        on_delta=_ignore,
    )
    assert read_timeouts == [60, 30]
    assert response.content == "repaired"
    assert response.response_metadata["stream_empty_args_fallback"] is True


async def test_external_cancellation_during_normal_eof_cleanup_propagates(monkeypatch):
    import httpx

    from agent_core.providers import _stream_activity

    entered = asyncio.Event()
    original_wait = _stream_activity.await_bounded

    async def wait_at_reader_cleanup(operation, timeout):
        if isinstance(operation, asyncio.Task):
            entered.set()
            await asyncio.Event().wait()
        return await original_wait(operation, timeout)

    monkeypatch.setattr(_stream_activity, "await_bounded", wait_at_reader_cleanup)

    class Body(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            if False:
                yield b""

        async def aclose(self):
            self.closed = True

    body = Body()

    class Stream:
        response = httpx.Response(200, stream=body)

        async def __aiter__(self):
            if False:
                yield object()

        async def close(self):
            await self.response.aclose()

    async def consume():
        async for _event in _stream_activity.stream_events_with_activity(Stream()):
            pass

    task = asyncio.create_task(consume())
    await asyncio.wait_for(entered.wait(), 0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 0.2)
    assert body.closed


async def test_cancelled_bounded_waiter_retrieves_already_failed_future():
    future = asyncio.get_running_loop().create_future()
    future.set_exception(RuntimeError("operation failed"))
    task = asyncio.create_task(await_bounded(future, 1))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    # asyncio would otherwise report "Future exception was never retrieved".
    assert not future._log_traceback
