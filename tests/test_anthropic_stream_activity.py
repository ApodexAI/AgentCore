"""Real SDK SSE parsing must preserve transport liveness during omitted thinking."""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import pytest
from anthropic import AsyncAnthropic, DefaultAsyncHttpxClient

from agent_core.errors import LLMStreamStalled
from agent_core.messages import user_msg
from agent_core.providers.anthropic import AnthropicClient
from agent_core.providers.fallback import FallbackEntry, LLMFallbackChain
from agent_core.runtime.loop._streaming import _stream_llm_response

if issubclass(DefaultAsyncHttpxClient, httpx.AsyncClient):
    sdk_httpx = httpx
else:
    import httpx2 as sdk_httpx


def _sse(event):
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


def _opening():
    return {"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-x",
        "content": [], "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 0},
    }}


def _ending():
    return [
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "signature_delta", "signature": "signed"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "text_delta", "text": "done"}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {
            "stop_reason": "end_turn", "stop_sequence": None,
        }, "usage": {"output_tokens": 20}},
        {"type": "message_stop"},
    ]


class _Body(sdk_httpx.AsyncByteStream):
    def __init__(self, mode, ending=None):
        self.mode = mode
        self.ending = _ending() if ending is None else ending
        self.closed = False
        self.reading = False
        self.pings = 0

    async def __aiter__(self):
        self.reading = True
        try:
            yield _sse(_opening())
            yield _sse({"type": "content_block_start", "index": 0,
                        "content_block": {"type": "thinking", "thinking": "", "signature": ""}})
            if self.mode == "silent":
                await asyncio.sleep(10)
            for _ in range(30):
                await asyncio.sleep(0.01)
                self.pings += 1
                yield _sse({"type": "ping"})
            if self.mode == "error":
                yield _sse({"type": "error", "error": {
                    "type": "overloaded_error", "message": "busy",
                }})
            else:
                for event in self.ending:
                    yield _sse(event)
        finally:
            self.reading = False

    async def aclose(self):
        self.closed = True


@asynccontextmanager
async def _client(mode="healthy"):
    body = _Body(mode)
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return sdk_httpx.Response(200, stream=body, headers={"content-type": "text/event-stream"})

    client = AnthropicClient("claude-x", api_key="k",
                             thinking={"type": "adaptive", "display": "omitted"})
    await client._client.close()
    async with AsyncAnthropic(api_key="k", max_retries=0, http_client=sdk_httpx.AsyncClient(
        transport=sdk_httpx.MockTransport(respond),
    )) as sdk:
        client._client = sdk
        yield client, body, requests


async def _ignore(*_args, **_kwargs):
    pass


@pytest.fixture(autouse=True)
def _short_stall(monkeypatch):
    monkeypatch.setattr("agent_core.runtime.loop._streaming._stream_stall_timeout_s", lambda: 0.1)


async def test_omitted_thinking_heartbeats_outlive_stall_window_without_observer_noise():
    observed = []

    async def on_delta(text, _acc, _index, thinking, **kwargs):
        observed.append((text, thinking, kwargs["tool_call_args_chunks"]))

    async with _client() as (client, body, requests):
        response = await _stream_llm_response(client, [user_msg("hi")], 2, on_delta)
    assert body.pings == 30  # 0.3s thinking is longer than the 0.1s stall window.
    assert body.closed and not body.reading
    assert observed == [("done", "", [])]
    assert response.content == [
        {"type": "thinking", "thinking": "", "signature": "signed"},
        {"type": "text", "text": "done"},
    ]
    assert response.usage["completion_tokens"] == 20
    assert response.finish_reason == "end_turn"
    assert requests[0]["stream"] is True
    assert requests[0]["thinking"]["display"] == "omitted"


async def test_silent_socket_still_stalls_and_cleans_up_reader():
    async with _client("silent") as (client, body, _requests):
        with pytest.raises(LLMStreamStalled):
            await _stream_llm_response(client, [user_msg("hi")], 2, _ignore)
        assert body.closed and not body.reading


async def test_heartbeats_do_not_extend_total_call_timeout():
    async with _client() as (client, body, _requests):
        with pytest.raises(TimeoutError):
            await _stream_llm_response(client, [user_msg("hi")], 0.15, _ignore)
        assert body.pings > 0
        assert body.closed and not body.reading


async def test_external_cancellation_closes_response_and_reader():
    async with _client() as (client, body, _requests):
        task = asyncio.create_task(_stream_llm_response(client, [user_msg("hi")], 2, _ignore))
        while body.pings == 0:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert body.closed and not body.reading


async def test_early_consumer_close_also_closes_reader():
    async with _client() as (client, body, _requests):
        stream = client.stream([user_msg("hi")])
        assert (await anext(stream)).transport_activity is True
        await stream.aclose()
        assert body.closed and not body.reading


async def test_heartbeat_does_not_commit_fallback_leg_before_sdk_error():
    async with _client("error") as (primary, body, _requests), _client() as (fallback, _, _):
        chain = LLMFallbackChain(entries=[
            FallbackEntry(model=primary, provider="primary", triggers=("any_error",)),
            FallbackEntry(model=fallback, provider="fallback"),
        ])
        response = await _stream_llm_response(chain, [user_msg("hi")], 2, _ignore)
    assert body.closed and not body.reading
    assert response.response_metadata["provider_actually_used"] == "fallback"
    assert response.finish_reason == "end_turn"


async def test_proxy_smoke_retries_sdk_error_after_heartbeats_before_output():
    from agent_core.components.middleware.llm.base import LLMMiddleware, LLMMiddlewareChain
    from agent_core.components.middleware.llm.proxy import LLMProxy

    bodies, requests, chunks, after_calls, errors = [], [], [], [], []

    class RetryRecorder(LLMMiddleware):
        name = "retry_recorder"

        async def on_llm_error(self, ctx, error, attempt):
            errors.append((error, attempt))
            return attempt == 0

        async def on_chunk(self, ctx, delta, accumulated):
            chunks.append(delta)
            return False

        async def after_llm(self, ctx, response):
            after_calls.append((response, dict(ctx.metadata)))
            return response

    def respond(request):
        requests.append(json.loads(request.content))
        body = _Body("error" if len(requests) == 1 else "healthy")
        bodies.append(body)
        return sdk_httpx.Response(200, stream=body, headers={"content-type": "text/event-stream"})

    client = AnthropicClient("claude-x", api_key="k",
                             thinking={"type": "adaptive", "display": "omitted"})
    await client._client.close()
    async with AsyncAnthropic(api_key="k", max_retries=0, http_client=sdk_httpx.AsyncClient(
        transport=sdk_httpx.MockTransport(respond),
    )) as sdk:
        client._client = sdk
        chain = LLMMiddlewareChain()
        chain.add(RetryRecorder())
        proxy = LLMProxy(client, chain)
        response = await _stream_llm_response(proxy, [user_msg("hi")], 2, _ignore)

    assert len(requests) == 2
    assert requests[0] == requests[1]
    assert len(errors) == 1 and errors[0][1] == 0
    assert all(body.pings == 30 and body.closed and not body.reading for body in bodies)
    assert all(not delta.transport_activity for delta in chunks)
    assert len(after_calls) == 1
    assert after_calls[0][0].content == "done"
    assert "error" not in after_calls[0][1]
    assert response.usage["completion_tokens"] == 20
    assert response.finish_reason == "end_turn"
    assert response.content[-1] == {"type": "text", "text": "done"}


async def test_loop_smoke_omitted_thinking_tool_then_final_answer():
    from agent_core.loop_types import LoopConfig, LoopPolicy
    from agent_core.runtime.loop.agent_loop import run_agent_loop
    from agent_core.runtime.loop.model_profile import ModelProfile

    requests, bodies, tools_called = [], [], []

    class Search:
        name = "search"

        async def ainvoke(self, args):
            tools_called.append(args)
            return "found"

        def to_openai_schema(self):
            return {"type": "function", "function": {
                "name": self.name, "parameters": {"type": "object"},
            }}

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        ending = _ending()
        if len(requests) == 1:
            ending = [
                ending[0], ending[1],
                {"type": "content_block_start", "index": 1, "content_block": {
                    "type": "tool_use", "id": "call_1", "name": "search", "input": {},
                }},
                {"type": "content_block_delta", "index": 1, "delta": {
                    "type": "input_json_delta", "partial_json": '{"q":"test"}',
                }},
                {"type": "content_block_stop", "index": 1},
                {"type": "message_delta", "delta": {
                    "stop_reason": "tool_use", "stop_sequence": None,
                }, "usage": {"output_tokens": 20}},
                {"type": "message_stop"},
            ]
        body = _Body("healthy", ending)
        bodies.append(body)
        return sdk_httpx.Response(200, stream=body, headers={"content-type": "text/event-stream"})

    client = AnthropicClient("claude-x", api_key="k",
                             thinking={"type": "adaptive", "display": "omitted"})
    await client._client.close()
    async with AsyncAnthropic(api_key="k", max_retries=0, http_client=sdk_httpx.AsyncClient(
        transport=sdk_httpx.MockTransport(respond),
    )) as sdk:
        client._client = sdk
        result = await run_agent_loop(
            system_prompt="s", user_message="hi", llm=client, tools=[Search()],
            config=LoopConfig(max_turns=3, max_llm_retries=1, llm_timeout=2,
                              stream_llm_tokens=True,
                              loop_policy=LoopPolicy(no_tool_behavior="stop")),
            model_profile=ModelProfile(model_id="claude-x", provider="anthropic",
                                       protocol="anthropic", thinking_format="content_block"),
        )
    assert result.final_content == "done"
    assert tools_called == [{"q": "test"}]
    assert len(requests) == 2
    assert all(request["stream"] for request in requests)
    assert all(body.pings == 30 and body.closed and not body.reading for body in bodies)
    assistant = next(message for message in requests[1]["messages"] if message["role"] == "assistant")
    assert assistant["content"][0] == {"type": "thinking", "thinking": "", "signature": "signed"}
    assert assistant["content"][1]["id"] == "call_1"
    assert requests[1]["messages"][-1]["content"][0]["tool_use_id"] == "call_1"


@pytest.mark.parametrize("visible_reasoning,first_chunk_s", [
    (True, 0.1), (False, 0.0), (False, 0.05),
])
async def test_signature_only_byte_progress_preserves_stall_and_first_chunk_bounds(
    monkeypatch, visible_reasoning, first_chunk_s,
):
    class SignatureBody(_Body):
        async def __aiter__(self):
            self.reading = True
            try:
                yield _sse(_opening())
                yield _sse({"type": "content_block_start", "index": 0,
                            "content_block": {"type": "thinking", "thinking": "", "signature": ""}})
                if visible_reasoning:
                    yield _sse({"type": "content_block_delta", "index": 0,
                                "delta": {"type": "thinking_delta", "thinking": "thinking"}})
                for _ in range(10):
                    await asyncio.sleep(0.03)
                    yield _sse({"type": "content_block_delta", "index": 0,
                                "delta": {"type": "signature_delta", "signature": "sig"}})
                for event in _ending()[1:]:
                    yield _sse(event)
            finally:
                self.reading = False

    monkeypatch.setattr(f"{__name__}._Body", SignatureBody)
    observed = []

    async def on_delta(text, accumulated, index, thinking, **kwargs):
        observed.append((text, thinking))

    async with _client() as (client, body, _):
        if not visible_reasoning and first_chunk_s:
            with pytest.raises(LLMStreamStalled) as exc:
                await _stream_llm_response(client, [user_msg("hi")], 2, on_delta,
                                           first_chunk_s=first_chunk_s)
            assert exc.value.chunks_seen == 0
            assert observed == []
        else:
            response = await _stream_llm_response(client, [user_msg("hi")], 2, on_delta,
                                                  first_chunk_s=first_chunk_s)
            assert response.content[0]["signature"] == "sig" * 10
            assert response.content[-1] == {"type": "text", "text": "done"}
            assert observed == ([("", "thinking")] if visible_reasoning else []) + [("done", "")]
        assert body.closed and not body.reading


@pytest.mark.parametrize("wrapper", ["direct", "chain", "cooldown_primary", "cooldown_fallback", "cooldown_degraded"])
async def test_middleware_abort_keeps_gate_until_sdk_body_closes(monkeypatch, wrapper):
    import contextlib

    from agent_core.components.middleware.llm.base import LLMMiddleware, LLMMiddlewareChain
    from agent_core.components.middleware.llm.proxy import LLMProxy
    from agent_core.providers import _stream_activity as activity_module
    from agent_core.providers.fallback import CooldownFallbackLLM
    from agent_core.runtime import async_utils
    from agent_core.runtime.loop import _call as call_module

    entered, release, closed = (asyncio.Event() for _ in range(3))

    class Body(_Body):
        async def __aiter__(self):
            yield _sse(_opening())
            yield _sse({"type": "content_block_start", "index": 0,
                        "content_block": {"type": "text", "text": ""}})
            yield _sse({"type": "content_block_delta", "index": 0,
                        "delta": {"type": "text_delta", "text": "hello"}})
            await asyncio.Event().wait()

        async def aclose(self):
            entered.set()
            while not release.is_set():
                with contextlib.suppress(asyncio.CancelledError):
                    await release.wait()
            await super().aclose()
            closed.set()

    class Abort(LLMMiddleware):
        name = "abort"

        async def on_chunk(self, *args):
            return True

    class Unavailable:
        model = "unavailable"

        async def stream(self, *args, **kwargs):
            raise RuntimeError("unavailable")
            yield  # pragma: no cover

    monkeypatch.setattr(f"{__name__}._Body", Body)
    monkeypatch.setattr(activity_module, "_CLEANUP_TIMEOUT_S", 0.02)
    monkeypatch.setattr(async_utils, "_STREAM_CLOSE_TIMEOUT_S", 0.03)
    gate = asyncio.Semaphore(1)
    monkeypatch.setattr(call_module, "_llm_gate", lambda: gate)
    async with _client() as (client, body, _):
        inner = client
        if wrapper == "chain":
            inner = LLMFallbackChain([FallbackEntry(model=client)])
        elif wrapper.startswith("cooldown"):
            inner = CooldownFallbackLLM(
                Unavailable() if wrapper == "cooldown_degraded" else client,
                client, max_retries=1,
            )
            if wrapper == "cooldown_fallback":
                inner._cooldown_until = float("inf")
        chain = LLMMiddlewareChain()
        chain.add(Abort())
        try:
            response = await asyncio.wait_for(call_module.call_llm(
                LLMProxy(inner, chain), [user_msg("hi")], 1, 1, 1, on_delta=_ignore,
            ), 0.5)
            assert response.content == "hello"
            assert entered.is_set()
            assert not closed.is_set() and not body.closed
            assert gate._value == 0  # Cleanup is still live after the caller returns.
        finally:
            release.set()
            await asyncio.wait_for(closed.wait(), 0.5)
            for _ in range(10):
                await asyncio.sleep(0)
        assert gate._value == 1  # Released exactly once when all held I/O settles.
