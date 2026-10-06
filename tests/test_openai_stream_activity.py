"""Real OpenAI SDK parsing must preserve transport activity and output semantics."""
from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import pytest
from openai import AsyncOpenAI, DefaultAsyncHttpxClient

from agent_core.components.middleware.llm.base import LLMMiddleware, LLMMiddlewareChain
from agent_core.components.middleware.llm.proxy import LLMProxy
from agent_core.errors import LLMReasoningRunaway, LLMStreamStalled
from agent_core.messages import user_msg
from agent_core.providers.fallback import FallbackEntry, LLMFallbackChain
from agent_core.providers.openai_chat import OpenAIClient
from agent_core.providers.openai_responses import OpenAIResponsesClient
from agent_core.runtime.loop._streaming import _stream_llm_response

if issubclass(DefaultAsyncHttpxClient, httpx.AsyncClient):
    sdk_httpx = httpx
else:
    import httpx2 as sdk_httpx


@pytest.fixture(params=[
    ("chat", "apodex-1.1-mini"),
    ("chat", "gpt-5.1"),
    ("responses", "gpt-5.1"),
])
def profile(request):
    # Protocol fixtures, not evidence of availability at a live endpoint.
    return request.param


def _sse(data, event=None):
    prefix = f"event: {event}\n" if event else ""
    return (prefix + f"data: {json.dumps(data)}\n\n").encode()


def _chat_chunk(model, delta, finish=None, usage=None):
    return {
        "id": "chat_1", "object": "chat.completion.chunk", "created": 1, "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], "usage": usage,
    }


def _ending(protocol, model, tool=False):
    if protocol == "chat":
        if tool:
            delta = {"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                     "function": {"name": "search", "arguments": '{"q":"test"}'}}]}
        else:
            delta = {"content": "done"}
        return [
            _sse(_chat_chunk(model, delta)),
            _sse(_chat_chunk(model, {}, "tool_calls" if tool else "stop")),
            _sse({"id": "chat_1", "object": "chat.completion.chunk", "created": 1,
                  "model": model, "choices": [],
                  "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}),
            b"data: [DONE]\n\n",
        ]
    return [
        _sse({"type": "response.output_text.delta", "item_id": "item_1", "output_index": 0,
              "content_index": 0, "delta": "done", "sequence_number": 1}, "response.output_text.delta"),
        _sse({"type": "response.completed", "sequence_number": 2, "response": {
            "id": "resp_1", "object": "response", "created_at": 1, "model": model,
            "status": "completed", "output": [],
            "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        }}, "response.completed"),
    ]


class _Body(sdk_httpx.AsyncByteStream):
    def __init__(self, profile, mode="healthy", tool=False):
        self.profile, self.mode, self.tool = profile, mode, tool
        self.closed = False
        self.reading = False
        self.pings = 0
        self._iterator = None

    def __aiter__(self):
        self._iterator = self._read()
        return self._iterator

    async def _read(self):
        self.reading = True
        try:
            if self.mode == "silent":
                await asyncio.sleep(10)
            if self.mode == "reasoning":
                protocol, model = self.profile
                if protocol == "chat":
                    yield _sse(_chat_chunk(model, {"reasoning_content": "thinking"}))
                else:
                    yield _sse({"type": "response.reasoning_summary_text.delta", "item_id": "r1",
                                "output_index": 0, "summary_index": 0, "delta": "thinking",
                                "sequence_number": 0}, "response.reasoning_summary_text.delta")
            beats = {"dense": 0, "long": 100}.get(self.mode, 30)
            for _ in range(beats):
                await asyncio.sleep(0.01)
                self.pings += 1
                # Comments are valid SSE heartbeats, filtered by the SDK.
                if self.mode == "filtered":
                    protocol, _model = self.profile
                    if protocol == "chat":
                        metadata = _chat_chunk("", {})
                        metadata["choices"] = []
                        yield _sse(metadata)
                    else:
                        yield _sse({"type": "response.in_progress", "sequence_number": self.pings,
                                    "response": {"id": "resp_1", "object": "response",
                                                 "created_at": 1, "model": self.profile[1],
                                                 "status": "in_progress", "output": []}},
                                   "response.in_progress")
                else:
                    yield b": keep-alive\n\n"
            if self.mode == "error":
                yield _sse({"error": {"message": "upstream unavailable", "type": "server_error"}})
            else:
                for chunk in _ending(*self.profile, tool=self.tool):
                    yield chunk
        finally:
            self.reading = False

    async def aclose(self):
        self.closed = True
        if self._iterator is not None:
            await self._iterator.aclose()


@asynccontextmanager
async def _client(profile, modes=("healthy",), tools_first=False):
    requests, bodies = [], []

    def respond(request):
        if request.method == "GET":
            return sdk_httpx.Response(200, json={"object": "list", "data": []})
        requests.append(json.loads(request.content))
        mode = modes[min(len(requests) - 1, len(modes) - 1)]
        body = _Body(profile, mode, tool=tools_first and len(requests) == 1)
        bodies.append(body)
        return sdk_httpx.Response(200, stream=body, headers={"content-type": "text/event-stream"})

    protocol, model = profile
    cls = OpenAIClient if protocol == "chat" else OpenAIResponsesClient
    client = cls(model, api_key="k")
    await client._client.close()
    async with AsyncOpenAI(api_key="k", max_retries=0, http_client=sdk_httpx.AsyncClient(
        transport=sdk_httpx.MockTransport(respond),
    )) as sdk:
        client._client = sdk
        # Warm SDK platform detection outside the deliberately tiny watchdog.
        await sdk.models.list()
        yield client, bodies, requests


async def _ignore(*args, **kwargs):
    pass


@pytest.fixture(autouse=True)
def _short_stall(monkeypatch):
    monkeypatch.setattr("agent_core.runtime.loop._streaming._stream_stall_timeout_s", lambda: 0.1)


async def test_heartbeats_outlive_stall_window_without_output_noise(profile):
    observed = []

    async def on_delta(text, accumulated, index, thinking, **kwargs):
        observed.append((text, thinking, kwargs["tool_call_args_chunks"]))

    async with _client(profile) as (client, bodies, requests):
        response = await _stream_llm_response(client, [user_msg("hi")], 2, on_delta)
    assert bodies[0].pings == 30
    assert bodies[0].closed and not bodies[0].reading
    assert observed == [("done", "", [])]
    assert response.content == "done"
    assert response.model == profile[1]
    assert response.usage["completion_tokens"] == 20
    assert response.finish_reason == "stop"
    assert requests[0]["stream"] is True and requests[0]["model"] == profile[1]


async def test_silent_socket_still_stalls_and_closes(profile):
    async with _client(profile, ("silent",)) as (client, bodies, _):
        with pytest.raises(LLMStreamStalled):
            await _stream_llm_response(client, [user_msg("hi")], 2, _ignore)
        assert bodies[0].closed and not bodies[0].reading


async def test_heartbeats_do_not_extend_total_timeout(profile):
    async with _client(profile) as (client, bodies, _):
        with pytest.raises(TimeoutError):
            await _stream_llm_response(client, [user_msg("hi")], 0.15, _ignore)
        assert bodies[0].pings > 0
        assert bodies[0].closed and not bodies[0].reading


async def test_heartbeats_do_not_extend_semantic_reasoning_deadline(profile):
    async with _client(profile, ("reasoning",)) as (client, bodies, _):
        with pytest.raises(LLMReasoningRunaway):
            await _stream_llm_response(client, [user_msg("hi")], 2, _ignore,
                                       reasoning_only_timeout_s=0.05)
        assert bodies[0].pings > 0
        assert bodies[0].closed and not bodies[0].reading


async def test_external_cancellation_closes_response_and_reader(profile):
    async with _client(profile) as (client, bodies, _):
        task = asyncio.create_task(_stream_llm_response(client, [user_msg("hi")], 2, _ignore))
        while not bodies or not bodies[0].pings:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert bodies[0].closed and not bodies[0].reading


async def test_early_consumer_close_closes_response(profile):
    async with _client(profile) as (client, bodies, _):
        stream = client.stream([user_msg("hi")])
        assert (await anext(stream)).transport_activity
        await stream.aclose()
        assert bodies[0].closed and not bodies[0].reading


async def test_heartbeat_does_not_commit_fallback_before_sdk_error(profile):
    async with _client(profile, ("error",)) as (primary, bodies, _), _client(profile) as (fallback, _, _):
        chain = LLMFallbackChain(entries=[
            FallbackEntry(model=primary, provider="primary", triggers=("any_error",)),
            FallbackEntry(model=fallback, provider="fallback"),
        ])
        response = await _stream_llm_response(chain, [user_msg("hi")], 2, _ignore)
    assert bodies[0].closed and not bodies[0].reading
    assert response.content == "done"
    assert response.response_metadata["provider_actually_used"] == "fallback"


async def test_proxy_smoke_retries_after_heartbeat_only_error(profile):
    errors, chunks, after = [], [], []

    class Recorder(LLMMiddleware):
        name = "retry_recorder"

        async def on_llm_error(self, ctx, error, attempt):
            errors.append(attempt)
            return attempt == 0

        async def on_chunk(self, ctx, delta, accumulated):
            chunks.append(delta)
            return False

        async def after_llm(self, ctx, response):
            after.append(response.content)
            return response

    async with _client(profile, ("error", "healthy")) as (client, bodies, requests):
        chain = LLMMiddlewareChain()
        chain.add(Recorder())
        response = await _stream_llm_response(LLMProxy(client, chain), [user_msg("hi")], 2, _ignore)
    assert len(requests) == 2 and requests[0] == requests[1]
    assert errors == [0] and after == ["done"]
    assert all(not delta.transport_activity for delta in chunks)
    assert all(body.closed and not body.reading for body in bodies)
    assert response.content == "done"


@pytest.mark.parametrize("model", ["apodex-1.1-mini", "gpt-5.1"])
async def test_loop_smoke_tool_then_final_answer(model):
    from agent_core.loop_types import LoopConfig, LoopPolicy
    from agent_core.runtime.loop.agent_loop import run_agent_loop

    calls = []

    class Search:
        name = "search"

        async def ainvoke(self, args):
            calls.append(args)
            return "found"

        def to_openai_schema(self):
            return {"type": "function", "function": {
                "name": self.name, "parameters": {"type": "object"},
            }}

    async with _client(("chat", model), tools_first=True) as (client, bodies, requests):
        result = await run_agent_loop(
            system_prompt="s", user_message="hi", llm=client, tools=[Search()],
            config=LoopConfig(max_turns=3, max_llm_retries=1, llm_timeout=2,
                              stream_llm_tokens=True,
                              loop_policy=LoopPolicy(no_tool_behavior="stop")),
        )
    assert result.final_content == "done"
    assert calls == [{"q": "test"}]
    assert len(requests) == 2
    assert all(body.pings == 30 and body.closed and not body.reading for body in bodies)
    assert requests[1]["messages"][-1]["tool_call_id"] == "call_1"


@pytest.mark.parametrize("transport_name", ["httpx", "httpx2"])
async def test_activity_wrapper_matches_each_response_transport(transport_name):
    from agent_core.providers._stream_activity import stream_events_with_activity

    transport = httpx if transport_name == "httpx" else pytest.importorskip("httpx2")
    seen = []

    class Body(transport.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            yield b"ping"

        async def aclose(self):
            self.closed = True

    body = Body()

    class Stream:
        response = transport.Response(200, stream=body)

        async def __aiter__(self):
            async for chunk in self.response.aiter_bytes():
                seen.append(chunk)
                yield "event"

        async def close(self):
            await self.response.aclose()

    events = [event async for event in stream_events_with_activity(Stream())]
    assert "event" in events and None in events
    assert seen == [b"ping"] and body.closed


async def test_heartbeats_do_not_satisfy_first_chunk_bound(profile):
    async with _client(profile, ("long",)) as (client, bodies, _):
        with pytest.raises(LLMStreamStalled) as exc_info:
            await _stream_llm_response(client, [user_msg("hi")], 5, _ignore,
                                       first_chunk_s=0.3)
        assert exc_info.value.chunks_seen == 0
        assert 0 < bodies[0].pings < 100
        assert bodies[0].closed and not bodies[0].reading


async def test_heartbeats_extend_stall_after_first_real_chunk(profile):
    async with _client(profile, ("reasoning",)) as (client, bodies, _):
        response = await _stream_llm_response(client, [user_msg("hi")], 2, _ignore,
                                               first_chunk_s=0.1)
    assert bodies[0].pings == 30
    assert response.content == "done"
    assert response.reasoning_content == "thinking"


async def test_dense_stream_does_not_pair_events_with_heartbeats(profile):
    async with _client(profile, ("dense",)) as (client, bodies, _):
        deltas = [delta async for delta in client.stream([user_msg("hi")])]
    activity = sum(delta.transport_activity for delta in deltas)
    events = len(deltas) - activity
    # Headers, plus at most the event-less ``[DONE]`` sentinel.
    assert activity <= 2 and events >= 2
    assert bodies[0].closed


async def test_unknown_byte_stream_is_left_untouched():
    from agent_core.providers._stream_activity import stream_events_with_activity

    class Body:  # neither httpx nor httpx2: e.g. a MagicMock or custom stream
        pass

    body = Body()

    class Stream:
        response = type("Response", (), {"stream": body})()

        async def __aiter__(self):
            yield "event"

    assert [event async for event in stream_events_with_activity(Stream())] == ["event"]
    assert Stream.response.stream is body


def test_httpx2_wrapper_class_is_built_once():
    httpx2 = pytest.importorskip("httpx2")
    from agent_core.providers._stream_activity import _wrap_byte_stream

    class Body(httpx2.AsyncByteStream):
        pass

    first = _wrap_byte_stream(Body(), lambda: None)
    second = _wrap_byte_stream(Body(), lambda: None)
    assert isinstance(first, httpx2.AsyncByteStream)
    assert type(first) is type(second)


@pytest.mark.parametrize("first_chunk_s", [0.0, 0.1])
async def test_filtered_sdk_events_keep_stall_alive_without_satisfying_first_chunk(profile, first_chunk_s):
    observed = []

    async def on_delta(text, accumulated, index, thinking, **kwargs):
        observed.append(text)

    async with _client(profile, ("filtered",)) as (client, bodies, _):
        if first_chunk_s:
            with pytest.raises(LLMStreamStalled) as exc:
                await _stream_llm_response(client, [user_msg("hi")], 2, on_delta,
                                           first_chunk_s=first_chunk_s)
            assert exc.value.chunks_seen == 0
            assert observed == []
        else:
            response = await _stream_llm_response(client, [user_msg("hi")], 2, on_delta,
                                                  first_chunk_s=first_chunk_s)
            assert response.content == "done" and observed == ["done"]
            assert bodies[0].pings == 30
        assert bodies[0].closed and not bodies[0].reading
