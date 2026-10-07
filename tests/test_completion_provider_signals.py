"""Real SDK serialization/parsing over local HTTP mocks; no API calls."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from openai import AsyncOpenAI, DefaultAsyncHttpxClient

from agent_core.loop_types import LoopConfig, LoopPolicy
from agent_core.providers.openai_chat import OpenAIClient
from agent_core.providers.openai_responses import OpenAIResponsesClient
from agent_core.runtime.loop.agent_loop import run_agent_loop
from agent_core.runtime.loop.model_profile import ModelProfile

if issubclass(DefaultAsyncHttpxClient, httpx.AsyncClient):
    sdk_httpx = httpx
else:
    import httpx2 as sdk_httpx


def sse(events):
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"


async def run(client, streaming, protocol):
    return await run_agent_loop(system_prompt="s", user_message="u", llm=client, tools=[],
        config=LoopConfig(max_turns=2, max_llm_retries=1, stream_llm_tokens=streaming,
            retry_wait_fixed=0, loop_policy=LoopPolicy(no_tool_behavior="nudge")),
        model_profile=ModelProfile(model_id="m", provider="openai", protocol=protocol,
            thinking_format="content_block" if protocol == "responses" else "none"))


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("refusal,finish,expected", [
    ("request declined", "stop", "refusal"),
    ("", "stop", "refusal"),
    (None, "content_filter", "content_filter"),
])
async def test_chat_refusal_and_filter_without_usage_survive_sdk_boundary(streaming, refusal, finish, expected):
    requests = []
    def respond(request):
        requests.append(json.loads(request.content))
        if streaming:
            events = [{
                "id": "id", "object": "chat.completion.chunk", "created": 1, "model": "m",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": None, "refusal": refusal}, "finish_reason": finish}],
            }]
            return sdk_httpx.Response(200, text=sse(events), headers={"content-type": "text/event-stream"})
        return sdk_httpx.Response(200, json={
            "id": "id", "object": "chat.completion", "created": 1, "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": None, "refusal": refusal}, "finish_reason": finish}],
        })
    client = OpenAIClient("m", api_key="test", base_url="https://openai.invalid")
    await client._client.close()
    async with AsyncOpenAI(api_key="test", base_url="https://openai.invalid", max_retries=0,
        http_client=DefaultAsyncHttpxClient(transport=sdk_httpx.MockTransport(respond))) as sdk:
        client._client = sdk
        result = await run(client, streaming, "chat_completions")
    assert len(requests) == 1
    assert result.stopped_by == expected
    assert result.final_content == (refusal or "")
    assert result.turns_used == 1


@pytest.mark.parametrize("streaming", [False, True])
async def test_chat_empty_refusal_beside_content_is_not_a_refusal(streaming):
    def respond(request):
        if streaming:
            events = [{
                "id": "id", "object": "chat.completion.chunk", "created": 1, "model": "m",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": None, "refusal": ""}, "finish_reason": None}],
            }, {
                "id": "id", "object": "chat.completion.chunk", "created": 1, "model": "m",
                "choices": [{"index": 0, "delta": {"content": "hello", "refusal": ""}, "finish_reason": "stop"}],
            }]
            return sdk_httpx.Response(200, text=sse(events), headers={"content-type": "text/event-stream"})
        return sdk_httpx.Response(200, json={
            "id": "id", "object": "chat.completion", "created": 1, "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello", "refusal": ""}, "finish_reason": "stop"}],
        })
    client = OpenAIClient("m", api_key="test", base_url="https://openai.invalid")
    await client._client.close()
    async with AsyncOpenAI(api_key="test", base_url="https://openai.invalid", max_retries=0,
        http_client=DefaultAsyncHttpxClient(transport=sdk_httpx.MockTransport(respond))) as sdk:
        client._client = sdk
        result = await run(client, streaming, "chat_completions")
    assert result.stopped_by != "refusal"
    assert result.final_content == "hello"


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("delta_events", [False, True])
@pytest.mark.parametrize("refusal", ["request declined", ""])
async def test_responses_refusal_parts_and_events_without_usage_survive_sdk_boundary(streaming, delta_events, refusal):
    requests = []
    response = {
        "id": "resp", "object": "response", "created_at": 1, "model": "m", "status": "completed",
        "output": [{"type": "message", "id": "msg", "role": "assistant", "status": "completed",
            "content": [{"type": "refusal", "refusal": refusal}]}],
        "usage": None,
    }
    def respond(request):
        requests.append(json.loads(request.content))
        if streaming:
            events = [{"type": "response.refusal.delta", "delta": refusal, "output_index": 0,
                "content_index": 0, "item_id": "msg", "sequence_number": 1}] if delta_events else []
            events.append({"type": "response.completed", "response": response, "sequence_number": 2})
            return sdk_httpx.Response(200, text=sse(events), headers={"content-type": "text/event-stream"})
        return sdk_httpx.Response(200, json=response)
    client = OpenAIResponsesClient("m", api_key="test", base_url="https://openai.invalid")
    await client._client.close()
    async with AsyncOpenAI(api_key="test", base_url="https://openai.invalid", max_retries=0,
        http_client=DefaultAsyncHttpxClient(transport=sdk_httpx.MockTransport(respond))) as sdk:
        client._client = sdk
        result = await run(client, streaming, "responses")
    assert len(requests) == 1
    assert result.stopped_by == "refusal"
    assert result.final_content == refusal
    assert result.turns_used == 1


@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
@pytest.mark.parametrize("streaming", [False, True])
async def test_gateway_estimated_usage_marker_survives_real_sdk_normalization(protocol, streaming):
    requests = []
    def respond(request):
        requests.append(json.loads(request.content))
        blank = len(requests) == 1
        text = "" if blank else "answer"
        if protocol == "chat_completions":
            payload = {
                "id": "id", "object": "chat.completion.chunk" if streaming else "chat.completion",
                "created": 1, "model": "m", "choices": [{"index": 0, "finish_reason": "stop",
                    "delta" if streaming else "message": {"role": "assistant", "content": text}}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "estimated": True} if blank else None,
            }
            events = [payload]
        else:
            payload = {
                "id": "resp", "object": "response", "created_at": 1, "model": "m", "status": "completed",
                "output": [{"type": "message", "id": "msg", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}]}],
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "estimated": True} if blank else None,
            }
            events = [{"type": "response.output_text.delta", "delta": text, "sequence_number": 1},
                {"type": "response.completed", "response": payload, "sequence_number": 2}]
        if streaming:
            return sdk_httpx.Response(200, text=sse(events), headers={"content-type": "text/event-stream"})
        return sdk_httpx.Response(200, json=payload)
    cls = OpenAIClient if protocol == "chat_completions" else OpenAIResponsesClient
    client = cls("m", api_key="test", base_url="https://openai.invalid")
    await client._client.close()
    async with AsyncOpenAI(api_key="test", base_url="https://openai.invalid", max_retries=0,
        http_client=DefaultAsyncHttpxClient(transport=sdk_httpx.MockTransport(respond))) as sdk:
        client._client = sdk
        result = await run_agent_loop(system_prompt="s", user_message="u", llm=client, tools=[],
            config=LoopConfig(max_turns=1, max_llm_retries=1, stream_llm_tokens=streaming,
                retry_wait_fixed=0, loop_policy=LoopPolicy(no_tool_behavior="stop")),
            model_profile=ModelProfile(model_id="m", provider="openai", protocol=protocol,
                thinking_format="content_block" if protocol == "responses" else "none"))
    assert len(requests) == 2
    assert result.final_content == "answer"


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("reasoning_field", ["reasoning_content", "reasoning"])
@pytest.mark.parametrize("finish", ["stop", None])
async def test_empty_refusal_beside_reasoning_continues_to_the_answer(streaming, reasoning_field, finish):
    requests = []
    def respond(request):
        requests.append(json.loads(request.content))
        first = len(requests) == 1
        message = {"role": "assistant", "content": None, "refusal": "", reasoning_field: "real thinking"} if first else {"role": "assistant", "content": "answer"}
        payload = {"id": "id", "created": 1, "model": "m",
            "object": "chat.completion.chunk" if streaming else "chat.completion",
            "choices": [{"index": 0, "delta" if streaming else "message": message,
                "finish_reason": finish if first else "stop"}]}
        return sdk_httpx.Response(200, text=sse([payload]), headers={"content-type": "text/event-stream"}) if streaming else sdk_httpx.Response(200, json=payload)
    client = OpenAIClient("m", api_key="test", base_url="https://openai.invalid")
    await client._client.close()
    async with AsyncOpenAI(api_key="test", base_url="https://openai.invalid", max_retries=0,
        http_client=DefaultAsyncHttpxClient(transport=sdk_httpx.MockTransport(respond))) as sdk:
        client._client = sdk
        result = await run_agent_loop(system_prompt="s", user_message="u", llm=client, tools=[],
            config=LoopConfig(max_turns=2, max_llm_retries=1, stream_llm_tokens=streaming,
                retry_wait_fixed=0, loop_policy=LoopPolicy(no_tool_behavior="nudge")),
            model_profile=ModelProfile(model_id="m", provider="openai", thinking_format="reasoning_content"))
    assert len(requests) == 2
    assert result.final_content == "answer"
    assert result.stopped_by == "no_tool"


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("finish", ["stop", None])
@pytest.mark.parametrize("with_usage", [False, True])
async def test_empty_refusal_at_clean_eof_is_preserved_with_or_without_finish_reason(streaming, finish, with_usage):
    requests = []
    def respond(request):
        requests.append(json.loads(request.content))
        payload = {"id": "id", "created": 1, "model": "m",
            "object": "chat.completion.chunk" if streaming else "chat.completion",
            "choices": [{"index": 0, "delta" if streaming else "message": {
                "role": "assistant", "content": None, "refusal": ""}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10} if with_usage else None}
        return sdk_httpx.Response(200, text=sse([payload]), headers={"content-type": "text/event-stream"}) if streaming else sdk_httpx.Response(200, json=payload)
    client = OpenAIClient("m", api_key="test", base_url="https://openai.invalid")
    await client._client.close()
    async with AsyncOpenAI(api_key="test", base_url="https://openai.invalid", max_retries=0,
        http_client=DefaultAsyncHttpxClient(transport=sdk_httpx.MockTransport(respond))) as sdk:
        client._client = sdk
        result = await run(client, streaming, "chat_completions")
    assert len(requests) == 1
    assert result.stopped_by == "refusal"
    assert result.final_content == ""


@pytest.mark.parametrize("output", ["text", "reasoning", "tool"])
async def test_empty_refusal_waits_for_the_entire_stream_before_inference(output):
    from agent_core.completion import response_rejection_reason
    from agent_core.runtime.loop._streaming import _stream_llm_response

    actual = {"content": "answer"} if output == "text" else {"reasoning_content": "thinking"} if output == "reasoning" else {"tool_calls": [{"index": 0, "id": "call", "type": "function", "function": {"name": "echo", "arguments": "{}"}}]}
    def respond(request):
        chunks = [{"refusal": "", "content": None}, actual]
        events = [{"id": "id", "created": 1, "model": "m", "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": delta, "finish_reason": "stop" if index == 0 else None}]}
            for index, delta in enumerate(chunks)]
        return sdk_httpx.Response(200, text=sse(events), headers={"content-type": "text/event-stream"})
    client = OpenAIClient("m", api_key="test", base_url="https://openai.invalid")
    await client._client.close()
    async def noop(*_args, **_kwargs):
        pass
    async with AsyncOpenAI(api_key="test", base_url="https://openai.invalid", max_retries=0,
        http_client=DefaultAsyncHttpxClient(transport=sdk_httpx.MockTransport(respond))) as sdk:
        client._client = sdk
        response = await _stream_llm_response(client, [], 10, noop)
    assert response_rejection_reason(response) == ""
    if output == "text":
        assert response.content == "answer"
    elif output == "reasoning":
        assert response.reasoning_content == "thinking"
    else:
        assert response.tool_calls[0]["function"]["name"] == "echo"


@pytest.mark.parametrize("abort", ["error", "close", "cancel"])
async def test_incomplete_stream_does_not_infer_refusal_on_error_or_consumer_close(abort):
    from types import SimpleNamespace

    async def events():
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(refusal="", content=None), finish_reason=None)], usage=None, model="m")
        if abort == "cancel":
            await asyncio.Event().wait()
        raise RuntimeError("transport broke")
    client = OpenAIClient("m", api_key="test")
    async def open_stream(_kwargs):
        return events()
    client._open_stream = open_stream
    seen = []
    try:
        stream = client.stream([])
        if abort == "error":
            with pytest.raises(RuntimeError, match="transport broke"):
                async for delta in stream:
                    seen.append(delta)
        elif abort == "close":
            seen.append(await anext(stream))
            await stream.aclose()
        else:
            ready = asyncio.Event()
            async def consume():
                async for delta in stream:
                    seen.append(delta)
                    ready.set()
            task = asyncio.create_task(consume())
            await asyncio.wait_for(ready.wait(), 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
    finally:
        await client._client.close()
    assert len(seen) == 1
    assert all(not delta.stop_details for delta in seen)


@pytest.mark.parametrize("protocol", ["chat_completions", "responses"])
async def test_truncated_refusal_stream_is_never_accepted_as_a_completed_decline(protocol, monkeypatch):
    from agent_core.errors import LLMOpenAITruncatedStream

    monkeypatch.setenv("AGENT_CORE_STREAM_REQUIRE_TERMINATOR", "1")
    def respond(request):
        event = {"id": "id", "object": "chat.completion.chunk", "created": 1, "model": "m",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": None, "refusal": ""}, "finish_reason": None}]}
        if protocol == "responses":
            event = {"type": "response.refusal.delta", "delta": "declined", "output_index": 0,
                "content_index": 0, "item_id": "msg", "sequence_number": 1}
        # Neither [DONE]/finish_reason nor a Responses terminal event arrives.
        return sdk_httpx.Response(200, text=f"data: {json.dumps(event)}\n\n", headers={"content-type": "text/event-stream"})
    cls = OpenAIClient if protocol == "chat_completions" else OpenAIResponsesClient
    client = cls("m", api_key="test", base_url="https://openai.invalid")
    await client._client.close()
    seen = []
    async with AsyncOpenAI(api_key="test", base_url="https://openai.invalid", max_retries=0,
        http_client=DefaultAsyncHttpxClient(transport=sdk_httpx.MockTransport(respond))) as sdk:
        client._client = sdk
        with pytest.raises(LLMOpenAITruncatedStream):
            async for delta in client.stream([]):
                seen.append(delta)
    if protocol == "chat_completions":
        assert all(not delta.stop_details for delta in seen)
