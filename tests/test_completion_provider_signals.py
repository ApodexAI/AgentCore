"""Real SDK serialization/parsing over local HTTP mocks; no API calls."""
from __future__ import annotations

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
