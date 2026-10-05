"""Current Claude request, SDK parsing, streaming, and signed-history contracts.

Model IDs and response shapes follow the Fable 5.1 / Opus 5.5 migration guides.
MockTransport exercises real SDK serialization and SSE parsing without live API
credentials; it does not claim to evaluate the models themselves.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

import httpx
import pytest
from anthropic import AsyncAnthropic, BadRequestError, DefaultAsyncHttpxClient

from agent_core.components.observers.trajectory import TrajectoryFileObserver
from agent_core.loop_types import LoopConfig, LoopPolicy
from agent_core.messages import assistant_msg, tool_msg, user_msg
from agent_core.providers.anthropic import AnthropicClient, _to_anthropic_msg
from agent_core.providers.protocol_client import build_protocol_client
from agent_core.runtime.loop._streaming import _stream_llm_response
from agent_core.runtime.loop.agent_loop import run_agent_loop
from agent_core.runtime.loop.model_profile import (
    DefaultThinkingParser,
    HistoryPolicy,
    ModelProfile,
    NativeMessageNormalizer,
)

MODELS = ["claude-fable-5-1", "claude-opus-5-5"]
if issubclass(DefaultAsyncHttpxClient, httpx.AsyncClient):
    sdk_httpx = httpx
else:
    import httpx2 as sdk_httpx


def _payload(model, *, content=None, stop_reason="end_turn", **extra):
    return {
        "id": "msg_latest", "type": "message", "role": "assistant", "model": model,
        "content": content or [], "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 20}, **extra,
    }


def _sse(payload):
    events = [{
        "type": "message_start",
        "message": {**payload, "content": [], "stop_reason": None, "stop_details": None},
    }]
    for index, block in enumerate(payload["content"]):
        opening = dict(block)
        deltas = []
        if block["type"] == "thinking":
            opening.update(thinking="", signature="")
            deltas = [
                {"type": "thinking_delta", "thinking": block["thinking"]},
                {"type": "signature_delta", "signature": block["signature"]},
            ]
        elif block["type"] == "text":
            opening["text"] = ""
            deltas = [{"type": "text_delta", "text": block["text"]}]
        elif block["type"] == "tool_use":
            opening["input"] = {}
            args = json.dumps(block["input"])
            deltas = [
                {"type": "input_json_delta", "partial_json": args[:3]},
                {"type": "input_json_delta", "partial_json": args[3:]},
            ]
        events.append({"type": "content_block_start", "index": index, "content_block": opening})
        events.extend({"type": "content_block_delta", "index": index, "delta": d} for d in deltas)
        events.append({"type": "content_block_stop", "index": index})
    events.extend([
        {"type": "message_delta", "delta": {
            "stop_reason": payload["stop_reason"], "stop_sequence": None,
            "stop_details": payload.get("stop_details"),
        }, "usage": payload["usage"]},
        {"type": "message_stop"},
    ])
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


@asynccontextmanager
async def _client(model, payload, *, error=None, **kwargs):
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        if error and (len(requests) == 1 or error.get("always")):
            return sdk_httpx.Response(400, json={
                "type": "error", "error": {
                    "type": "invalid_request_error", "message": error["message"],
                },
            })
        response_payload = payload(body) if callable(payload) else payload
        if body.get("stream"):
            return sdk_httpx.Response(200, text=_sse(response_payload), headers={"content-type": "text/event-stream"})
        return sdk_httpx.Response(200, json=response_payload)

    adapter = AnthropicClient(model, api_key="test", **kwargs)
    await adapter._client.close()
    async with AsyncAnthropic(
        api_key="test", base_url="https://anthropic.invalid", max_retries=0,
        http_client=DefaultAsyncHttpxClient(transport=sdk_httpx.MockTransport(respond)),
    ) as sdk:
        adapter._client = sdk
        yield adapter, requests


async def _call(client, streaming, messages):
    if not streaming:
        return await client.chat(messages, temperature=0.2)

    async def on_delta(*_):
        pass

    return await _stream_llm_response(client, messages, 10, on_delta)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens", "end_turn"])
async def test_latest_stop_metadata_and_effort_through_real_sdk(model, streaming, stop_reason):
    extra = {}
    if stop_reason == "refusal":
        extra["stop_details"] = {
            "type": "refusal", "category": "future_category", "explanation": None,
            "future_field": "kept",
        }
    payload = _payload(model, stop_reason=stop_reason, **extra)
    async with _client(model, payload, effort="xhigh") as (client, requests):
        response = await _call(client, streaming, [user_msg("hi")])
    assert response.finish_reason == ("length" if stop_reason == "max_tokens" else stop_reason)
    assert response.response_metadata["stop_reason"] == stop_reason
    if stop_reason == "refusal":
        assert response.response_metadata["stop_details"] == {
            "type": "refusal", "category": "future_category", "future_field": "kept",
        }
    else:
        assert "stop_details" not in response.response_metadata
    assert requests[0]["model"] == model
    assert requests[0]["output_config"] == {"effort": "xhigh"}
    assert "thinking" not in requests[0]
    assert {"temperature", "top_p", "top_k"}.isdisjoint(requests[0])


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("streaming", [False, True])
async def test_empty_signed_thinking_and_interleaved_tool_calls_roundtrip(model, streaming):
    blocks = [
        {"type": "thinking", "thinking": "", "signature": "sig_first"},
        {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "first"}},
        {"type": "thinking", "thinking": "progress update", "signature": "sig_second"},
        {"type": "tool_use", "id": "t2", "name": "search", "input": {"q": "second"}},
        {"type": "redacted_thinking", "data": "opaque"},
    ]
    payload = _payload(model, content=blocks, stop_reason="tool_use")
    async with _client(model, payload, thinking={"type": "adaptive", "display": "summarized"}) as (client, requests):
        response = await _call(client, streaming, [user_msg("hi")])
    parsed = DefaultThinkingParser().extract(response, ModelProfile(
        model_id=model, provider="anthropic", thinking_format="content_block",
    ))
    history = NativeMessageNormalizer().to_history(response, parsed, HistoryPolicy(), "content_block")
    replay = _to_anthropic_msg(history)
    assert response.content == blocks
    assert replay["content"] == blocks
    assert parsed.thinking == "\nprogress update"
    assert len(response.tool_calls) == 2
    assert requests[0]["thinking"] == {"type": "adaptive", "display": "summarized"}
    # A filtered canonical call must not be reintroduced from the native blocks.
    filtered = {**history, "tool_calls": history["tool_calls"][:1]}
    assert [b["id"] for b in _to_anthropic_msg(filtered)["content"] if b["type"] == "tool_use"] == ["t1"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("streaming", [False, True])
async def test_prefix_bound_signature_retry_is_once_and_does_not_edit_history(model, streaming, caplog):
    messages = [
        user_msg("hi"),
        assistant_msg([
            {"type": "thinking", "thinking": "", "signature": "old_sig"},
            {"type": "redacted_thinking", "data": "old_opaque"},
            {"type": "text", "text": "checking"},
        ], tool_calls=[{"id": "t1", "type": "function", "function": {
            "name": "search", "arguments": '{"q":"test"}',
        }}]),
        tool_msg("found", "t1"),
    ]
    snapshot = json.dumps(messages)
    error = {"message": "Invalid `signature` in `thinking` block. The block is bound to a different conversation."}
    async with _client(model, _payload(model), error=error) as (client, requests):
        response = await _call(client, streaming, messages)
    assert len(requests) == 2
    assert [b["type"] for b in requests[1]["messages"][1]["content"]] == ["text", "tool_use"]
    assert requests[1]["messages"][2] == requests[0]["messages"][2]
    assert json.dumps(messages) == snapshot
    assert response.response_metadata["thinking_history_reset"] is True
    assert "retrying once" in caplog.text


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("message,expected_calls", [
    ("max_tokens is too large", 1),
    ("thinking.budget_tokens must be less than max_tokens", 1),
    ("Invalid signature in thinking block", 2),
    ("Invalid `signature` in `thinking` block. The block is bound to a different conversation.", 2),
])
async def test_unrelated_or_repeated_bad_requests_propagate(streaming, message, expected_calls):
    messages = [user_msg("hi"), assistant_msg([
        {"type": "thinking", "thinking": "", "signature": "old"},
        {"type": "text", "text": "answer"},
    ]), user_msg("continue")]
    async with _client(MODELS[0], _payload(MODELS[0]), error={"message": message, "always": True}) as (client, requests):
        with pytest.raises(BadRequestError):
            await _call(client, streaming, messages)
    assert len(requests) == expected_calls


@pytest.mark.parametrize("model", MODELS)
async def test_native_profile_defaults_support_current_models(model):
    client = build_protocol_client({"protocol": "anthropic", "model": model, "effort": "medium"}, title="test")
    try:
        kwargs = client._build_kwargs([user_msg("hi")], tools=None, temperature=0.2, max_tokens=None, extra_headers=None, timeout=None)
        assert kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert kwargs["extra_body"]["output_config"]["effort"] == "medium"
        assert {"temperature", "top_p", "top_k"}.isdisjoint(kwargs)
    finally:
        await client._client.close()


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("streaming", [False, True])
async def test_loop_commits_signature_reset_and_preserves_new_reasoning(model, streaming):
    initial = [user_msg("hi"), assistant_msg([
        {"type": "thinking", "thinking": "", "signature": "invalid_old"},
        {"type": "text", "text": "previous answer"},
    ]), user_msg("continue")]
    original = json.dumps(initial)

    def response_payload(request):
        signatures = [
            block.get("signature")
            for message in request["messages"] if isinstance(message["content"], list)
            for block in message["content"] if block["type"] == "thinking"
        ]
        assert "invalid_old" not in signatures
        if "valid_new" in signatures:
            return _payload(model, content=[{"type": "text", "text": "done"}])
        return _payload(model, stop_reason="tool_use", content=[
            {"type": "thinking", "thinking": "", "signature": "valid_new"},
            {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "test"}},
        ])

    class Search:
        name = "search"

        async def ainvoke(self, args):
            return "found"

        def to_openai_schema(self):
            return {"type": "function", "function": {
                "name": self.name, "parameters": {"type": "object"},
            }}

    class DeltaObserver:
        wants_llm_delta = streaming

    error = {"message": "Invalid `signature` in `thinking` block. The block is bound to a different conversation."}
    async with _client(model, response_payload, error=error) as (client, requests):
        result = await run_agent_loop(
            system_prompt="s", user_message="hi", initial_messages=initial,
            llm=client, tools=[Search()],
            observers=[DeltaObserver()],
            config=LoopConfig(max_turns=3, max_llm_retries=1, loop_policy=LoopPolicy(no_tool_behavior="stop")),
            model_profile=ModelProfile(model_id=model, provider="anthropic", thinking_format="content_block"),
        )
    assert result.final_content == "done"
    assert len(requests) == 3  # one failed request, recovery, then a clean next turn
    assert json.dumps(initial) == original
    assert "invalid_old" not in json.dumps(result.messages)
    assert "valid_new" in json.dumps(result.messages)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("streaming", [False, True])
async def test_partial_refusal_reaches_both_trajectory_formats(model, streaming, tmp_path):
    details = {"type": "refusal", "category": "bio", "explanation": "declined"}
    payload = _payload(model, stop_reason="refusal", stop_details=details,
                       content=[{"type": "text", "text": "Partial output"}])

    class DeltaObserver:
        wants_llm_delta = streaming

    trajectory = TrajectoryFileObserver(tmp_path, filename="refusal", formats=["json", "jsonl"])
    async with _client(model, payload) as (client, _):
        await run_agent_loop(
            system_prompt="s", user_message="hi", llm=client, tools=[],
            observers=[trajectory, DeltaObserver()],
            config=LoopConfig(max_turns=3, max_llm_retries=1, loop_policy=LoopPolicy(no_tool_behavior="stop")),
            model_profile=ModelProfile(model_id=model, provider="anthropic", thinking_format="content_block"),
        )
    events = [json.loads(line) for line in (tmp_path / "refusal.jsonl").read_text().splitlines()]
    record = next(event for event in events if event["t"] == "llm")
    assistant = next(msg for msg in json.loads((tmp_path / "refusal.json").read_text())["messages"] if msg["role"] == "assistant")
    for item in [record, assistant]:
        assert item["content"] == "Partial output"
        assert item["finish_reason"] == "refusal"
        assert item["stop_details"] == details
