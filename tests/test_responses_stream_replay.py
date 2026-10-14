"""A streamed Responses turn must replay exactly like its non-streaming twin.

``responses`` sat in ``UNVERIFIED_STREAM_PROTOCOLS`` because nothing proved
that. It was not merely unproven: ``OpenAIResponsesClient.stream`` mapped only
text / reasoning-summary / usage events, so a streamed turn lost every
``function_call`` (the loop saw no tool call and stopped) and every
``encrypted_content`` reasoning item (the next request could not continue the
model's reasoning state). These tests drive the real client through the real
loop against a fake SDK, once per mode, and require the same history and the
same follow-up request.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from agent_core.loop_types import LoopConfig, LoopPolicy
from agent_core.providers import openai_responses as rc
from agent_core.runtime.loop.agent_loop import run_agent_loop
from agent_core.runtime.loop.model_profile import ModelProfile

USAGE = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

TOOL_TURN = {
    "id": "resp_1", "model": "gpt-x", "status": "completed", "usage": USAGE,
    "output": [
        {"type": "reasoning", "id": "rs_1", "encrypted_content": "ENC_A",
         "summary": [{"type": "summary_text", "text": "look up a"}]},
        {"type": "function_call", "id": "fc_1", "call_id": "call_a",
         "name": "search", "arguments": '{"q": "a"}'},
        {"type": "reasoning", "id": "rs_2", "encrypted_content": "ENC_B", "summary": []},
        {"type": "function_call", "id": "fc_2", "call_id": "call_b",
         "name": "search", "arguments": '{"q": "b"}'},
    ],
}
FINAL_TURN = {
    "id": "resp_2", "model": "gpt-x", "status": "completed", "usage": USAGE,
    "output": [
        {"type": "reasoning", "id": "rs_3", "encrypted_content": "ENC_C", "summary": []},
        {"type": "message", "role": "assistant",
         "content": [{"type": "output_text", "text": "done"}]},
    ],
}


def _events(payload: dict[str, Any], *, item_events: bool):
    """SDK-shaped semantic events for ``payload``, ending in ``completed``.

    ``item_events=False`` models a gateway that forwards only the deltas and
    the terminal event — the completed response must still be enough.
    """
    ev = SimpleNamespace
    out: list[Any] = []
    for index, item in enumerate(payload["output"]):
        if item_events:
            added = {**item, "arguments": ""} if item["type"] == "function_call" else item
            out.append(ev(type="response.output_item.added", output_index=index, item=added))
        if item["type"] == "reasoning":
            for part in item["summary"]:
                out.append(ev(type="response.reasoning_summary_text.delta",
                              output_index=index, delta=part["text"]))
        elif item["type"] == "message":
            for part in item["content"]:
                text = part["text"]
                for chunk in (text[:2], text[2:]):
                    out.append(ev(type="response.output_text.delta",
                                  output_index=index, delta=chunk))
        elif item["type"] == "function_call" and item_events:
            args = item["arguments"]
            for chunk in (args[:3], args[3:]):
                out.append(ev(type="response.function_call_arguments.delta",
                              output_index=index, item_id=item["id"], delta=chunk))
    out.append(ev(type="response.completed", response=payload))

    async def gen():
        for event in out:
            yield event
    return gen()


def _client(*, streaming: bool, item_events: bool = True):
    requests: list[dict[str, Any]] = []

    async def create(**kwargs):
        requests.append(json.loads(json.dumps(kwargs["input"])))
        assert bool(kwargs.get("stream")) is streaming
        payload = TOOL_TURN if len(requests) == 1 else FINAL_TURN
        return _events(payload, item_events=item_events) if streaming else payload

    client = rc.OpenAIResponsesClient("gpt-x", api_key="x")
    client._client = SimpleNamespace(responses=SimpleNamespace(create=create))
    return client, requests


class Search:
    name = "search"

    async def ainvoke(self, args):
        return f"found {args.get('q')}"

    def to_openai_schema(self):
        return {"type": "function", "function": {
            "name": self.name, "parameters": {"type": "object"},
        }}


async def _run(*, streaming: bool, item_events: bool = True):
    client, requests = _client(streaming=streaming, item_events=item_events)
    result = await run_agent_loop(
        system_prompt="s", user_message="hi", llm=client, tools=[Search()],
        config=LoopConfig(
            max_turns=4, max_llm_retries=1, stream_llm_tokens=streaming,
            loop_policy=LoopPolicy(no_tool_behavior="stop"),
        ),
        model_profile=ModelProfile(
            model_id="gpt-x", provider="openai", protocol="responses",
            thinking_format="content_block",
        ),
    )
    return result, requests


@pytest.mark.asyncio
@pytest.mark.parametrize("item_events", [True, False])
async def test_streamed_tool_turn_replays_like_non_streaming(item_events):
    batch, batch_requests = await _run(streaming=False)
    streamed, streamed_requests = await _run(streaming=True, item_events=item_events)

    assert batch.final_content == streamed.final_content == "done"
    assert len(streamed_requests) == len(batch_requests) == 2
    # The follow-up request is what the model actually sees: same items, same
    # order, encrypted reasoning bound to the call that followed it.
    assert streamed_requests[1] == batch_requests[1]
    replay = [(i.get("type"), i.get("encrypted_content") or i.get("call_id"))
              for i in streamed_requests[1] if i.get("type") in ("reasoning", "function_call")]
    assert replay == [("reasoning", "ENC_A"), ("function_call", "call_a"),
                      ("reasoning", "ENC_B"), ("function_call", "call_b")]
    assert json.dumps(streamed.messages, sort_keys=True) == json.dumps(batch.messages, sort_keys=True)


@pytest.mark.asyncio
async def test_stream_yields_each_call_once_with_full_arguments():
    client, _ = _client(streaming=True)
    deltas = [d async for d in client.stream([{"role": "user", "content": "hi"}])]
    from agent_core.stream_tools import StreamToolCalls

    calls = StreamToolCalls()
    for d in deltas:
        calls.feed(d.tool_call_deltas)
    assert [(c["id"], c["function"]["arguments"]) for c in calls.complete()] == [
        ("call_a", '{"q": "a"}'), ("call_b", '{"q": "b"}'),
    ]
    assert deltas[-1].reasoning_blocks == rc._parse_responses_output(TOOL_TURN).content
