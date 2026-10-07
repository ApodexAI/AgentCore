from __future__ import annotations

from typing import Any

import pytest

from agent_core.llm import LLMResponse, StreamDelta
from agent_core.loop_types import LoopConfig, LoopPolicy
from agent_core.runtime.loop._call import call_llm
from agent_core.runtime.loop.agent_loop import run_agent_loop
from agent_core.runtime.loop.tool_call_validation import invalid_native_tool_calls


def _call(arguments: str) -> dict[str, Any]:
    return {"id": "tc1", "type": "function", "function": {"name": "save", "arguments": arguments}}


class SaveTool:
    name = "save"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def to_openai_schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": "save", "parameters": {
            "type": "object", "properties": {"value": {"type": "string"}},
            "required": ["value"],
        }}}

    async def ainvoke(self, args: dict[str, Any]) -> str:
        self.calls.append(args)
        return "saved"


@pytest.mark.parametrize("raw,expected", [
    ("", "invalid JSON"),
    ("{", "invalid JSON"),
    ("{}", "required property"),
    ('{"value": 1}', "not of type 'string'"),
])
def test_assembled_tool_arguments_are_checked(raw: str, expected: str) -> None:
    llm = type("Bound", (), {"tools": [SaveTool().to_openai_schema()]})()
    issues = invalid_native_tool_calls(LLMResponse(tool_calls=[_call(raw)]), llm)
    assert expected in issues[0]["reason"]
    assert issues[0]["raw_arguments"] == raw


def test_empty_object_is_valid_for_a_tool_without_required_fields() -> None:
    llm = type("Bound", (), {"tools": [{"function": {
        "name": "ping", "parameters": {"type": "object", "properties": {}}
    }}]})()
    response = LLMResponse(tool_calls=[{
        "id": "tc1", "type": "function",
        "function": {"name": "ping", "arguments": "{}"},
    }])
    assert invalid_native_tool_calls(response, llm) == []


@pytest.mark.asyncio
async def test_invalid_native_call_never_invokes_tool() -> None:
    tool = SaveTool()

    class Client:
        model = "test"

        async def chat(self, messages, **kwargs):
            if not hasattr(self, "called"):
                self.called = True
                return LLMResponse(tool_calls=[_call("{")])
            return LLMResponse(content="done")

    result = await run_agent_loop(
        system_prompt="system", user_message="start", llm=Client(), tools=[tool],
        config=LoopConfig(max_turns=2, max_llm_retries=1,
                          loop_policy=LoopPolicy(no_tool_behavior="stop")),
    )
    assert tool.calls == []
    assert any("[invalid tool call]" in str(message.get("content"))
               for message in result.messages if message.get("role") == "tool")


@pytest.mark.asyncio
async def test_stream_retry_uses_stream_transport_for_both_requests() -> None:
    class Client:
        model = "test"

        def __init__(self) -> None:
            self.stream_calls = 0

        async def chat(self, messages, **kwargs):
            raise AssertionError("recovery must not use chat")

        async def stream(self, messages, **kwargs):
            self.stream_calls += 1
            arguments = "" if self.stream_calls == 1 else '{"value":"ok"}'
            yield StreamDelta(tool_call_deltas=[{
                "index": 0, "id": "tc1", "name": "save", "arguments": arguments,
            }])
            yield StreamDelta(finish_reason="tool_calls")

    client = Client()
    response = await call_llm(client, [{"role": "user", "content": "save"}],
                              timeout=10, max_retries=1, turn=1, stream=True)
    assert client.stream_calls == 2
    assert response.tool_calls[0]["function"]["arguments"] == '{"value":"ok"}'
    assert response.response_metadata["invalid_tool_call_retry"] is True


@pytest.mark.asyncio
async def test_invalid_stream_after_retry_yields_tool_error_without_invocation() -> None:
    tool = SaveTool()

    class Client:
        model = "test"

        def __init__(self) -> None:
            self.stream_calls = 0

        async def chat(self, messages, **kwargs):
            raise AssertionError("stream transport must be kept")

        async def stream(self, messages, **kwargs):
            self.stream_calls += 1
            yield StreamDelta(tool_call_deltas=[{
                "index": 0, "id": f"tc{self.stream_calls}",
                "name": "save", "arguments": "{}",
            }])
            yield StreamDelta(finish_reason="tool_calls")

    client = Client()
    result = await run_agent_loop(
        system_prompt="system", user_message="start", llm=client, tools=[tool],
        config=LoopConfig(max_turns=1, max_llm_retries=1, stream_transport=True,
                          loop_policy=LoopPolicy(no_tool_behavior="stop")),
    )
    assert client.stream_calls == 2
    assert tool.calls == []
    assert result.metadata["invalid_tool_calls"][0]["raw_arguments"] == "{}"
