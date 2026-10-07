from __future__ import annotations

from typing import Any

import pytest

from agent_core.llm import LLMResponse, StreamDelta
from agent_core.loop_types import LoopConfig, LoopPolicy
from agent_core.runtime.loop._bind import bind_tools
from agent_core.runtime.loop._call import call_llm
from agent_core.runtime.loop.agent_loop import run_agent_loop
from agent_core.runtime.loop.tool_call_validation import (
    ensure_tool_call_ids,
    invalid_native_tool_calls,
    validate_arguments,
)

_PING_SCHEMA = {"type": "function", "function": {
    "name": "ping", "parameters": {"type": "object", "properties": {}},
}}


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


@pytest.mark.parametrize("raw,expected,mode", [
    ("", "required property", "structural"),
    ("{", "invalid JSON", "structural"),
    ("[]", "must be a JSON object", "structural"),
    ("{}", "required property", "structural"),
    ('{"value": 1}', "not of type 'string'", "strict"),
])
def test_assembled_tool_arguments_are_checked(raw: str, expected: str, mode: str) -> None:
    llm = type("Bound", (), {"tools": [SaveTool().to_openai_schema()]})()
    issues = invalid_native_tool_calls(LLMResponse(tool_calls=[_call(raw)]), llm, mode)
    assert expected in issues[0]["reason"]
    assert issues[0]["raw_arguments"] == raw


def test_structural_mode_leaves_property_types_to_the_tool() -> None:
    llm = type("Bound", (), {"tools": [SaveTool().to_openai_schema()]})()
    response = LLMResponse(tool_calls=[_call('{"value": 1}')])
    assert invalid_native_tool_calls(response, llm) == []
    assert invalid_native_tool_calls(response, llm, "off") == []


@pytest.mark.parametrize("raw", ["", "   ", "{}"])
def test_blank_arguments_are_an_empty_object_for_zero_arg_tools(raw: str) -> None:
    llm = type("Bound", (), {"tools": [_PING_SCHEMA]})()
    response = LLMResponse(tool_calls=[{
        "id": "tc1", "type": "function", "function": {"name": "ping", "arguments": raw},
    }])
    assert invalid_native_tool_calls(response, llm, "strict") == []


def test_malformed_tool_schema_never_fails_the_call() -> None:
    schema = {"type": "object", "properties": {"x": {"type": "any"}}}
    assert validate_arguments({"x": 1}, schema) is None
    llm = type("Bound", (), {"tools": [{"function": {"name": "odd", "parameters": schema}}]})()
    response = LLMResponse(tool_calls=[{
        "id": "tc1", "type": "function", "function": {"name": "odd", "arguments": '{"x": 1}'},
    }])
    assert invalid_native_tool_calls(response, llm, "strict") == []


def test_missing_native_ids_are_repaired_in_place() -> None:
    response = LLMResponse(tool_calls=[
        {"id": "", "type": "function", "function": {"name": "save", "arguments": "{}"}},
        {"id": "keep", "type": "function", "function": {"name": "save", "arguments": "{}"}},
    ])
    assert ensure_tool_call_ids(response) == [0]
    assert response.tool_calls[0]["id"].startswith("call_")
    assert response.tool_calls[1]["id"] == "keep"


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
    response = await call_llm(bind_tools(client, [SaveTool()]), [{"role": "user", "content": "save"}],
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


class PingTool:
    name = "ping"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def to_openai_schema(self) -> dict[str, Any]:
        return _PING_SCHEMA

    async def ainvoke(self, args: dict[str, Any]) -> str:
        self.calls.append(args)
        return "pong"


class _OneToolCallChat:
    model = "test"

    def __init__(self, call: dict[str, Any]) -> None:
        self.call = call
        self.chat_calls = 0

    async def chat(self, messages, **kwargs):
        self.chat_calls += 1
        if self.chat_calls == 1:
            return LLMResponse(tool_calls=[self.call])
        return LLMResponse(content="done")


async def _run_once(llm: Any, tool: Any, **config: Any):
    return await run_agent_loop(
        system_prompt="system", user_message="start", llm=llm, tools=[tool],
        config=LoopConfig(max_turns=2, max_llm_retries=1,
                          loop_policy=LoopPolicy(no_tool_behavior="stop"), **config),
    )


@pytest.mark.asyncio
async def test_zero_arg_tool_with_blank_arguments_runs_without_retry_on_stream() -> None:
    tool = PingTool()

    class Client:
        model = "test"
        stream_calls = 0

        async def stream(self, messages, **kwargs):
            self.stream_calls += 1
            if self.stream_calls == 1:
                yield StreamDelta(tool_call_deltas=[{
                    "index": 0, "id": "tc1", "name": "ping", "arguments": "",
                }])
                yield StreamDelta(finish_reason="tool_calls")
            else:
                yield StreamDelta(content="done")
                yield StreamDelta(finish_reason="stop")

    client = Client()
    await _run_once(client, tool, stream_transport=True)
    assert tool.calls == [{}]
    assert client.stream_calls == 2  # one per turn: no validation retry


@pytest.mark.asyncio
async def test_zero_arg_tool_with_blank_arguments_runs_on_chat() -> None:
    tool = PingTool()
    await _run_once(_OneToolCallChat({
        "id": "tc1", "type": "function", "function": {"name": "ping", "arguments": ""},
    }), tool)
    assert tool.calls == [{}]


@pytest.mark.asyncio
async def test_type_mismatch_still_reaches_the_tool_unless_strict() -> None:
    call = {"id": "tc1", "type": "function", "function": {"name": "save", "arguments": '{"value": 7}'}}
    lenient = SaveTool()
    await _run_once(_OneToolCallChat(call), lenient)
    assert lenient.calls == [{"value": 7}]

    strict = SaveTool()
    result = await _run_once(_OneToolCallChat(dict(call)), strict, tool_argument_validation="strict")
    assert strict.calls == []
    assert "not of type 'string'" in result.metadata["invalid_tool_calls"][0]["reason"]


@pytest.mark.asyncio
async def test_text_mode_calls_are_not_blocked_by_schema_validation() -> None:
    # Text-mode parameters are JSON-decoded, so a numeric-looking string
    # parameter arrives as an int; strict mode must not reject it.
    tool = SaveTool()
    text_call = (
        "<tool_call>\n<function=save>\n<parameter=value>\n600519\n"
        "</parameter>\n</function>\n</tool_call>"
    )

    class Client:
        model = "test"
        chat_calls = 0

        async def chat(self, messages, **kwargs):
            self.chat_calls += 1
            return LLMResponse(content=text_call if self.chat_calls == 1 else "done")

    await _run_once(Client(), tool, tool_argument_validation="strict")
    assert tool.calls == [{"value": 600519}]


@pytest.mark.asyncio
async def test_native_call_without_id_gets_a_replayable_id() -> None:
    tool = PingTool()
    result = await _run_once(_OneToolCallChat({
        "type": "function", "function": {"name": "ping", "arguments": "{}"},
    }), tool)
    assert tool.calls == [{}]
    assistant = next(m for m in result.messages if m.get("role") == "assistant" and m.get("tool_calls"))
    reply = next(m for m in result.messages if m.get("role") == "tool")
    assert assistant["tool_calls"][0]["id"]
    assert reply["tool_call_id"] == assistant["tool_calls"][0]["id"]


@pytest.mark.asyncio
async def test_observers_see_no_private_validation_keys() -> None:
    seen: list[dict[str, Any]] = []

    class Spy:
        async def on_tool_call(self, ctx, tool_call):
            seen.append(dict(tool_call))

    tool = PingTool()
    await run_agent_loop(
        system_prompt="system", user_message="start", tools=[tool],
        llm=_OneToolCallChat({"id": "tc1", "type": "function",
                              "function": {"name": "ping", "arguments": "{}"}}),
        observers=[Spy()],
        config=LoopConfig(max_turns=2, max_llm_retries=1,
                          loop_policy=LoopPolicy(no_tool_behavior="stop")),
    )
    assert seen and not any(key.startswith("_") for call in seen for key in call)
