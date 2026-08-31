from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from agent_core.llm import LLMResponse
from agent_core.runtime.loop.tool_call_recovery import (
    _lost_required_tool_arguments,
    _required_tool_arguments,
    stream_tool_calls_missing_required_arguments,
)


@dataclass
class BoundLLM:
    tools: list[Any] | None


def tool_schema(name: str, *, required: list[Any] | None = None) -> dict[str, Any]:
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    if required is not None:
        parameters["required"] = required
    return {
        "type": "function",
        "function": {"name": name, "parameters": parameters},
    }


def tool_call(name: str, arguments: Any) -> dict[str, Any]:
    return {
        "id": f"call-{name}",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def test_required_tool_arguments_ignores_optional_and_malformed_schemas() -> None:
    llm = BoundLLM(
        tools=[
            tool_schema("search", required=["query", "limit", 3]),
            tool_schema("clock", required=[]),
            tool_schema("ping"),
            None,
            {"type": "function"},
            {"function": {"name": 42, "parameters": {"required": ["value"]}}},
        ]
    )

    assert _required_tool_arguments(llm) == {"search": {"query", "limit"}}


@pytest.mark.parametrize("arguments", [None, "", "   ", '{\"query\":'])
def test_lost_required_tool_arguments_detects_empty_or_invalid_json(arguments: Any) -> None:
    assert _lost_required_tool_arguments(arguments, {"query"})


def test_lost_required_tool_arguments_detects_each_missing_field() -> None:
    assert _lost_required_tool_arguments('{"query": "weather"}', {"query", "limit"})
    assert not _lost_required_tool_arguments(
        '{"query": "weather", "limit": 5}', {"query", "limit"}
    )


def test_non_string_arguments_are_left_to_downstream_validation() -> None:
    assert not _lost_required_tool_arguments({}, {"query"})


def test_stream_check_reports_only_known_tools_missing_required_arguments() -> None:
    llm = BoundLLM(
        tools=[
            tool_schema("search", required=["query"]),
            tool_schema("fetch", required=["url"]),
            tool_schema("clock", required=[]),
        ]
    )
    response = LLMResponse(
        tool_calls=[
            tool_call("search", "{}"),
            tool_call("fetch", '{"url": "https://example.com"}'),
            tool_call("clock", "{}"),
            tool_call("unknown", ""),
        ]  # type: ignore[arg-type]
    )

    assert stream_tool_calls_missing_required_arguments(response, llm) == ["search"]


def test_stream_check_preserves_duplicate_failed_calls_for_diagnostics() -> None:
    llm = BoundLLM(tools=[tool_schema("search", required=["query"])])
    response = LLMResponse(
        tool_calls=[tool_call("search", ""), tool_call("search", "{}")]  # type: ignore[arg-type]
    )

    assert stream_tool_calls_missing_required_arguments(response, llm) == ["search", "search"]


def test_stream_check_is_silent_without_bound_required_tools() -> None:
    response = LLMResponse(tool_calls=[tool_call("search", "")])  # type: ignore[arg-type]

    assert stream_tool_calls_missing_required_arguments(response, BoundLLM(tools=None)) == []
