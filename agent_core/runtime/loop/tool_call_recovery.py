"""Recovery checks for incomplete streamed tool calls."""

from __future__ import annotations

import json
from typing import Any, cast

from agent_core.llm import LLMResponse


def _required_tool_arguments(llm: Any) -> dict[str, set[str]]:
    """Map bound tool name to its non-empty set of required arguments.

    Tools with no required property are omitted entirely: for them ``{}`` is
    a legitimate call and must never trigger a second model request.
    """
    required_by_name: dict[str, set[str]] = {}
    raw_tools: object = getattr(llm, "tools", None)
    if not isinstance(raw_tools, list):
        return required_by_name

    for raw_schema in cast(list[object], raw_tools):
        if not isinstance(raw_schema, dict):
            continue
        schema = cast(dict[object, object], raw_schema)
        raw_function = schema.get("function")
        if not isinstance(raw_function, dict):
            continue
        function = cast(dict[object, object], raw_function)
        name = function.get("name")
        raw_parameters = function.get("parameters")
        if not isinstance(name, str) or not isinstance(raw_parameters, dict):
            continue
        parameters = cast(dict[object, object], raw_parameters)
        raw_required = parameters.get("required")
        if isinstance(raw_required, list) and raw_required:
            fields = {
                field for field in cast(list[object], raw_required) if isinstance(field, str)
            }
            if fields:
                required_by_name[name] = fields
    return required_by_name


def _lost_required_tool_arguments(
    raw_arguments: Any,
    required_fields: set[str],
) -> bool:
    """Return whether a streamed arguments payload lost required fields.

    Empty payloads and JSON objects missing a required field both count as
    lost. Invalid JSON also counts as lost because the native tool-call
    normalizer cannot preserve or repair the raw fragment: it degrades a
    ``json.loads`` failure to ``args={}`` before tool validation runs.
    """
    if raw_arguments is None:
        return True
    if not isinstance(raw_arguments, str):
        return False
    if not raw_arguments.strip():
        return True
    try:
        parsed: object = json.loads(raw_arguments)
    except (ValueError, TypeError):
        return True
    if not isinstance(parsed, dict):
        return False
    parsed_arguments = cast(dict[object, object], parsed)
    return bool(required_fields - parsed_arguments.keys())


def stream_tool_calls_missing_required_arguments(
    response: LLMResponse,
    llm: Any,
) -> list[str]:
    """Return required-argument tool names with incomplete streamed args.

    Some OpenAI-compatible serving parsers emit a native tool-call shell with
    a valid function name but empty arguments when generation stops before the
    closing parameter marker. The non-streaming parser can often recover the
    same truncated payload, so callers use this result to request one replay.
    """
    required_by_name = _required_tool_arguments(llm)
    if not required_by_name:
        return []

    missing: list[str] = []
    for raw_tool_call in cast(list[object], response.tool_calls):
        if not isinstance(raw_tool_call, dict):
            continue
        tool_call = cast(dict[object, object], raw_tool_call)
        raw_function = tool_call.get("function")
        if not isinstance(raw_function, dict):
            continue
        function = cast(dict[object, object], raw_function)
        name = function.get("name")
        if not isinstance(name, str):
            continue
        required_fields = required_by_name.get(name)
        if required_fields and _lost_required_tool_arguments(
            function.get("arguments"), required_fields
        ):
            missing.append(name)
    return missing


__all__ = ["stream_tool_calls_missing_required_arguments"]
