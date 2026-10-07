# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportMissingModuleSource=false
"""Validate assembled native tool calls before any tool side effect."""

from __future__ import annotations

import json
from typing import Any

from jsonschema import ValidationError
from jsonschema.validators import validator_for


def _schema_for(llm: Any, name: str) -> dict[str, Any] | None:
    for tool in getattr(llm, "tools", None) or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict) and function.get("name") == name:
            parameters = function.get("parameters")
            return parameters if isinstance(parameters, dict) else None
    return None


def validate_arguments(value: Any, schema: dict[str, Any], path: str = "arguments") -> str | None:
    """Validate against the tool's complete JSON Schema before execution."""
    validator_cls = validator_for(schema)
    validator_cls.check_schema(schema)
    try:
        validator_cls(schema).validate(value)
    except ValidationError as exc:
        location = ".".join(str(part) for part in exc.absolute_path)
        return f"{path}{'.' + location if location else ''}: {exc.message}"
    return None


def invalid_native_tool_calls(response: Any, llm: Any) -> list[dict[str, Any]]:
    """Return diagnostics while preserving each call's original arguments."""
    issues: list[dict[str, Any]] = []
    for index, call in enumerate(getattr(response, "tool_calls", None) or []):
        function = call.get("function") if isinstance(call, dict) else None
        name = function.get("name") if isinstance(function, dict) else None
        raw = function.get("arguments") if isinstance(function, dict) else None
        reason: str | None = None
        if not isinstance(name, str) or not name:
            reason = "missing tool name"
        elif not isinstance(call.get("id"), str) or not call["id"]:
            reason = "missing tool call id"
        elif not isinstance(raw, str):
            reason = "arguments are not a JSON string"
        else:
            try:
                args = json.loads(raw)
            except ValueError:
                reason = "arguments are invalid JSON"
            else:
                if not isinstance(args, dict):
                    reason = "arguments must be a JSON object"
                else:
                    schema = _schema_for(llm, name)
                    if schema is not None:
                        reason = validate_arguments(args, schema)
        if reason:
            issues.append({"index": index, "id": call.get("id") if isinstance(call, dict) else None,
                           "name": name, "raw_arguments": raw, "reason": reason})
    return issues


__all__ = ["invalid_native_tool_calls", "validate_arguments"]
