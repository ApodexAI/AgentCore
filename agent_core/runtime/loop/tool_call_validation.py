# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportMissingModuleSource=false
"""Validate assembled native tool calls before any tool side effect.

Three levels, selected by ``LoopConfig.tool_argument_validation``:

``"structural"`` (default)
    Arguments must decode to a JSON object that carries every top-level
    ``required`` property. A blank argument string is ``{}`` -- several
    providers (Anthropic streaming among them) encode a zero-argument call
    that way. Property types are *not* checked, so tools that coerce
    ``"5"`` to ``5`` keep working exactly as before.
``"strict"``
    The structural checks plus full JSON Schema validation.
``"off"``
    No validation; only the legacy empty-required-arguments retry remains.

A schema that is itself invalid never fails a call: validation for that tool
is skipped with a warning, matching how providers accept loose schemas.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable, Mapping
from functools import lru_cache
from typing import Any, Literal

logger = logging.getLogger(__name__)

ToolArgumentValidation = Literal["structural", "strict", "off"]
TOOL_ARGUMENT_VALIDATION_MODES: frozenset[str] = frozenset({"structural", "strict", "off"})

# Reasons that describe the call's identity rather than its arguments. They are
# reported, but a model retry cannot be expected to fix them: missing ids are
# repaired locally and nameless slots are already answered as dropped calls.
IDENTITY_REASONS: frozenset[str] = frozenset({"missing tool name", "missing tool call id"})


def _normalize_mode(mode: str | None) -> str:
    if mode in TOOL_ARGUMENT_VALIDATION_MODES:
        return str(mode)
    if mode is not None:
        logger.warning("Unknown tool_argument_validation %r; using 'structural'", mode)
    return "structural"


def bound_tool_parameters(llm: Any) -> dict[str, dict[str, Any]]:
    """Map each tool bound on ``llm`` (OpenAI schema shape) to its parameters."""
    parameters_by_name: dict[str, dict[str, Any]] = {}
    for tool in getattr(llm, "tools", None) or []:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        parameters = function.get("parameters")
        if isinstance(name, str) and isinstance(parameters, dict):
            parameters_by_name[name] = parameters
    return parameters_by_name


@lru_cache(maxsize=256)
def _cached_validator(schema_json: str) -> Any | None:
    from jsonschema.exceptions import SchemaError
    from jsonschema.validators import validator_for

    schema = json.loads(schema_json)
    try:
        validator_cls = validator_for(schema)
        validator_cls.check_schema(schema)
        return validator_cls(schema)
    except SchemaError as exc:
        logger.warning(
            "Tool schema is not valid JSON Schema; skipping strict argument "
            "validation for it: %s", exc.message,
        )
        return None


def _validator_for(schema: dict[str, Any]) -> Any | None:
    try:
        schema_json = json.dumps(schema, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return None
    return _cached_validator(schema_json)


def validate_arguments(value: Any, schema: dict[str, Any], path: str = "arguments") -> str | None:
    """Validate against the tool's complete JSON Schema.

    Returns a diagnostic string, or ``None`` when the value is valid *or* the
    schema itself cannot be used for validation.
    """
    validator = _validator_for(schema)
    if validator is None:
        return None
    from jsonschema.exceptions import best_match

    try:
        error = best_match(validator.iter_errors(value))
    except Exception as exc:  # unresolvable $ref and similar schema faults
        logger.warning("Strict tool argument validation skipped: %s", exc)
        return None
    if error is None:
        return None
    location = ".".join(str(part) for part in error.absolute_path)
    return f"{path}{'.' + location if location else ''}: {error.message}"


def _missing_required(args: dict[str, Any], schema: dict[str, Any]) -> str | None:
    required = schema.get("required")
    if not isinstance(required, list):
        return None
    missing = [field for field in required if isinstance(field, str) and field not in args]
    if not missing:
        return None
    if len(missing) == 1:
        return f"arguments: '{missing[0]}' is a required property"
    return f"arguments: required properties missing: {', '.join(repr(m) for m in missing)}"


def argument_issue(raw: Any, schema: dict[str, Any] | None, mode: str = "structural") -> str | None:
    """Return why ``raw`` (a native ``function.arguments`` value) is invalid."""
    mode = _normalize_mode(mode)
    if mode == "off":
        return None
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        args: Any = {}
    elif isinstance(raw, str):
        try:
            args = json.loads(raw)
        except ValueError:
            return "arguments are invalid JSON"
    elif isinstance(raw, dict):
        args = raw
    else:
        return "arguments must be a JSON object"
    if not isinstance(args, dict):
        return "arguments must be a JSON object"
    if schema is None:
        return None
    missing = _missing_required(args, schema)
    if missing:
        return missing
    if mode == "strict":
        return validate_arguments(args, schema)
    return None


def native_tool_call_issues(
    tool_calls: Any,
    schema_for: Callable[[str], dict[str, Any] | None] | Mapping[str, dict[str, Any]],
    mode: str = "structural",
) -> list[dict[str, Any]]:
    """Diagnose native OpenAI-shape tool calls, keeping the raw arguments."""
    lookup: Callable[[str], dict[str, Any] | None] = (
        schema_for.get if isinstance(schema_for, Mapping) else schema_for
    )
    issues: list[dict[str, Any]] = []
    for index, call in enumerate(tool_calls or []):
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        raw = function.get("arguments")
        call_id = call.get("id")
        if not isinstance(name, str) or not name:
            reason: str | None = "missing tool name"
        elif not isinstance(call_id, str) or not call_id:
            reason = "missing tool call id"
        else:
            reason = argument_issue(raw, lookup(name), mode)
        if reason:
            issues.append({
                "index": index, "id": call_id, "name": name,
                "raw_arguments": raw, "reason": reason,
            })
    return issues


def invalid_native_tool_calls(
    response: Any, llm: Any, mode: str = "structural",
) -> list[dict[str, Any]]:
    """Return argument diagnostics for ``response`` against ``llm``'s tools.

    Identity problems (missing name / id) are excluded: a retry is not the
    remedy for them -- see :func:`ensure_tool_call_ids`.
    """
    return [
        issue for issue in native_tool_call_issues(
            getattr(response, "tool_calls", None), bound_tool_parameters(llm), mode,
        )
        if issue["reason"] not in IDENTITY_REASONS
    ]


def ensure_tool_call_ids(response: Any) -> list[int]:
    """Give every named native call without an id a unique one, in place.

    The id lands on ``response.tool_calls`` before the assistant turn is
    written to history, so the tool reply and the replayed assistant message
    agree. Returns the indexes that were filled.
    """
    filled: list[int] = []
    for index, call in enumerate(getattr(response, "tool_calls", None) or []):
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        name = function.get("name") if isinstance(function, dict) else call.get("name")
        if not name:
            continue
        call_id = call.get("id")
        if isinstance(call_id, str) and call_id:
            continue
        call["id"] = f"call_{uuid.uuid4().hex[:24]}"
        filled.append(index)
    if filled:
        logger.warning("Assigned ids to %d native tool call(s) the provider sent without one", len(filled))
    return filled


__all__ = [
    "IDENTITY_REASONS",
    "TOOL_ARGUMENT_VALIDATION_MODES",
    "ToolArgumentValidation",
    "argument_issue",
    "bound_tool_parameters",
    "ensure_tool_call_ids",
    "invalid_native_tool_calls",
    "native_tool_call_issues",
    "validate_arguments",
]
