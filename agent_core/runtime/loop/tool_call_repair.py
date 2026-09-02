"""Configurable, best-effort repair of common LLM tool-call mistakes."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from typing import Any

from agent_core.protocols import ExecutionMiddleware, ToolCallContext

ToolRepair = Callable[[str, dict[str, Any]], dict[str, Any]]

DEFAULT_KEY_ALIASES: dict[str, dict[str, str]] = {
    "web_search": {
        "search_query": "query", "q": "query", "text": "query",
        "results": "num_results", "count": "num_results", "n": "num_results",
    },
    "web_fetch": {"link": "url", "page": "url", "site": "url"},
    "read_file": {"filename": "path", "file": "path", "file_path": "path"},
    "read_text": {"filename": "path", "file": "path", "file_path": "path"},
    "write_file": {
        "filename": "path", "file": "path", "file_path": "path",
        "text": "content", "body": "content",
    },
    "bash": {"cmd": "command", "shell": "command", "run": "command"},
    "grep_search": {
        "regex": "pattern", "search": "pattern", "query": "pattern",
        "directory": "path", "glob": "glob_filter", "lines": "context_lines",
    },
    "glob_search": {
        "glob": "pattern", "file_pattern": "pattern", "directory": "path",
    },
    "delegate_subtask": {"tasks": "subtasks", "questions": "subtasks"},
    "file_editor_str_replace": {
        "file": "path", "file_path": "path", "before": "old_string",
        "old": "old_string", "after": "new_string", "new": "new_string",
    },
    "file_editor_view": {"file": "path", "file_path": "path"},
    "file_editor_create": {
        "file": "path", "file_path": "path", "file_text": "content",
    },
    "view_image": {"file": "path", "file_path": "path", "image": "path"},
}

DEFAULT_TYPE_COERCIONS: dict[str, dict[str, type[Any]]] = {
    "web_search": {"num_results": int},
    "grep_search": {"context_lines": int, "max_results": int},
}

# Argument names whose value is literal file/document content. Leading indent
# and trailing newlines are semantically significant for exact-match editors,
# so these are never whitespace-stripped. Keys are matched across every tool
# because host tool catalogs reuse the same names.
LITERAL_CONTENT_KEYS: frozenset[str] = frozenset({
    "content",
    "file_text",
    "new_str",
    "new_string",
    "old_str",
    "old_string",
    "patch",
    "replacement",
    "text",
})


def repair_truncated_json(raw: str) -> str | None:
    """Close simple truncated strings/brackets and return valid JSON text."""
    raw = raw.strip()
    if not raw:
        return None
    try:
        json.loads(raw)
        return raw
    except (json.JSONDecodeError, ValueError):
        pass
    text = re.sub(r",\s*([}\]])", r"\1", raw)
    text = re.sub(r",\s*$", "", text)
    try:
        json.loads(text)
        return text
    except (json.JSONDecodeError, ValueError):
        pass
    in_string = False
    for index, char in enumerate(text):
        if char == '"' and (index == 0 or text[index - 1] != "\\"):
            in_string = not in_string
    if in_string:
        text += '"'
    text = re.sub(r",\s*$", "", text)
    stack: list[str] = []
    in_string = False
    for index, char in enumerate(text):
        if char == '"' and (index == 0 or text[index - 1] != "\\"):
            in_string = not in_string
        elif not in_string and char == "{":
            stack.append("}")
        elif not in_string and char == "[":
            stack.append("]")
        elif not in_string and char in "}]" and stack and stack[-1] == char:
            stack.pop()
    text += "".join(reversed(stack))
    try:
        json.loads(text)
        return text
    except (json.JSONDecodeError, ValueError):
        return None


def _default_tool_repair(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    if tool_name == "web_search" and isinstance(args.get("query"), list):
        args["query"] = " ".join(str(item) for item in args["query"])
    if tool_name == "web_fetch":
        url = args.get("url")
        if isinstance(url, str) and url.startswith("<") and url.endswith(">"):
            args["url"] = url[1:-1]
    if tool_name == "bash" and isinstance(args.get("command"), str):
        match = re.match(
            r"^```(?:bash|sh|shell)?\s*\n(.*?)```\s*$",
            args["command"],
            re.DOTALL,
        )
        if match:
            args["command"] = match.group(1).strip()
    if tool_name == "delegate_subtask" and isinstance(args.get("subtasks"), str):
        raw = args["subtasks"]
        try:
            parsed = json.loads(raw)
            args["subtasks"] = parsed if isinstance(parsed, list) else raw
        except (json.JSONDecodeError, ValueError):
            args["subtasks"] = [{"question": raw}]
    if tool_name == "grep_search" and isinstance(args.get("pattern"), list):
        args["pattern"] = "|".join(str(item) for item in args["pattern"])
    return args


class ToolCallRepairMiddleware(ExecutionMiddleware):
    """Normalize aliases, primitive types, whitespace, and host repairs."""

    def __init__(
        self,
        *,
        key_aliases: Mapping[str, Mapping[str, str]] | None = None,
        type_coercions: Mapping[str, Mapping[str, type[Any]]] | None = None,
        tool_repair: ToolRepair = _default_tool_repair,
        literal_content_keys: frozenset[str] | None = None,
    ) -> None:
        # ``is None`` rather than falsiness: an empty mapping is a host
        # explicitly disabling the default table, not a request for it.
        self._key_aliases = (
            DEFAULT_KEY_ALIASES if key_aliases is None else key_aliases
        )
        self._type_coercions = (
            DEFAULT_TYPE_COERCIONS if type_coercions is None else type_coercions
        )
        self._literal_content_keys = (
            LITERAL_CONTENT_KEYS
            if literal_content_keys is None
            else literal_content_keys
        )
        self._tool_repair = tool_repair
        self._stats = {
            "key_renames": 0,
            "type_coercions": 0,
            "whitespace_strips": 0,
            "total_calls": 0,
        }

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    async def before_tool_call(self, ctx: ToolCallContext) -> ToolCallContext:
        self._stats["total_calls"] += 1
        args = ctx.tool_args
        for old, new in self._key_aliases.get(ctx.tool_name, {}).items():
            if old in args and new not in args:
                args[new] = args.pop(old)
                self._stats["key_renames"] += 1
        for key, expected in self._type_coercions.get(ctx.tool_name, {}).items():
            if key in args and not isinstance(args[key], expected):
                try:
                    args[key] = expected(args[key])
                    self._stats["type_coercions"] += 1
                except (TypeError, ValueError):
                    pass
        for key, value in args.items():
            if key in self._literal_content_keys:
                continue
            if isinstance(value, str) and value != value.strip():
                args[key] = value.strip()
                self._stats["whitespace_strips"] += 1
        ctx.tool_args = self._tool_repair(ctx.tool_name, args)
        return ctx


__all__ = [
    "DEFAULT_KEY_ALIASES",
    "DEFAULT_TYPE_COERCIONS",
    "LITERAL_CONTENT_KEYS",
    "ToolCallRepairMiddleware",
    "repair_truncated_json",
]
