from __future__ import annotations

import asyncio
import json

from agent_core.protocols import ToolCallContext
from agent_core.runtime.loop.guardrails import GuardrailsMiddleware
from agent_core.runtime.loop.tool_call_repair import (
    ToolCallRepairMiddleware,
    repair_truncated_json,
)


def _context(tool: str, args: dict[str, object]) -> ToolCallContext:
    return ToolCallContext(
        task_id="task",
        phase_id="phase",
        role_id="role",
        tool_name=tool,
        tool_args=args,
        metadata={},
    )


def test_guardrails_thresholds_are_configurable() -> None:
    middleware = GuardrailsMiddleware(duplicate_thresholds={"custom": 1})
    first = asyncio.run(middleware.before_tool_call(_context("custom", {"x": 1})))
    second = asyncio.run(middleware.before_tool_call(_context("custom", {"x": 1})))
    assert not first.metadata.get("blocked")
    assert second.metadata["blocked"] is True


def test_guardrails_escalate_repeated_nonconsecutive_patterns() -> None:
    middleware = GuardrailsMiddleware(max_loop_hints=1)
    asyncio.run(middleware.before_tool_call(_context("custom", {"x": 1})))
    asyncio.run(middleware.before_tool_call(_context("custom", {"x": 2})))
    middleware.notify_loop_hint("task")
    result = asyncio.run(
        middleware.before_tool_call(_context("custom", {"x": 1})),
    )
    assert result.metadata["blocked"] is True
    middleware.cleanup_task("task")


def test_tool_repair_normalizes_aliases_types_and_special_cases() -> None:
    middleware = ToolCallRepairMiddleware()
    search = asyncio.run(
        middleware.before_tool_call(
            _context("web_search", {"q": ["one", "two"], "n": "3"}),
        ),
    )
    assert search.tool_args == {"query": "one two", "num_results": 3}
    bash = asyncio.run(
        middleware.before_tool_call(
            _context("bash", {"cmd": "```bash\necho ok\n```"}),
        ),
    )
    assert bash.tool_args == {"command": "echo ok"}


def test_tool_repair_accepts_host_alias_tables() -> None:
    middleware = ToolCallRepairMiddleware(
        key_aliases={"host_tool": {"old": "new"}},
        type_coercions={},
        tool_repair=lambda _name, args: args,
    )
    result = asyncio.run(
        middleware.before_tool_call(_context("host_tool", {"old": " value "})),
    )
    assert result.tool_args == {"new": "value"}


def test_repair_truncated_json() -> None:
    repaired = repair_truncated_json('{"items": [1, {"ok": true')
    assert repaired is not None
    assert json.loads(repaired) == {"items": [1, {"ok": True}]}
    assert repair_truncated_json("not json") is None
