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


def test_literal_content_args_are_not_whitespace_stripped() -> None:
    """Indent and trailing newlines are semantic for exact-match editors."""
    middleware = ToolCallRepairMiddleware()
    result = asyncio.run(
        middleware.before_tool_call(
            _context(
                "file_editor_str_replace",
                {
                    "path": "  a.py  ",
                    "old_string": "    def foo():\n        pass\n",
                    "new_string": "    def foo():\n        return 1\n",
                },
            ),
        ),
    )
    assert result.tool_args["old_string"] == "    def foo():\n        pass\n"
    assert result.tool_args["new_string"] == "    def foo():\n        return 1\n"
    # Non-content args are still normalized.
    assert result.tool_args["path"] == "a.py"

    written = asyncio.run(
        middleware.before_tool_call(
            _context("write_file", {"path": "x.py", "content": "print(1)\n"}),
        ),
    )
    assert written.tool_args["content"] == "print(1)\n"


def test_empty_host_tables_disable_defaults() -> None:
    """An empty mapping is an explicit opt-out, not a request for defaults."""
    middleware = ToolCallRepairMiddleware(
        key_aliases={},
        type_coercions={},
        tool_repair=lambda _name, args: args,
    )
    result = asyncio.run(
        middleware.before_tool_call(_context("web_search", {"q": "hi", "n": "3"})),
    )
    assert result.tool_args == {"q": "hi", "n": "3"}


def test_block_helper_sets_the_documented_contract_keys() -> None:
    from agent_core.protocols import BLOCK_REASON_KEY, BLOCKED_KEY

    middleware = GuardrailsMiddleware(duplicate_thresholds={"x": 1})
    asyncio.run(middleware.before_tool_call(_context("x", {"a": 1})))
    blocked = asyncio.run(middleware.before_tool_call(_context("x", {"a": 1})))
    assert blocked.is_blocked is True
    assert blocked.metadata[BLOCKED_KEY] is True
    assert "identical arguments" in blocked.block_reason
    assert blocked.metadata[BLOCK_REASON_KEY] == blocked.block_reason


def test_fingerprint_tolerates_non_json_arguments() -> None:
    middleware = GuardrailsMiddleware()
    result = asyncio.run(middleware.before_tool_call(_context("x", {"a": {1, 2}})))
    assert not result.is_blocked


def test_budget_exhausted_reads_total_tokens_and_survives_junk_config() -> None:
    from agent_core.runtime.loop.guardrails import (
        DEFAULT_MAX_TOKENS,
        check_budget_exhausted,
    )

    plan = {"execution_plan": {"budget": {"allocated": {"max_tokens": 100}}}}
    # The key every provider and UsageMetadata actually emits.
    assert check_budget_exhausted(plan, {"total_tokens": 150}) is not None
    assert check_budget_exhausted(plan, {"total": 150}) is not None
    assert check_budget_exhausted(plan, {"total_tokens": 50}) is None
    # Non-numeric config falls back instead of raising out of an advisory check.
    junk = {"execution_plan": {"budget": {"max_tokens": "unlimited"}}}
    assert check_budget_exhausted(junk, {"total_tokens": 10}) is None
    assert (
        check_budget_exhausted(junk, {"total_tokens": DEFAULT_MAX_TOKENS + 1})
        is not None
    )
    # Numeric strings and floats are honoured.
    assert check_budget_exhausted(
        {"execution_plan": {"budget": {"max_tokens": "100"}}},
        {"total_tokens": 150},
    ) is not None
