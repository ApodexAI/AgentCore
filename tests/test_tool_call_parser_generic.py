"""Tests for the generic ToolCallParser (native FC + JSON fallback)."""

from __future__ import annotations

from unittest.mock import MagicMock

from agent_core.runtime.loop.tool_call_parser import (
    DefaultToolCallParser,
    ToolCallParser,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_response(
    *,
    content: str | list | None = None,
    tool_calls: list[dict] | None = None,
) -> MagicMock:
    """Build a fake AIMessage-style response object."""
    mock = MagicMock()
    mock.content = content if content is not None else ""
    mock.tool_calls = tool_calls if tool_calls is not None else []
    return mock


_ALL_TOOLS: set[str] = {
    "web_search",
    "web_fetch",
    "bash",
    "delegate_subtask",
    "use_mcp_tool",
}


# ── Tests ────────────────────────────────────────────────────────────────────


class TestNativeFunctionCalling:
    def test_native_fc_parsed(self):
        """Native tool_calls are returned correctly."""
        parser = DefaultToolCallParser()
        response = _make_response(
            tool_calls=[
                {"name": "web_search", "args": {"query": "AI chips 2025"}, "id": "tc1"}
            ]
        )
        result = parser.parse(response, _ALL_TOOLS)
        assert len(result) == 1
        assert result[0]["name"] == "web_search"
        assert result[0]["args"] == {"query": "AI chips 2025"}

    def test_native_fc_keeps_unknown_tools(self):
        """Native tool calls are passed through even when the name is unknown,
        so the executor can return an explicit "unknown tool" error the model
        can recover from (silently dropping them stranded the turn as empty)."""
        parser = DefaultToolCallParser(keep_unknown_native_companions=True)
        response = _make_response(
            tool_calls=[
                {"name": "web_search", "args": {"query": "test"}, "id": "tc1"},
                {"name": "unknown_tool", "args": {}, "id": "tc2"},
            ]
        )
        result = parser.parse(response, _ALL_TOOLS)
        assert [tc["name"] for tc in result] == ["web_search", "unknown_tool"]

    def test_native_fc_takes_priority(self):
        """When both native tool_calls and JSON text are present, native wins."""
        parser = DefaultToolCallParser()
        response = _make_response(
            content='<tool_call>{"tool": "bash", "args": {"command": "ls"}}</tool_call>',
            tool_calls=[
                {"name": "web_search", "args": {"query": "from native"}, "id": "tc1"}
            ],
        )
        result = parser.parse(response, _ALL_TOOLS)
        assert len(result) == 1
        assert result[0]["name"] == "web_search"
        # bash from JSON text is NOT returned
        assert all(tc["name"] != "bash" for tc in result)


class TestJsonFallback:
    def test_json_fallback_single(self):
        """A single <tool_call> block is parsed correctly."""
        parser = DefaultToolCallParser()
        response = _make_response(
            content='<tool_call>{"tool": "web_search", "args": {"query": "NVIDIA H100"}}</tool_call>',
        )
        result = parser.parse(response, _ALL_TOOLS)
        assert len(result) == 1
        assert result[0]["name"] == "web_search"
        assert result[0]["args"]["query"] == "NVIDIA H100"

    def test_json_fallback_multiple(self):
        """Two <tool_call> blocks in the same response are both parsed."""
        parser = DefaultToolCallParser()
        content = (
            "<tool_call>"
            '{"tool": "web_search", "args": {"query": "NVIDIA H100"}}'
            "</tool_call>"
            "\nSome thinking text in between.\n"
            "<tool_call>"
            '{"tool": "web_fetch", "args": {"url": "https://example.com"}}'
            "</tool_call>"
        )
        response = _make_response(content=content)
        result = parser.parse(response, _ALL_TOOLS)
        assert len(result) == 2
        names = {tc["name"] for tc in result}
        assert names == {"web_search", "web_fetch"}

    def test_json_fallback_filters_unknown(self):
        """Unknown tool names in JSON blocks are skipped."""
        parser = DefaultToolCallParser()
        content = (
            '<tool_call>{"tool": "web_search", "args": {"query": "test"}}</tool_call>'
            '<tool_call>{"tool": "ghost_tool", "args": {}}</tool_call>'
        )
        response = _make_response(content=content)
        result = parser.parse(response, _ALL_TOOLS)
        assert len(result) == 1
        assert result[0]["name"] == "web_search"

    def test_malformed_json_skipped(self):
        """Broken JSON inside <tool_call> → empty list, no exception raised."""
        parser = DefaultToolCallParser()
        response = _make_response(
            content="<tool_call>{bad json here!!!!}</tool_call>",
        )
        result = parser.parse(response, _ALL_TOOLS)
        assert result == []

    def test_list_content_handled(self):
        """Anthropic-style list content (content=[{type:text, text:...}]) is handled."""
        parser = DefaultToolCallParser()
        # Simulate Anthropic content format: list of blocks
        content_blocks = [
            {"type": "text", "text": "Let me search for that."},
            {
                "type": "text",
                "text": '<tool_call>{"tool": "web_search", "args": {"query": "AI"}}</tool_call>',
            },
        ]
        response = _make_response(content=content_blocks)
        result = parser.parse(response, _ALL_TOOLS)
        assert len(result) == 1
        assert result[0]["name"] == "web_search"


class TestMCPToolFallback:
    def test_use_mcp_tool_xml_fallback(self):
        parser = DefaultToolCallParser()
        response = _make_response(
            content=(
                "<use_mcp_tool>"
                "<server_name>deepwiki</server_name>"
                "<tool_name>read_repo</tool_name>"
                '<arguments>{"repo":"modelcontextprotocol/servers"}</arguments>'
                "</use_mcp_tool>"
            ),
        )

        result = parser.parse(response, _ALL_TOOLS)

        assert result == [{
            "name": "use_mcp_tool",
            "args": {
                "server_name": "deepwiki",
                "tool_name": "read_repo",
                "arguments": {"repo": "modelcontextprotocol/servers"},
            },
            "id": "mcp_tc_0",
        }]

    def test_use_mcp_tool_requires_router_tool_to_be_allowed(self):
        parser = DefaultToolCallParser()
        response = _make_response(
            content=(
                "<use_mcp_tool>"
                "<server_name>deepwiki</server_name>"
                "<tool_name>read_repo</tool_name>"
                "<arguments>{}</arguments>"
                "</use_mcp_tool>"
            ),
        )

        result = parser.parse(response, {"web_search"})

        assert result == []


class TestEdgeCases:
    def test_empty_response(self):
        """No content and no tool_calls → empty list."""
        parser = DefaultToolCallParser()
        response = _make_response(content="", tool_calls=[])
        result = parser.parse(response, _ALL_TOOLS)
        assert result == []

    def test_empty_tool_names_set_keeps_native_calls(self):
        """Native calls are passed through even with an empty tool_names set —
        the executor surfaces the "unknown tool" error (text/JSON parsing still
        filters, but native intent is never silently dropped)."""
        parser = DefaultToolCallParser()
        response = _make_response(
            tool_calls=[{"name": "web_search", "args": {}, "id": "tc1"}]
        )
        result = parser.parse(response, set())
        assert [tc["name"] for tc in result] == ["web_search"]

    def test_content_with_no_tool_call_tags(self):
        """Plain text without <tool_call> tags → empty list."""
        parser = DefaultToolCallParser()
        response = _make_response(content="This is just a text answer, no tool call.")
        result = parser.parse(response, _ALL_TOOLS)
        assert result == []


class TestProtocolCompliance:
    def test_protocol_compliance(self):
        """DefaultToolCallParser satisfies the ToolCallParser Protocol."""
        parser = DefaultToolCallParser()
        assert isinstance(parser, ToolCallParser)

    def test_protocol_has_parse_method(self):
        """ToolCallParser Protocol requires a parse method."""
        assert hasattr(ToolCallParser, "parse")
