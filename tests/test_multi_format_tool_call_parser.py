"""Tests for MultiFormatToolCallParser — Qwen/Seed XML fallbacks.

The parser is a superset of DefaultToolCallParser. All existing
DefaultToolCallParser behavior is preserved; these tests cover only the
new XML fallback paths and guard the priority invariants.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from agent_core.runtime.loop.tool_call_parser import (
    DefaultToolCallParser,
    MultiFormatToolCallParser,
    ToolCallParser,
)


def _make_response(
    *,
    content: str | list | None = None,
    tool_calls: list[dict] | None = None,
) -> MagicMock:
    mock = MagicMock()
    mock.content = content if content is not None else ""
    mock.tool_calls = tool_calls if tool_calls is not None else []
    return mock


_ALL_TOOLS: set[str] = {
    "web_search", "web_fetch", "bash", "delegate_subtask",
    "create_subagent", "assign_task", "run_python_code",
}


class TestMultiFormatPriority:
    def test_native_still_wins(self):
        """Native tool_calls take priority over any XML in content."""
        parser = MultiFormatToolCallParser()
        response = _make_response(
            tool_calls=[{"name": "web_search", "args": {"query": "x"}, "id": "n1"}],
            content='<tool_call><function=bash><parameter=command>ls</parameter></function></tool_call>',
        )
        result = parser.parse(response, _ALL_TOOLS)
        assert len(result) == 1
        assert result[0]["name"] == "web_search"

    def test_json_fallback_still_works(self):
        """Existing JSON fallback inherited from base parser still works."""
        parser = MultiFormatToolCallParser()
        response = _make_response(
            content='<tool_call>{"tool": "web_search", "args": {"query": "hi"}}</tool_call>',
        )
        result = parser.parse(response, _ALL_TOOLS)
        assert len(result) == 1
        assert result[0]["name"] == "web_search"
        assert result[0]["args"] == {"query": "hi"}

    def test_native_with_all_unknown_tools_does_not_fall_back(self):
        """If native had only unknown tools, don't second-guess with XML — the
        unknown native call is returned as-is (so the executor can error on it),
        and the XML web_search in content is NOT used."""
        parser = MultiFormatToolCallParser()
        response = _make_response(
            tool_calls=[{"name": "unknown_tool", "args": {}, "id": "n1"}],
            content='<function name="web_search"><parameter name="query">x</parameter></function>',
        )
        result = parser.parse(response, _ALL_TOOLS)
        assert [tc["name"] for tc in result] == ["unknown_tool"]


class TestQwenXmlFallback:
    def test_qwen_single_call(self):
        parser = MultiFormatToolCallParser()
        response = _make_response(content=(
            '<tool_call><function=web_search>'
            '<parameter=query>NVIDIA H100</parameter>'
            '<parameter=num>5</parameter>'
            '</function></tool_call>'
        ))
        result = parser.parse(response, _ALL_TOOLS)
        assert len(result) == 1
        assert result[0]["name"] == "web_search"
        assert result[0]["args"]["query"] == "NVIDIA H100"
        assert result[0]["args"]["num"] == 5        # coerced from JSON-parseable string

    def test_qwen_multiple_calls(self):
        parser = MultiFormatToolCallParser()
        content = (
            '<tool_call><function=web_search>'
            '<parameter=query>AMD</parameter>'
            '</function></tool_call>'
            'thinking text\n'
            '<tool_call><function=web_fetch>'
            '<parameter=url>https://example.com</parameter>'
            '</function></tool_call>'
        )
        response = _make_response(content=content)
        result = parser.parse(response, _ALL_TOOLS)
        assert len(result) == 2
        names = {tc["name"] for tc in result}
        assert names == {"web_search", "web_fetch"}

    def test_qwen_unknown_tool_filtered(self):
        parser = MultiFormatToolCallParser()
        response = _make_response(content=(
            '<tool_call><function=ghost_tool>'
            '<parameter=x>1</parameter>'
            '</function></tool_call>'
        ))
        result = parser.parse(response, _ALL_TOOLS)
        assert result == []

    def test_qwen_emits_langchain_shape(self):
        parser = MultiFormatToolCallParser()
        response = _make_response(content=(
            '<tool_call><function=bash>'
            '<parameter=command>echo hi</parameter>'
            '</function></tool_call>'
        ))
        result = parser.parse(response, _ALL_TOOLS)
        assert len(result) == 1
        tc = result[0]
        assert set(tc.keys()) >= {"name", "args", "id"}
        assert isinstance(tc["args"], dict)


class TestSeedXmlFallback:
    def test_seed_single_call(self):
        parser = MultiFormatToolCallParser()
        response = _make_response(content=(
            '<function name="web_search">'
            '<parameter name="query">AMD MI300X</parameter>'
            '<parameter name="num">3</parameter>'
            '</function>'
        ))
        result = parser.parse(response, _ALL_TOOLS)
        assert len(result) == 1
        assert result[0]["name"] == "web_search"
        assert result[0]["args"]["query"] == "AMD MI300X"
        assert result[0]["args"]["num"] == 3

    def test_seed_nested_json_param(self):
        """Seed parameters containing JSON lists are decoded to real lists."""
        parser = MultiFormatToolCallParser()
        response = _make_response(content=(
            '<function name="web_search">'
            '<parameter name="q">["foo", "bar"]</parameter>'
            '</function>'
        ))
        result = parser.parse(response, _ALL_TOOLS)
        assert len(result) == 1
        assert result[0]["args"]["q"] == ["foo", "bar"]

    def test_seed_unknown_tool_filtered(self):
        parser = MultiFormatToolCallParser()
        response = _make_response(content=(
            '<function name="ghost_tool">'
            '<parameter name="x">1</parameter>'
            '</function>'
        ))
        result = parser.parse(response, _ALL_TOOLS)
        assert result == []

    def test_seed_with_tag_hint(self):
        """Seed-flavored content with <seed...tool> hint also triggers fallback."""
        parser = MultiFormatToolCallParser()
        content = (
            '<seed:tool_use>'
            '<function name="bash">'
            '<parameter name="command">pwd</parameter>'
            '</function>'
            '</seed:tool_use>'
        )
        response = _make_response(content=content)
        result = parser.parse(response, _ALL_TOOLS)
        assert len(result) == 1
        assert result[0]["name"] == "bash"


class TestMalformedInput:
    def test_empty_response(self):
        parser = MultiFormatToolCallParser()
        result = parser.parse(_make_response(content="", tool_calls=[]), _ALL_TOOLS)
        assert result == []

    def test_plain_text_no_xml(self):
        parser = MultiFormatToolCallParser()
        result = parser.parse(
            _make_response(content="Just an answer, no tool call."), _ALL_TOOLS,
        )
        assert result == []

    def test_partial_qwen_tag_is_ignored(self):
        """Unclosed <tool_call> without matching </tool_call> → no parse."""
        parser = MultiFormatToolCallParser()
        result = parser.parse(
            _make_response(content="<tool_call><function=web_search>incomplete"),
            _ALL_TOOLS,
        )
        assert result == []


class TestDanglingJsonRecovery:
    """Recover ``<tool_call>{...}`` truncated mid-stream by ``max_tokens``.

    Real failure mode observed on qwen3.5-397b heavy_mode_smoke when a
    long ``assign_task`` payload pushed the closing ``</tool_call>`` past
    the response cap. JSON body is preserved up to the close brace; only
    the tag is missing. Recovery extracts the balanced JSON object.
    """

    def test_recovers_complete_json_missing_close_tag(self):
        parser = MultiFormatToolCallParser()
        result = parser.parse(
            _make_response(content=(
                '<tool_call>{"tool": "web_search", "args": '
                '{"query": "NVIDIA H100"}}'
            )),
            _ALL_TOOLS,
        )
        assert len(result) == 1
        assert result[0]["name"] == "web_search"
        assert result[0]["args"] == {"query": "NVIDIA H100"}

    def test_recovers_nested_json_missing_close_tag(self):
        parser = MultiFormatToolCallParser()
        result = parser.parse(
            _make_response(content=(
                '<tool_call>{"tool": "assign_task", "args": '
                '{"tasks": [{"agent": "researcher", "prompt": "find X"}]}}'
            )),
            _ALL_TOOLS,
        )
        assert len(result) == 1
        assert result[0]["name"] == "assign_task"
        assert result[0]["args"]["tasks"][0]["agent"] == "researcher"

    def test_does_not_recover_truncation_inside_string(self):
        """Truncation mid-string is unrecoverable — string never closes."""
        parser = MultiFormatToolCallParser()
        result = parser.parse(
            _make_response(content=(
                '<tool_call>{"tool": "assign_task", "args": '
                '{"tasks": [{"agent": "x", "prompt": "Research the 96th Aca'
            )),
            _ALL_TOOLS,
        )
        assert result == []

    def test_does_not_recover_unknown_tool(self):
        parser = MultiFormatToolCallParser()
        result = parser.parse(
            _make_response(content=(
                '<tool_call>{"tool": "nope_not_a_tool", "args": {}}'
            )),
            _ALL_TOOLS,
        )
        assert result == []

    def test_skipped_when_close_tag_present(self):
        """Closed ``<tool_call>...</tool_call>`` goes through the regular
        JSON path, not the dangling recovery. Guard against
        double-parsing."""
        parser = MultiFormatToolCallParser()
        result = parser.parse(
            _make_response(content=(
                '<tool_call>{"tool": "web_search", "args": '
                '{"query": "x"}}</tool_call>'
            )),
            _ALL_TOOLS,
        )
        assert len(result) == 1
        # ID is set by the JSON path (no ``id`` field) — dangling
        # recovery would have used ``dangling_tc_0``.
        assert result[0].get("id") != "dangling_tc_0"

    def test_native_still_beats_dangling(self):
        parser = MultiFormatToolCallParser()
        result = parser.parse(
            _make_response(
                tool_calls=[
                    {"name": "web_fetch", "args": {"url": "u"}, "id": "n1"},
                ],
                content='<tool_call>{"tool": "web_search", "args": {"query": "x"}}',
            ),
            _ALL_TOOLS,
        )
        assert len(result) == 1
        assert result[0]["name"] == "web_fetch"


class TestProtocolCompliance:
    def test_is_tool_call_parser(self):
        assert isinstance(MultiFormatToolCallParser(), ToolCallParser)

    def test_is_subclass_of_default(self):
        assert issubclass(MultiFormatToolCallParser, DefaultToolCallParser)

