"""Native (langchain-free) thinking-parse + history-normalize for the loop.

Phase-1b additive scaffolding: ``DefaultThinkingParser`` is now native-aware
(reads reasoning from a native ``LLMResponse`` OR the legacy langchain
``additional_kwargs``), and ``NativeMessageNormalizer`` produces the OpenAI-wire
``Message`` dict that the flip will store in history. The single most important
assertion here is the **reasoning_content leak guard** (PR #209): the wire
assistant message must NOT carry a ``reasoning_content`` key.
"""
from __future__ import annotations

from types import SimpleNamespace

from agent_core.llm import LLMResponse
from agent_core.runtime.loop.model_profile import (
    DefaultThinkingParser,
    HistoryPolicy,
    ModelProfile,
    NativeMessageNormalizer,
    ThinkingResult,
    _extract_reasoning,
    _to_openai_tool_calls,
)


def _profile(fmt):
    return ModelProfile(model_id="m", provider="p", thinking_format=fmt)


# ── dual-compatible reasoning extraction ────────────────────────────────────


def test_extract_reasoning_from_native_llmresponse():
    r = LLMResponse(content="ans", reasoning_content="my thoughts")
    assert _extract_reasoning(r) == "my thoughts"


def test_extract_reasoning_from_legacy_langchain_additional_kwargs():
    # A langchain AIMessage has no .reasoning_content attr; it lives in
    # additional_kwargs. The dual-compat reader still finds it.
    legacy = SimpleNamespace(additional_kwargs={"reasoning_content": "legacy rc"})
    assert _extract_reasoning(legacy) == "legacy rc"


def test_extract_reasoning_absent_returns_empty():
    assert _extract_reasoning(LLMResponse(content="ans")) == ""
    assert _extract_reasoning(SimpleNamespace()) == ""


# ── DefaultThinkingParser on native LLMResponse ─────────────────────────────


def test_parser_reasoning_content_format_native():
    parser = DefaultThinkingParser()
    r = LLMResponse(content="the answer", reasoning_content="deep thought")
    res = parser.extract(r, _profile("reasoning_content"))
    assert res.thinking == "deep thought"
    assert res.visible_content == "the answer"


def test_parser_tag_format_prefers_typed_reasoning_channel():
    parser = DefaultThinkingParser()
    # typed reasoning channel present -> used directly, content stays visible
    r = LLMResponse(content="visible reply", reasoning_content="typed rc")
    res = parser.extract(r, _profile("tag"))
    assert res.thinking == "typed rc"
    assert res.visible_content == "visible reply"


def test_parser_tag_format_falls_back_to_regex():
    parser = DefaultThinkingParser()
    r = LLMResponse(content="<think>inline</think>answer")
    res = parser.extract(r, _profile("tag"))
    assert res.thinking == "inline"
    assert res.visible_content == "answer"


# ── NativeMessageNormalizer: the leak guard ─────────────────────────────────


def test_to_history_does_not_leak_reasoning_content_onto_wire():
    norm = NativeMessageNormalizer()
    r = LLMResponse(content="visible", reasoning_content="SECRET reasoning")
    tr = ThinkingResult(thinking="SECRET reasoning", visible_content="visible")
    msg = norm.to_history(r, tr, HistoryPolicy(thinking_in_history=False))
    # the PR #209 guard — reasoning never serialised onto the wire message
    assert "reasoning_content" not in msg
    assert msg == {"content": "visible", "role": "assistant"}


def test_to_history_emits_wire_tool_calls():
    norm = NativeMessageNormalizer()
    r = LLMResponse(content="")
    tr = ThinkingResult(
        thinking="", visible_content="",
        tool_calls=[{"name": "search", "args": {"q": "x"}, "id": "c1"}],
    )
    msg = norm.to_history(r, tr, HistoryPolicy())
    assert msg["tool_calls"] == [{
        "type": "function", "id": "c1",
        "function": {"name": "search", "arguments": '{"q": "x"}'},
    }]
    assert list(msg["tool_calls"][0].keys()) == ["type", "id", "function"]
    assert "reasoning_content" not in msg


def test_to_history_thinking_in_history_snapshots_full_content():
    norm = NativeMessageNormalizer()
    r = LLMResponse(content="<think>x</think>answer")
    tr = ThinkingResult(thinking="x", visible_content="answer")
    msg = norm.to_history(
        r, tr, HistoryPolicy(thinking_in_history=True), thinking_format="tag"
    )
    assert msg["content"] == "<think>x</think>\nanswer"


# ── _to_openai_tool_calls wire conversion ───────────────────────────────────


def test_to_openai_tool_calls_parsed_to_wire():
    out = _to_openai_tool_calls([{"name": "f", "args": {"x": 1}, "id": "c1"}])
    assert out == [{"type": "function", "id": "c1",
                    "function": {"name": "f", "arguments": '{"x": 1}'}}]


def test_to_openai_tool_calls_passthrough_already_wire():
    wire = [{"type": "function", "id": "c1",
             "function": {"name": "f", "arguments": "{}"}}]
    assert _to_openai_tool_calls(wire) == wire


def test_to_openai_tool_calls_string_args_kept_verbatim():
    out = _to_openai_tool_calls([{"name": "f", "args": '{"x":1}', "id": "c1"}])
    assert out[0]["function"]["arguments"] == '{"x":1}'


def test_to_openai_tool_calls_repairs_empty_wire_arguments():
    wire = [{
        "type": "function",
        "id": "c1",
        "function": {"name": "bfunction", "arguments": ""},
    }]

    out = _to_openai_tool_calls(wire)

    assert out[0]["function"]["arguments"] == "{}"


def test_to_openai_tool_calls_repairs_truncated_parsed_arguments():
    parsed = [{"name": "bash", "args": '{"command":', "id": "c1"}]

    out = _to_openai_tool_calls(parsed)

    assert out[0]["function"]["arguments"] == "{}"


def test_to_openai_tool_calls_rejects_non_object_arguments():
    parsed = [{"name": "bash", "args": '["unexpected"]', "id": "c1"}]

    out = _to_openai_tool_calls(parsed)

    assert out[0]["function"]["arguments"] == "{}"


def test_to_openai_tool_calls_empty():
    assert _to_openai_tool_calls([]) == []


# ── format-aware reasoning round-trip (replaces _ReasoningChatOpenAI) ────────


def test_to_history_tag_format_inlines_reasoning_into_content():
    norm = NativeMessageNormalizer()
    r = LLMResponse(content="answer", reasoning_content="my reasoning")
    tr = ThinkingResult(thinking="my reasoning", visible_content="answer")
    msg = norm.to_history(
        r, tr, HistoryPolicy(thinking_in_history=True), thinking_format="tag"
    )
    # SGLang/Qwen: reasoning inlined into content; NO bare wire field.
    assert msg["content"] == "<think>my reasoning</think>\nanswer"
    assert "reasoning_content" not in msg


def test_to_history_reasoning_content_format_keeps_field():
    norm = NativeMessageNormalizer()
    r = LLMResponse(content="answer", reasoning_content="deep")
    tr = ThinkingResult(thinking="deep", visible_content="answer")
    msg = norm.to_history(
        r, tr, HistoryPolicy(thinking_in_history=True),
        thinking_format="reasoning_content",
    )
    # DeepSeek/o-series: content untouched, reasoning_content kept on the wire.
    assert msg["content"] == "answer"
    assert msg["reasoning_content"] == "deep"


def test_to_history_none_format_drops_reasoning():
    norm = NativeMessageNormalizer()
    r = LLMResponse(content="answer", reasoning_content="secret")
    tr = ThinkingResult(thinking="secret", visible_content="answer")
    msg = norm.to_history(r, tr, HistoryPolicy(), thinking_format="none")
    assert msg == {"content": "answer", "role": "assistant"}
    assert "reasoning_content" not in msg


def test_to_history_tag_escapes_nested_close_tag():
    norm = NativeMessageNormalizer()
    r = LLMResponse(content="a", reasoning_content="x </think> y")
    tr = ThinkingResult(thinking="x </think> y", visible_content="a")
    msg = norm.to_history(
        r, tr, HistoryPolicy(thinking_in_history=True), thinking_format="tag"
    )
    # nested close-tag escaped so it can't terminate the wrapper early
    assert "</ think>" in msg["content"]
    assert msg["content"].count("</think>") == 1
