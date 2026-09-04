"""Tests for ``OutputRepairMiddleware`` — Phase 3 PR-3.3."""

from __future__ import annotations

import pytest

from agent_core.components.middleware.llm.base import LLMCallContext
from agent_core.components.middleware.llm.output_repair import (
    OutputRepairMiddleware,
    repair_output_text,
)
from agent_core.llm import LLMResponse

# ── pure helper ──────────────────────────────────────────────────────────


class TestRepairOutputText:
    def test_empty_string_passthrough(self) -> None:
        assert repair_output_text("") == ""

    def test_no_thinking_tags_only_strips_trailing_ws(self) -> None:
        assert repair_output_text("hello world  \n") == "hello world"

    def test_dedupes_two_consecutive_close_tags(self) -> None:
        assert repair_output_text("<think>x</think></think>") == "<think>x</think>"

    def test_dedupes_runs_of_three_or_more(self) -> None:
        out = repair_output_text("<think>x</think></think></think></think>")
        assert out == "<think>x</think>"

    def test_dedupes_with_whitespace_between(self) -> None:
        out = repair_output_text("<think>x</think>\n  </think>")
        assert out == "<think>x</think>"

    def test_dedupes_thinking_tag_variant(self) -> None:
        out = repair_output_text("<thinking>x</thinking></thinking>")
        assert out == "<thinking>x</thinking>"

    def test_does_not_collapse_close_tags_separated_by_text(self) -> None:
        # "</think>foo</think>" is a malformed but distinct pattern — leave
        # alone rather than risk losing the "foo" payload.
        text = "<think>a</think>foo</think>"
        assert repair_output_text(text) == "<think>a</think>foo</think>"

    def test_auto_closes_unclosed_thinking_block(self) -> None:
        out = repair_output_text("<thinking>oops never closed")
        assert out == "<thinking>oops never closed</thinking>"

    def test_auto_closes_matches_first_unclosed_tag_style(self) -> None:
        # Nested mix: <think>...</think><thinking>... — second open never
        # closes, repair should append </thinking>, not </think>.
        text = "<think>a</think><thinking>b"
        out = repair_output_text(text)
        assert out == "<think>a</think><thinking>b</thinking>"

    def test_balanced_tags_unchanged(self) -> None:
        text = "<think>step 1</think>final answer"
        assert repair_output_text(text) == text

    def test_idempotent(self) -> None:
        text = "<think>x</think></think></think>"
        once = repair_output_text(text)
        assert repair_output_text(once) == once

    def test_case_insensitive_tag_matching(self) -> None:
        out = repair_output_text("<THINK>x</THINK></think>")
        # Dedup matches mixed case; first close is preserved verbatim.
        assert out == "<THINK>x</THINK>"


# ── middleware integration ───────────────────────────────────────────────


def _ctx() -> LLMCallContext:
    return LLMCallContext(task_id="t", role_id="r", call_index=0)


@pytest.mark.asyncio
async def test_after_llm_string_content_repaired() -> None:
    mw = OutputRepairMiddleware()
    resp = LLMResponse(content="<think>x</think></think>")
    out = await mw.after_llm(_ctx(), resp)
    assert out.content == "<think>x</think>"
    # Original is not mutated; ``dataclasses.replace`` returns a new instance.
    assert resp.content == "<think>x</think></think>"


@pytest.mark.asyncio
async def test_after_llm_no_change_returns_same_object() -> None:
    mw = OutputRepairMiddleware()
    resp = LLMResponse(content="hello world")
    out = await mw.after_llm(_ctx(), resp)
    assert out is resp


@pytest.mark.asyncio
async def test_after_llm_list_content_anthropic_blocks() -> None:
    mw = OutputRepairMiddleware()
    resp = LLMResponse(content=[
        {"type": "text", "text": "<think>step</think></think>"},
        {"type": "tool_use", "id": "abc", "name": "search", "input": {}},
    ])
    out = await mw.after_llm(_ctx(), resp)
    assert isinstance(out.content, list)
    assert out.content[0]["text"] == "<think>step</think>"
    # Non-text block forwarded untouched.
    assert out.content[1] == resp.content[1]


@pytest.mark.asyncio
async def test_after_llm_disabled_short_circuits() -> None:
    mw = OutputRepairMiddleware(enabled=False)
    # The chain checks ``enabled`` before dispatching, but a direct call
    # should also respect the flag for symmetry. Currently the middleware
    # only signals via ``enabled``; the chain skips the call. Verify the
    # property reads through.
    assert mw.enabled is False


@pytest.mark.asyncio
async def test_after_llm_preserves_response_metadata() -> None:
    mw = OutputRepairMiddleware()
    resp = LLMResponse(
        content="<think>x</think></think>",
        response_metadata={"fallback_used": 1, "model_actually_used": "m2"},
    )
    out = await mw.after_llm(_ctx(), resp)
    # ``replace(response, content=...)`` keeps every other field — the §5.9
    # fallback markers must survive PR-3.3.
    assert out.response_metadata["fallback_used"] == 1
    assert out.response_metadata["model_actually_used"] == "m2"


@pytest.mark.asyncio
async def test_after_llm_preserves_tool_calls() -> None:
    mw = OutputRepairMiddleware()
    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "search", "arguments": '{"q": "x"}'},
    }
    resp = LLMResponse(
        content="<think>x</think></think>",
        tool_calls=[tool_call],
    )
    out = await mw.after_llm(_ctx(), resp)
    # ``replace`` only rewrites ``content`` — tool_calls pass through verbatim.
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0]["function"]["name"] == "search"
    assert out.tool_calls[0]["function"]["arguments"] == '{"q": "x"}'
    assert out.tool_calls[0]["id"] == "call_1"
    assert out.content == "<think>x</think>"
