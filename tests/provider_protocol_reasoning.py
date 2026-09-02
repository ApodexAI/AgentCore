"""Per-profile ``protocol`` switch + verbatim reasoning capture/replay.

Covers the native Anthropic extended-thinking path, the new OpenAI Responses
client (pure conversion/parse helpers — no live API), and the cross-cutting
plumbing that carries reasoning blocks through the loop and back onto the wire:

- Anthropic: ``thinking=`` is sent (temperature dropped), signed thinking blocks
  round-trip verbatim on re-send, reasoning tokens surface in usage.
- Responses: messages → ``input`` items (reasoning items re-sent with
  encrypted_content), tools flattened, output parsed to a block list, usage
  normalised.
- model_profile ``content_block`` parser keeps ``raw_content_blocks`` and
  ``to_history`` replays them unmodified; ``TurnContext`` / ``extract_usage`` /
  ``node_context`` carry them end-to-end.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_core.llm import LLMClient, LLMResponse
from agent_core.messages import assistant_msg, tool_msg, user_msg
from agent_core.providers import anthropic as ac
from agent_core.providers import openai_responses as rc
from agent_core.providers.anthropic import AnthropicClient
from agent_core.providers.openai_responses import OpenAIResponsesClient
from agent_core.providers.protocol_client import (
    build_protocol_client,
    protocol_of,
    thinking_format_for_protocol,
)


@pytest.mark.parametrize("value", [["anthropic"], {"protocol": "anthropic"}, 5, True, "unknown"])
def test_protocol_of_rejects_unusable_values(value):
    assert protocol_of({"protocol": value}) == "chat_completions"

# ── Anthropic extended thinking ─────────────────────────────────────────────


def test_anthropic_build_kwargs_sends_thinking_and_drops_temperature():
    c = AnthropicClient(
        "claude-x", api_key="x", temperature=0.3,
        thinking={"type": "adaptive", "display": "summarized"}, effort="high",
    )
    kw = c._build_kwargs(
        [user_msg("hi")], tools=None, temperature=0.9,
        max_tokens=None, extra_headers=None, timeout=None,
    )
    assert kw["thinking"] == {"type": "adaptive", "display": "summarized"}
    # Anthropic 400s on temperature + thinking → temperature must be omitted.
    assert "temperature" not in kw
    # effort rides on extra_body.output_config (bypasses the SDK Literal check).
    assert kw["extra_body"] == {"output_config": {"effort": "high"}}


def test_anthropic_build_kwargs_keeps_temperature_when_thinking_off():
    c = AnthropicClient("claude-x", api_key="x", temperature=0.3)
    kw = c._build_kwargs(
        [user_msg("hi")], tools=None, temperature=None,
        max_tokens=None, extra_headers=None, timeout=None,
    )
    assert kw["temperature"] == 0.3
    assert "thinking" not in kw and "extra_body" not in kw


def test_anthropic_assistant_roundtrips_verbatim_signed_blocks():
    # History kept the verbatim block list (content is a list, not a string).
    m = assistant_msg([
        {"type": "thinking", "thinking": "reason", "signature": "SIG"},
        {"type": "redacted_thinking", "data": "REDACTED"},
        {"type": "text", "text": "answer"},
    ], tool_calls=[{"id": "c1", "type": "function",
                    "function": {"name": "f", "arguments": '{"a": 1}'}}])
    out = ac._to_anthropic_msg(m)
    assert out["role"] == "assistant"
    kinds = [b["type"] for b in out["content"]]
    assert kinds == ["thinking", "redacted_thinking", "text", "tool_use"]
    assert out["content"][0]["signature"] == "SIG"      # signature echoed back
    assert out["content"][1]["data"] == "REDACTED"
    assert out["content"][3] == {
        "type": "tool_use", "id": "c1", "name": "f", "input": {"a": 1},
    }


def test_anthropic_plain_string_assistant_unchanged():
    out = ac._to_anthropic_msg(assistant_msg("just text"))
    assert out == {"role": "assistant", "content": [{"type": "text", "text": "just text"}]}


def test_anthropic_response_preserves_redacted_thinking_block():
    # Inbound redacted_thinking must survive verbatim (raw_content_blocks) so the
    # outbound replay (which already echoes redacted_thinking) has it to re-send;
    # dropping it would break signature/replay continuity. Its presence also
    # forces the structured block-list content (no readable text to key on).
    raw = SimpleNamespace(
        content=[SimpleNamespace(type="redacted_thinking", data="REDACTED"),
                 SimpleNamespace(type="text", text="answer")],
        usage=SimpleNamespace(input_tokens=5, output_tokens=8,
                              cache_read_input_tokens=0, output_tokens_details=None),
        stop_reason="end_turn", model="claude-x", id="m1",
    )
    resp = ac._to_llm_response(raw)
    assert isinstance(resp.content, list)
    kinds = [b["type"] for b in resp.content]
    assert kinds == ["redacted_thinking", "text"]
    assert resp.content[0] == {"type": "redacted_thinking", "data": "REDACTED"}


def test_provider_label_derives_from_protocol():
    from agent_core.providers.protocol_client import provider_label
    assert provider_label({"protocol": "anthropic"}) == "anthropic"
    assert provider_label({"protocol": "bedrock"}) == "bedrock"
    assert provider_label({"protocol": "responses"}) == "openai"
    assert provider_label({"protocol": "chat_completions"}) == "openai"
    assert provider_label({}) == "openai"
    # explicit label / provider wins over the protocol-derived default.
    assert provider_label({"protocol": "anthropic", "provider": "new_api"}) == "new_api"
    assert provider_label({"protocol": "bedrock", "_provider_label": "x"}) == "x"


def test_anthropic_usage_surfaces_reasoning_tokens():
    raw = SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking="t", signature="s"),
                 SimpleNamespace(type="text", text="a")],
        usage=SimpleNamespace(
            input_tokens=10, output_tokens=20, cache_read_input_tokens=2,
            output_tokens_details=SimpleNamespace(thinking_tokens=7)),
        stop_reason="end_turn", model="claude-x", id="m1",
    )
    resp = ac._to_llm_response(raw)
    assert resp.usage["reasoning_tokens"] == 7
    assert resp.usage["completion_tokens"] == 20


# ── OpenAI Responses client (pure conversions) ──────────────────────────────


def test_responses_tools_flattened():
    out = rc._to_responses_tools(
        [{"type": "function", "function": {
            "name": "search", "description": "d", "parameters": {"type": "object"}}}])
    assert out == [{"type": "function", "name": "search",
                    "description": "d", "parameters": {"type": "object"}}]


def test_responses_input_reemits_reasoning_and_tool_calls():
    messages = [
        user_msg("q"),
        assistant_msg(
            [{"type": "reasoning", "id": "rs_1",
              "summary": [{"type": "summary_text", "text": "plan"}],
              "encrypted_content": "ENC"},
             {"type": "text", "text": "calling tool"}],
            tool_calls=[{"id": "call_1", "type": "function",
                         "function": {"name": "f", "arguments": '{"a":1}'}}]),
        tool_msg("result", "call_1"),
    ]
    items = rc._to_responses_input(messages)
    types = [it.get("type") or ("message:" + it.get("role", "")) for it in items]
    # user message, then reasoning item FIRST, assistant message, function_call,
    # then the tool result as function_call_output.
    assert types == [
        "message:user", "reasoning", "message", "function_call",
        "function_call_output",
    ]
    reasoning_item = items[1]
    assert reasoning_item["id"] == "rs_1"
    assert reasoning_item["encrypted_content"] == "ENC"   # replayed verbatim
    fc = items[3]
    assert fc["call_id"] == "call_1" and fc["name"] == "f"
    fco = items[4]
    assert fco["call_id"] == "call_1" and fco["output"] == "result"


def test_responses_parse_output_keeps_reasoning_blocks_and_tool_calls():
    raw = SimpleNamespace(
        output=[
            SimpleNamespace(type="reasoning", id="rs_1",
                            summary=[SimpleNamespace(type="summary_text", text="plan")],
                            encrypted_content="ENC"),
            SimpleNamespace(type="message", content=[
                SimpleNamespace(type="output_text", text="the answer")]),
            SimpleNamespace(type="function_call", call_id="call_1",
                            name="f", arguments='{"a":1}'),
        ],
        usage=SimpleNamespace(
            input_tokens=10, output_tokens=30, total_tokens=40,
            input_tokens_details=SimpleNamespace(cached_tokens=2),
            output_tokens_details=SimpleNamespace(reasoning_tokens=12)),
        status="completed", model="gpt-x", id="resp_1",
    )
    resp = rc._parse_responses_output(raw)
    assert isinstance(resp.content, list)
    assert resp.content[0]["type"] == "reasoning"
    assert resp.content[0]["encrypted_content"] == "ENC"
    assert resp.content[1] == {"type": "text", "text": "the answer"}
    assert resp.reasoning_content == "plan"
    assert resp.tool_calls[0]["id"] == "call_1"
    assert resp.tool_calls[0]["function"]["name"] == "f"
    assert resp.usage == {
        "prompt_tokens": 10, "completion_tokens": 30, "total_tokens": 40,
        "cached_tokens": 2, "reasoning_tokens": 12,
    }


@pytest.mark.asyncio
async def test_responses_chat_sends_reasoning_include_and_store():
    from unittest.mock import AsyncMock
    c = OpenAIResponsesClient("gpt-x", api_key="x",
                              reasoning={"effort": "high", "summary": "auto"})
    raw = SimpleNamespace(output=[SimpleNamespace(
        type="message", content=[SimpleNamespace(type="output_text", text="ok")])],
        usage=None, status="completed", model="gpt-x", id="r1")
    create = AsyncMock(return_value=raw)
    c._client = SimpleNamespace(responses=SimpleNamespace(create=create))
    resp = await c.chat([user_msg("hi")],
                        tools=[{"type": "function", "function": {"name": "f"}}])
    kw = create.call_args.kwargs
    assert kw["include"] == ["reasoning.encrypted_content"]
    assert kw["store"] is False
    assert kw["reasoning"] == {"effort": "high", "summary": "auto"}
    assert kw["tools"][0]["name"] == "f"          # flattened tool schema
    assert isinstance(resp, LLMResponse)


def test_responses_client_satisfies_protocol():
    assert isinstance(OpenAIResponsesClient("gpt-x", api_key="x"), LLMClient)


# ── protocol_client selection ───────────────────────────────────────────────


def test_anthropic_builder_defaults_to_adaptive_thinking():
    # Default thinking_type=adaptive → the recommended / only mode on current
    # Claude (enabled is rejected with 400 on Opus 4.7/4.8, Sonnet 5).
    # display defaults to summarized so the readable thinking text is captured.
    c = build_protocol_client(
        {"model": "claude", "protocol": "anthropic", "max_tokens": 2048,
         "effort": "high"}, title="T")
    kw = c._build_kwargs([user_msg("hi")], tools=None, temperature=None,
                         max_tokens=None, extra_headers=None, timeout=None)
    assert kw["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert "temperature" not in kw
    # adaptive forwards effort → output_config.effort.
    assert kw["extra_body"] == {"output_config": {"effort": "high"}}


def test_anthropic_builder_enabled_is_legacy_optin_and_drops_effort():
    # Opt-in enabled (legacy models < Opus 4.6): budget clamped < max_tokens,
    # and effort is NOT forwarded (oldest models 400 on enabled+effort;
    # budget_tokens is the control knob for enabled).
    c = build_protocol_client(
        {"model": "claude", "protocol": "anthropic", "max_tokens": 2048,
         "thinking_type": "enabled", "effort": "high"}, title="T")
    kw = c._build_kwargs([user_msg("hi")], tools=None, temperature=None,
                         max_tokens=None, extra_headers=None, timeout=None)
    assert kw["thinking"] == {"type": "enabled", "budget_tokens": 2047}
    assert "temperature" not in kw
    assert "extra_body" not in kw          # effort dropped for enabled


def test_protocol_client_selection():
    assert isinstance(
        build_protocol_client({"model": "claude", "protocol": "anthropic"}, title="T"),
        AnthropicClient)
    assert isinstance(
        build_protocol_client({"model": "gpt", "protocol": "responses"}, title="T"),
        OpenAIResponsesClient)
    # chat_completions (and default) → None: caller keeps its OpenAIClient path.
    assert build_protocol_client({"model": "gpt"}, title="T") is None
    assert build_protocol_client({"model": "gpt", "protocol": "chat_completions"},
                                 title="T") is None


def test_thinking_format_for_protocol():
    assert thinking_format_for_protocol("anthropic") == "content_block"
    assert thinking_format_for_protocol("responses") == "content_block"
    assert thinking_format_for_protocol("bedrock") == "content_block"
    assert thinking_format_for_protocol("chat_completions") is None
    assert protocol_of({}) == "chat_completions"
    assert protocol_of({"protocol": "ANTHROPIC"}) == "anthropic"


def test_bedrock_protocol_builds_anthropic_client_with_thinking():
    # protocol=bedrock → AnthropicClient over the Bedrock transport (Bearer
    # API-key + /model/{id}/invoke). Same thinking request shape as direct
    # Anthropic (adaptive default, temperature dropped). Bedrock signature is
    # cross-platform compatible so replay works the same.
    c = build_protocol_client(
        {"model": "global.anthropic.claude-sonnet-4-6", "protocol": "bedrock",
         "api_key": "ABSK-x", "base_url": "https://bedrock-runtime.us-east-1.amazonaws.com",
         "max_tokens": 2048}, title="T")
    assert isinstance(c, AnthropicClient)
    kw = c._build_kwargs([user_msg("hi")], tools=None, temperature=None,
                         max_tokens=None, extra_headers=None, timeout=None)
    assert kw["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert "temperature" not in kw


# ── Cross-cutting plumbing ──────────────────────────────────────────────────


def test_content_block_parser_and_history_roundtrip():
    from agent_core.runtime.loop.model_profile import (
        DefaultThinkingParser,
        HistoryPolicy,
        ModelProfile,
        NativeMessageNormalizer,
    )
    blocks = [
        {"type": "thinking", "thinking": "think", "signature": "SIG"},
        {"type": "text", "text": "answer"},
    ]
    resp = LLMResponse(content=blocks, reasoning_content="think")
    prof = ModelProfile(model_id="claude", provider="anthropic",
                        thinking_format="content_block", protocol="anthropic")
    tr = DefaultThinkingParser().extract(resp, prof)
    assert tr.raw_content_blocks == blocks
    assert tr.thinking == "think" and tr.visible_content == "answer"
    hist = NativeMessageNormalizer().to_history(resp, tr, HistoryPolicy(), "content_block")
    # Verbatim replay: history content is the exact block list incl. signature.
    assert hist["content"] == blocks


def test_content_block_parser_handles_plain_string_content():
    # Regression: a content_block turn can arrive as a BARE STRING — the native
    # Anthropic client returns a string whenever a turn has no thinking block
    # (adaptive thinking omitted / the post-tool-call final answer turn). The
    # parser must NOT iterate it char-by-char (which silently dropped the whole
    # visible answer); the string is the visible content.
    from agent_core.runtime.loop.model_profile import (
        DefaultThinkingParser,
        ModelProfile,
    )
    prof = ModelProfile(model_id="claude", provider="anthropic",
                        thinking_format="content_block", protocol="anthropic")
    resp = LLMResponse(content="MIRO-4271-ZK", reasoning_content="")
    tr = DefaultThinkingParser().extract(resp, prof)
    assert tr.visible_content == "MIRO-4271-ZK"
    assert tr.raw_content_blocks is None       # nothing to replay this turn


def test_turn_context_and_extract_usage_carry_reasoning():
    from agent_core.loop_types import TurnContext
    from agent_core.runtime.loop.llm_client import extract_usage

    tc = TurnContext(
        turn=1, max_turns=5, task_id="t", role_id="r", ai_text="a",
        thinking="th", tool_calls=[], messages=[], usage=None, metadata={},
    )
    assert tc.thinking_blocks == []          # default empty for other formats

    resp = LLMResponse(model="m", usage={
        "prompt_tokens": 5, "completion_tokens": 10, "reasoning_tokens": 4})
    u = extract_usage(resp)
    assert u["reasoning_tokens"] == 4
    # The normalized shape is stable even when no reasoning was used.
    resp2 = LLMResponse(model="m", usage={"prompt_tokens": 5, "completion_tokens": 10})
    assert extract_usage(resp2)["reasoning_tokens"] == 0
