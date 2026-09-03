"""Tests for kernel/loop/compact.py::compact_messages.

The summary block must preserve enough tool-call metadata that the LLM
can re-fetch a source it dropped when the naive truncation would have
cut off the URL.
"""

from __future__ import annotations

import json

from agent_core.messages import (
    ToolCall,
    assistant_msg,
    is_tool_msg,
    system_msg,
    text_of,
    user_msg,
)
from agent_core.runtime.loop.compact import (
    CompactionPolicy,
    DefaultCompactionPolicy,
    DefaultMessageCompactor,
    MessageCompactor,
    compact_messages,
)


def _long(filler: str, n: int) -> str:
    return (filler + " ") * n


def _tc(call_id: str, name: str, args: dict) -> ToolCall:
    """Build a native wire tool_call: {id, type, function:{name, arguments}}."""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def _tool_msg(content: str, tool_call_id: str, name: str) -> dict:
    return {
        "content": content,
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": name,
    }


def test_noop_when_under_keep_recent():
    msgs = [
        system_msg("sys"),
        user_msg("q"),
        assistant_msg("a"),
    ]
    out = compact_messages(msgs, keep_recent=5)
    assert out == msgs


def test_system_msg_preserved_and_recent_verbatim():
    msgs = [
        system_msg("sys"),
        *[user_msg(f"old-{i}") for i in range(10)],
        user_msg("recent-1"),
        user_msg("recent-2"),
    ]
    out = compact_messages(msgs, keep_recent=2)
    # [system, original task, compact_summary, recent-1, recent-2]
    assert len(out) == 5
    assert out[0].get("role") == "system"
    assert out[0].get("content") == "sys"
    assert out[1].get("role") == "user"
    assert out[1] is msgs[1]
    assert text_of(out[2].get("content")).startswith("[Compacted")
    assert "recent-1" in text_of(out[3].get("content"))
    assert "recent-2" in text_of(out[4].get("content"))


def test_tool_url_preserved_in_summary():
    """A tool message with a URL in the body keeps the URL in the summary."""
    url = "https://example.com/article-42"
    body = "Title: X\nFull article text: " + _long("lorem", 200) + "\nSource: " + url
    msgs = [
        system_msg("sys"),
        user_msg("q"),
        assistant_msg("", tool_calls=[_tc("1", "web_search", {"query": "X"})]),
        _tool_msg(body, "1", "web_search"),
        user_msg("follow-up"),
        user_msg("recent"),
    ]
    out = compact_messages(msgs, keep_recent=1)
    summary = text_of(out[2].get("content"))
    # Even though body was 1000+ chars, the URL survives.
    assert url in summary
    # Tool name is in the summary
    assert "web_search" in summary


def test_ai_tool_calls_preserved_in_summary():
    """An assistant message with tool_calls gets name + args preview in the summary."""
    msgs = [
        system_msg("sys"),
        user_msg("q"),
        assistant_msg(
            "I'll search",
            tool_calls=[_tc("1", "web_search", {"query": "NVIDIA H100"})],
        ),
        _tool_msg("result", "1", "web_search"),
        user_msg("next"),
        user_msg("recent"),
    ]
    out = compact_messages(msgs, keep_recent=1)
    summary = text_of(out[2].get("content"))
    assert "web_search" in summary
    assert "NVIDIA H100" in summary


def test_split_not_on_tool_message():
    """Split point must not leave an orphan tool message at the head of recent."""
    msgs = [
        system_msg("sys"),
        user_msg("q1"),
        assistant_msg("", tool_calls=[_tc("1", "t", {})]),
        _tool_msg("r1", "1", "t"),
        user_msg("q2"),
        assistant_msg("", tool_calls=[_tc("2", "t", {})]),
        _tool_msg("r2", "2", "t"),  # would be recent[0]
        user_msg("final"),
    ]
    # keep_recent=2 would try to split right before the tool message r2 — bad.
    out = compact_messages(msgs, keep_recent=2)
    # The message right after the summary must not be a bare tool message.
    first_after_summary = out[2]
    assert not is_tool_msg(first_after_summary), (
        "split landed on a tool message, orphaned tool_call_id"
    )


def test_skips_prior_compacted_summary():
    """An already-compacted summary in history is not re-wrapped."""
    msgs = [
        system_msg("sys"),
        user_msg("[Compacted 10 earlier messages]\n[User: q]"),
        *[user_msg(f"t-{i}") for i in range(8)],
        user_msg("recent"),
    ]
    out = compact_messages(msgs, keep_recent=1)
    summary = text_of(out[1].get("content"))
    # The prior [Compacted ...] content is skipped (not double-nested)
    assert summary.count("[Compacted") == 1


def test_recent_window_kept_verbatim():
    msgs = [
        system_msg("sys"),
        *[user_msg(f"old-{i}") for i in range(12)],
        user_msg("R1"),
        assistant_msg("R2"),
        user_msg("R3"),
    ]
    out = compact_messages(msgs, keep_recent=3)
    assert text_of(out[-3].get("content")) == "R1"
    assert text_of(out[-2].get("content")) == "R2"
    assert text_of(out[-1].get("content")) == "R3"


def test_empty_middle_content_skipped():
    """An assistant message with no text and no tool_calls contributes nothing."""
    msgs = [
        system_msg("sys"),
        assistant_msg(""),
        assistant_msg(""),
        user_msg("useful"),
        user_msg("recent"),
    ]
    out = compact_messages(msgs, keep_recent=1)
    summary = text_of(out[1].get("content"))
    # Only one User snippet from the middle, no empty-agent lines.
    assert "[User: useful]" in summary
    assert "[Agent:" not in summary


# ── Pluggable compactor ────────────────────────────────────────────────


def test_default_compactor_matches_compact_messages():
    """DefaultMessageCompactor.compact(...) must match compact_messages(...) 1:1."""
    msgs = [
        system_msg("sys"),
        *[user_msg(f"old-{i}") for i in range(8)],
        user_msg("r"),
    ]
    compactor = DefaultMessageCompactor()
    assert compactor.compact(msgs, 1) == compact_messages(msgs, 1)


def test_custom_compactor_runs_and_is_protocol_compliant():
    """A workflow-owned compactor just needs a ``compact(messages, keep_recent)``
    method — ``MessageCompactor`` is a ``Protocol``, no base class needed."""

    class KeepOnlySystem:
        """Trivial strategy: drop everything but system prompts."""

        def compact(self, messages, keep_recent):
            return [m for m in messages if m.get("role") == "system"]

    custom = KeepOnlySystem()
    assert isinstance(custom, MessageCompactor)  # runtime_checkable

    msgs = [
        system_msg("sys"),
        user_msg("q"),
        assistant_msg("a"),
    ]
    out = custom.compact(msgs, keep_recent=1)
    assert out == [msgs[0]]


# ── Compaction policy ──────────────────────────────────────────────────


def test_default_policy_turn_threshold():
    policy = DefaultCompactionPolicy(
        compact_after_turns=10,
        context_token_limit=100_000,
    )
    assert policy.should_compact(turn=5, messages=[], estimated_tokens=0) is False
    assert policy.should_compact(turn=11, messages=[], estimated_tokens=0) is True


def test_default_policy_token_threshold():
    policy = DefaultCompactionPolicy(
        compact_after_turns=10,
        context_token_limit=5_000,
    )
    assert policy.should_compact(turn=1, messages=[], estimated_tokens=4_000) is False
    assert policy.should_compact(turn=1, messages=[], estimated_tokens=5_001) is True


def test_custom_policy_is_protocol_compliant():
    """A workflow-owned policy just needs ``should_compact(...)`` — no base class."""

    class AlwaysCompact:
        def should_compact(self, turn, messages, estimated_tokens):
            return True

    class NeverCompact:
        def should_compact(self, turn, messages, estimated_tokens):
            return False

    assert isinstance(AlwaysCompact(), CompactionPolicy)
    assert isinstance(NeverCompact(), CompactionPolicy)

    assert AlwaysCompact().should_compact(0, [], 0) is True
    assert NeverCompact().should_compact(100, [], 10_000_000) is False
