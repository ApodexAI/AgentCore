"""Compaction must see and shed ``reasoning_content`` (DeepSeek-V4 round-trip)."""

from __future__ import annotations

from agent_core.components.middleware.llm import compaction as mw
from agent_core.messages import assistant_msg_with_reasoning
from agent_core.runtime.loop.compact import (
    KeepLastNToolResultsCompactor,
    compress_tool_results,
    estimate_tokens,
    strip_stale_reasoning,
)


def _history(turns: int, reasoning_chars: int = 20_000) -> list[dict]:
    msgs: list[dict] = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    for i in range(turns):
        msgs.append(assistant_msg_with_reasoning(
            "", f"r{i} " * (reasoning_chars // 4),
            tool_calls=[{"id": f"c{i}", "type": "function",
                         "function": {"name": "bash", "arguments": "{}"}}],
            thinking_format="reasoning_content",
        ))
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "out"})
    return msgs


def _with_reasoning(msgs: list[dict]) -> list[int]:
    return [i for i, m in enumerate(msgs) if m.get("reasoning_content")]


def test_estimators_count_reasoning_content() -> None:
    msgs = _history(4)
    bare = [{k: v for k, v in m.items() if k != "reasoning_content"} for m in msgs]
    assert estimate_tokens(msgs) > 10 * estimate_tokens(bare)
    assert mw.estimate_tokens(msgs) > 10 * mw.estimate_tokens(bare)


def test_strip_keeps_latest_and_does_not_mutate() -> None:
    msgs = _history(5)
    out = strip_stale_reasoning(msgs, keep_last=2)
    assert _with_reasoning(out) == _with_reasoning(msgs)[-2:]
    assert len(_with_reasoning(msgs)) == 5
    assert strip_stale_reasoning(msgs, keep_last=-1) is msgs
    assert _with_reasoning(strip_stale_reasoning(msgs, keep_last=0)) == []


def test_tier1_sheds_reasoning_outside_its_window() -> None:
    msgs = _history(12)
    out = KeepLastNToolResultsCompactor(keep_tool_result=3).compact(msgs, 5)
    assert len(_with_reasoning(out)) == 3
    assert estimate_tokens(out) < estimate_tokens(msgs) / 3


def test_tier1_disabled_keeps_reasoning() -> None:
    msgs = _history(4)
    assert KeepLastNToolResultsCompactor(keep_tool_result=-1).compact(msgs, 5) is msgs


def test_compress_tool_results_keeps_only_latest_reasoning() -> None:
    msgs = _history(6)
    out = compress_tool_results(msgs)
    assert _with_reasoning(out) == _with_reasoning(msgs)[-1:]
    assert out[-2]["tool_calls"] == msgs[-2]["tool_calls"]
