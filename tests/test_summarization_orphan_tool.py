"""The kept window after summarization must never start on a tool message.

An orphan tool result — one whose assistant ``tool_calls`` was compressed into
the summary — makes Azure reject the request outright:
``400 No tool call found for function call output with call_id ...``. The
middleware already advanced the split point past leading tool messages, but the
loop bound stopped one element short, so a tail made entirely of tool results
still left exactly one orphan behind.
"""

from __future__ import annotations

import pytest

from agent_core.components.middleware.llm.base import LLMCallContext
from agent_core.components.middleware.llm.summarization import (
    SummarizationMiddleware,
)


def _ctx() -> LLMCallContext:
    return LLMCallContext(task_id="t1", role_id="r", phase_id="p", call_index=1)


def _sys(text: str = "sys") -> dict:
    return {"role": "system", "content": text}


def _tool(text: str) -> dict:
    return {"role": "tool", "content": text, "tool_call_id": text}


def _assistant_with_calls(text: str = "calling") -> dict:
    return {
        "role": "assistant", "content": text,
        "tool_calls": [{"id": "c1", "function": {"name": "web_search"}}],
    }


def _big(text: str, *, repeat: int = 4000) -> str:
    return text * repeat


async def _compress(messages: list[dict], *, keep_recent: int) -> list[dict]:
    mw = SummarizationMiddleware(threshold=10, keep_recent=keep_recent)
    return await mw.before_llm(_ctx(), messages)


def _assert_no_orphan_tool(result: list[dict]) -> None:
    """No tool message may appear before the assistant that requested it."""
    seen_tool_calls = False
    for msg in result:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            seen_tool_calls = True
        if msg.get("role") == "tool":
            assert seen_tool_calls, f"orphan tool message in {[m['role'] for m in result]}"


@pytest.mark.asyncio
async def test_all_tool_tail_leaves_no_orphan():
    """keep_recent=3 over [assistant(tool_calls), tool, tool, tool]."""
    messages = [
        _sys(),
        {"role": "user", "content": _big("question ")},
        _assistant_with_calls(),
        _tool(_big("r1 ")),
        _tool(_big("r2 ")),
        _tool(_big("r3 ")),
    ]

    result = await _compress(messages, keep_recent=3)

    assert len(result) < len(messages), "should have compressed"
    _assert_no_orphan_tool(result)
    assert result[-1].get("role") != "tool"


@pytest.mark.asyncio
@pytest.mark.parametrize("keep_recent", [1, 2, 3, 4, 5])
async def test_no_orphan_for_any_keep_recent_over_a_tool_tail(keep_recent):
    messages = [
        _sys(),
        {"role": "user", "content": _big("q ")},
        _assistant_with_calls(),
        _tool(_big("r1 ")),
        _tool(_big("r2 ")),
        _tool(_big("r3 ")),
        _tool(_big("r4 ")),
    ]

    result = await _compress(messages, keep_recent=keep_recent)

    _assert_no_orphan_tool(result)


@pytest.mark.asyncio
async def test_split_still_keeps_a_healthy_tail_intact():
    """Guard against over-advancing: a non-tool tail must be kept as-is."""
    messages = [
        _sys(),
        {"role": "user", "content": _big("q ")},
        _assistant_with_calls(),
        _tool(_big("r1 ")),
        {"role": "assistant", "content": "answer A"},
        {"role": "user", "content": "follow up"},
        {"role": "assistant", "content": "answer B"},
    ]

    result = await _compress(messages, keep_recent=3)

    assert [m["content"] for m in result[-3:]] == [
        "answer A", "follow up", "answer B",
    ]
    _assert_no_orphan_tool(result)


@pytest.mark.asyncio
async def test_below_threshold_is_untouched():
    messages = [_sys(), {"role": "user", "content": "hi"}, _tool("r")]
    mw = SummarizationMiddleware(threshold=10_000, keep_recent=2)

    assert await mw.before_llm(_ctx(), messages) is messages
