"""A turn that ran without tools must never be replayed with tools.

``LastTurnForcer`` strips tools for the landing turn by planting
``_llm_strip_tools``, which ``_prepare_llm_request`` consumes one-shot. On that
turn nothing can be parsed, so ``ctx.tool_calls`` and ``ctx.blocked_tool_calls``
are both empty — while a prose answer that merely mentions ``finalize_answer``
still trips ``_looks_like_leak``. The observer then asked for
``continue_to_next_turn``, and because the strip flag was already gone the
replayed turn was bound *with* tools: the one turn the policy guaranteed would
have none.
"""

from __future__ import annotations

import pytest

from agent_core.components.observers.last_turn_forcer import LastTurnForcer
from agent_core.components.observers.leaked_tool_call_retry import (
    LeakedToolCallRetryObserver,
)
from agent_core.loop_types import TurnContext

TOOLS = ("finalize_answer", "web_search")
LEAK_TEXT = "I will call finalize_answer with the complete answer now."


def _ctx(text: str, *, metadata: dict | None = None, turn: int = 3) -> TurnContext:
    return TurnContext(
        turn=turn,
        max_turns=4,
        task_id="t1",
        role_id="solver",
        ai_text=text,
        thinking="",
        tool_calls=[],
        messages=[],
        usage=None,
        metadata=metadata if metadata is not None else {},
    )


def _observer() -> LeakedToolCallRetryObserver:
    return LeakedToolCallRetryObserver(TOOLS)


@pytest.mark.asyncio
async def test_a_genuine_leak_on_a_normal_turn_still_retries():
    """The observer's whole reason to exist must keep working."""
    ctx = _ctx(LEAK_TEXT, metadata={"_llm_tools_stripped": False})

    intervention = await _observer().on_llm_response(ctx)

    assert intervention is not None
    assert intervention.continue_to_next_turn is True
    assert ctx.metadata["_llm_temp_override"] == 0.3


@pytest.mark.asyncio
async def test_no_replay_when_the_turn_ran_without_tools():
    ctx = _ctx(LEAK_TEXT, metadata={"_llm_tools_stripped": True})

    assert await _observer().on_llm_response(ctx) is None
    assert "_llm_temp_override" not in ctx.metadata
    assert "_leak_retry_count" not in ctx.metadata


@pytest.mark.asyncio
async def test_a_pending_retry_is_abandoned_on_the_stripped_turn():
    """State carried in from an earlier nudge must not resurrect the replay."""
    ctx = _ctx(LEAK_TEXT, metadata={
        "_llm_tools_stripped": True,
        "_leak_retry_count": 1,
        "_llm_temp_override": 0.6,
    })

    assert await _observer().on_llm_response(ctx) is None
    assert "_llm_temp_override" not in ctx.metadata
    assert "_leak_retry_count" not in ctx.metadata


@pytest.mark.asyncio
async def test_blocked_tool_calls_guard_is_unaffected():
    ctx = _ctx(LEAK_TEXT, metadata={"_llm_tools_stripped": False})
    ctx.blocked_tool_calls = [{"name": "web_search"}]

    assert await _observer().on_llm_response(ctx) is None


@pytest.mark.asyncio
async def test_end_to_end_with_last_turn_forcer():
    """The two observers must agree about the landing turn.

    LastTurnForcer plants the flag at the end of the penultimate turn;
    ``_prepare_llm_request`` converts it into the per-turn marker; the retry
    observer then declines to replay.
    """
    from agent_core.loop_types import LoopConfig
    from agent_core.runtime.loop.agent_loop import _prepare_llm_request

    metadata: dict = {}
    forcer = LastTurnForcer(terminal_tool="finalize_answer")
    penultimate = _ctx("...", metadata=metadata, turn=3)
    assert await forcer.on_turn_end(penultimate) is not None
    assert metadata["_llm_strip_tools"] is True

    with_tools = object()
    without_tools = object()
    llm_for_turn, _messages, _stripped, _ = await _prepare_llm_request(
        LoopConfig(task_id="t1", role_id="solver", max_turns=4),
        [], without_tools, with_tools, [], metadata, 4,
    )

    assert llm_for_turn is without_tools, "landing turn must run without tools"
    assert metadata["_llm_tools_stripped"] is True
    assert "_llm_strip_tools" not in metadata

    landing = _ctx(LEAK_TEXT, metadata=metadata, turn=4)
    assert await _observer().on_llm_response(landing) is None
