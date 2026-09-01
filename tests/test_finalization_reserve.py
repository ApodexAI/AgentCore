"""Early finalization reserve observer regression tests."""

from __future__ import annotations

import pytest

from agent_core.components.observers.finalization_reserve import (
    FinalizationReserveObserver,
)
from agent_core.loop_types import TurnContext


def _ctx(
    turn: int,
    metadata: dict | None = None,
    *,
    max_turns: int = 10,
) -> TurnContext:
    return TurnContext(
        turn=turn,
        max_turns=max_turns,
        task_id="task",
        role_id="agent",
        ai_text="",
        thinking="",
        tool_calls=[],
        messages=[],
        usage=None,
        metadata=metadata if metadata is not None else {},
    )


@pytest.mark.asyncio
async def test_reserve_warns_once_while_tool_turns_remain() -> None:
    observer = FinalizationReserveObserver(
        reserve_turns=3,
        message="finish deliverables",
    )
    metadata: dict = {}

    assert await observer.on_turn_end(_ctx(6, metadata)) is None
    intervention = await observer.on_turn_end(_ctx(7, metadata))
    assert intervention is not None
    assert intervention.inject_messages == ["finish deliverables"]
    assert metadata["finalization_phase"] is True
    assert metadata["finalization_reserve_turns"] == 3
    assert await observer.on_turn_end(_ctx(8, metadata)) is None


@pytest.mark.asyncio
async def test_three_turn_smoke_loop_relies_on_last_turn_forcer() -> None:
    observer = FinalizationReserveObserver(
        reserve_turns=8,
        message="finish deliverables",
    )
    metadata: dict = {}

    for turn in (1, 2, 3):
        assert (
            await observer.on_turn_end(
                _ctx(turn, metadata, max_turns=3),
            )
            is None
        )

    assert "finalization_phase" not in metadata
