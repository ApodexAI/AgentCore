"""The task-board reminder must actually reach the loop."""

from __future__ import annotations

import asyncio

from agent_core.components.observers.task_board import TaskBoardObserver
from agent_core.loop_types import Intervention, TurnContext, notify_observers


def _observer() -> TaskBoardObserver:
    return TaskBoardObserver(
        board_size=lambda _task_id: 1,
        render_board=lambda _task_id, _bus_task_id: "- [ ] write the report",
        resolve_bus_task_id=lambda _scope: None,
        cooldown_turns=1,
    )


def test_board_reminder_is_collected_as_an_intervention() -> None:
    ctx = TurnContext(
        turn=1, max_turns=10, task_id="t1", role_id="coordinator", ai_text="",
        thinking="", tool_calls=[], messages=[], usage=None, metadata={},
    )

    async def run() -> list[Intervention]:
        return await notify_observers([_observer()], "on_turn_end", ctx)

    interventions = asyncio.run(run())

    assert len(interventions) == 1
    assert "write the report" in interventions[0].inject_messages[0]
