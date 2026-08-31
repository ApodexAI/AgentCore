from __future__ import annotations

from agent_core.messages import assistant_msg, tool_msg, user_msg
from agent_core.runtime.loop.message_trimmer import (
    NullTrimmer,
    TaskBoundaryTrimmer,
    find_final_assistant,
    trim_and_remap_boundaries,
)


def _history() -> list[dict]:
    return [
        user_msg("task one"),
        assistant_msg(
            "",
            tool_calls=[
                {
                    "type": "function",
                    "id": "call-1",
                    "function": {"name": "search", "arguments": "{}"},
                }
            ],
        ),
        tool_msg("result", "call-1"),
        assistant_msg("report one"),
        user_msg("task two"),
        assistant_msg("working"),
    ]


def test_null_trimmer_copies_without_changing_content() -> None:
    messages = _history()
    trimmed = NullTrimmer().trim(messages, [(0, 3), (4, None)])
    assert trimmed == messages
    assert trimmed is not messages


def test_boundary_trimmer_keeps_completed_report_and_open_task() -> None:
    trimmed = TaskBoundaryTrimmer().trim(_history(), [(0, 3), (4, None)])
    assert trimmed == [
        user_msg("task one"),
        assistant_msg("report one"),
        user_msg("task two"),
        assistant_msg("working"),
    ]


def test_trim_and_remap_boundaries_is_atomic() -> None:
    messages, boundaries = trim_and_remap_boundaries(
        _history(),
        [(0, 3), (4, None)],
    )
    assert boundaries == [(0, 1), (2, None)]
    assert messages[1] == assistant_msg("report one")
    assert messages[2] == user_msg("task two")


def test_find_final_assistant_ignores_tool_call_stub() -> None:
    messages = _history()
    assert find_final_assistant(messages, 1, 3) == assistant_msg("report one")
