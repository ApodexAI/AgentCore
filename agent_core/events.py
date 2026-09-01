"""Product-neutral kernel event identifiers."""

from __future__ import annotations

from enum import StrEnum


class EventType(StrEnum):
    """Framework-mechanical events shared by all products."""

    TASK_CREATED = "task_created"
    TASK_STATUS_CHANGED = "task_status_changed"
    PHASE_TRANSITION = "phase_transition"
    AGENT_ACTION = "agent_action"
    AGENT_MESSAGE = "agent_message"
    AGENT_TOOL_CALL = "agent_tool_call"
    TOOL_CALLED = "tool_called"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    REPORT_GENERATED = "report_generated"
    WORKING_MEMORY_SNAPSHOT = "working_memory_snapshot"


__all__ = ["EventType"]
