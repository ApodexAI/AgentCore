"""Product-neutral identity and task lifecycle types."""

from __future__ import annotations

from enum import StrEnum
from typing import NewType
from uuid import uuid4

TaskId = NewType("TaskId", str)
EventId = NewType("EventId", str)
SessionId = NewType("SessionId", str)
AgentSessionId = NewType("AgentSessionId", str)
PromptId = NewType("PromptId", str)
StepId = NewType("StepId", str)
AgentRoleId = NewType("AgentRoleId", str)


def new_task_id() -> TaskId:
    return TaskId(uuid4().hex[:12])


def new_event_id() -> EventId:
    raw = uuid4().hex
    # Keep the historical 16-character lowercase-hex shape while making the
    # opaque-id namespace disjoint from decimal store ordinals. UUID4's
    # version nibble at index 12 is fixed to ``4``; replacing that non-random
    # nibble with an alphabetic marker preserves the original 60 random bits
    # and guarantees ``event_ordinal()`` cannot mistake this id for a cursor.
    return EventId(f"e{raw[:12]}{raw[13:16]}")


def new_session_id() -> SessionId:
    return SessionId(uuid4().hex[:12])


def new_prompt_id() -> PromptId:
    return PromptId(uuid4().hex[:12])


def new_step_id() -> StepId:
    return StepId(uuid4().hex[:10])


class TaskStatus(StrEnum):
    """Portable task lifecycle states."""

    CREATED = "created"
    RUNNING = "running"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


__all__ = [
    "AgentRoleId",
    "AgentSessionId",
    "EventId",
    "PromptId",
    "SessionId",
    "StepId",
    "TaskId",
    "TaskStatus",
    "new_event_id",
    "new_prompt_id",
    "new_session_id",
    "new_step_id",
    "new_task_id",
]
