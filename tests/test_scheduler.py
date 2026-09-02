from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from agent_core.scheduling.scheduler import (
    Scheduler,
    TaskWallTimeExceeded,
    resolve_wall_time_s,
)
from agent_core.scheduling.workflow_defaults import (
    clear_workflow_defaults,
    register_workflow_defaults,
)
from agent_core.types import TaskId, TaskStatus


@dataclass
class _Task:
    status: TaskStatus = TaskStatus.CREATED
    thread_id: str = "thread"
    pipeline_id: str | None = None
    context: dict[str, Any] = field(default_factory=dict)


class _ProcessManager:
    def __init__(self) -> None:
        self.task = _Task()
        self.error: str | None = None
        self.abort_reason: str | None = None

    async def get_task(self, task_id: TaskId) -> _Task:
        del task_id
        return self.task

    async def update_status(self, task_id: TaskId, status: TaskStatus) -> None:
        del task_id
        self.task.status = status

    async def abort_task(self, task_id: TaskId, reason: str) -> None:
        del task_id
        self.abort_reason = reason
        self.task.status = TaskStatus.ABORTED

    async def set_error(self, task_id: TaskId, error: str) -> None:
        del task_id
        self.error = error
        self.task.status = TaskStatus.FAILED


class _EventSink:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def append(self, **event: Any) -> None:
        self.events.append(event)


class _ReportGraph:
    async def astream(self, *_args: Any, **_kwargs: Any):
        yield {"report": {"report": "done"}}

    async def aget_state(self, _config: dict[str, Any]):
        return SimpleNamespace(values={"report": "done"})


class _EmptyGraph:
    async def astream(self, *_args: Any, **_kwargs: Any):
        yield {"research": {"current_phase": "research"}}

    async def aget_state(self, _config: dict[str, Any]):
        return SimpleNamespace(values={"current_phase": "research"})


class _HangingGraph:
    async def astream(self, *_args: Any, **_kwargs: Any):
        yield {"research": {"current_phase": "research"}}
        await asyncio.sleep(60)


def test_wall_time_resolution_supports_portable_and_legacy_env(monkeypatch):
    monkeypatch.setenv("MIROHARNESS_TASK_WALL_TIME_S", "120")
    assert resolve_wall_time_s(None) == 120
    monkeypatch.setenv("AGENT_CORE_TASK_WALL_TIME_S", "30")
    assert resolve_wall_time_s(None) == 30
    assert resolve_wall_time_s(10) == 10
    assert resolve_wall_time_s(0) is None


def test_workflow_defaults_are_product_injected(monkeypatch):
    for name in (
        "AGENT_CORE_TASK_WALL_TIME_S",
        "MIROHARNESS_TASK_WALL_TIME_S",
        "FRONTIER_AGENT_TASK_WALL_TIME_S",
    ):
        monkeypatch.delenv(name, raising=False)
    clear_workflow_defaults()
    register_workflow_defaults(
        ("research", "research-report"),
        {"TASK_WALL_TIME_S": 900, "TASK_WALL_TIME_MODE": "soft_research"},
    )
    assert resolve_wall_time_s(None, "research") == 900
    clear_workflow_defaults()


@pytest.mark.asyncio
async def test_scheduler_completes_only_with_terminal_output():
    pm = _ProcessManager()
    sink = _EventSink()
    scheduler = Scheduler(_ReportGraph(), pm, sink)
    chunks = [chunk async for chunk in scheduler.execute(TaskId("t"), {})]
    assert chunks == [("updates", {"report": {"report": "done"}})]
    assert pm.task.status == TaskStatus.COMPLETED
    assert len(sink.events) == 1


@pytest.mark.asyncio
async def test_scheduler_records_missing_terminal_output():
    pm = _ProcessManager()
    scheduler = Scheduler(_EmptyGraph(), pm, _EventSink())
    with pytest.raises(RuntimeError, match="produced no final answer"):
        async for _ in scheduler.execute(TaskId("t"), {}):
            pass
    assert pm.task.status == TaskStatus.FAILED
    assert pm.error is not None


@pytest.mark.asyncio
async def test_scheduler_aborts_a_hung_graph():
    pm = _ProcessManager()
    scheduler = Scheduler(_HangingGraph(), pm, _EventSink())
    with pytest.raises(TaskWallTimeExceeded):
        async for _ in scheduler.execute(TaskId("t"), {}, wall_time_s=1):
            pass
    assert pm.task.status == TaskStatus.ABORTED
    assert pm.abort_reason is not None
