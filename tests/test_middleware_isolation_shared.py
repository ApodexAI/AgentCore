"""Regression coverage for shared middleware state and host-owned summaries."""

from __future__ import annotations

from typing import Any

import pytest

from agent_core.components.memory import WorkingMemory, current_working_memory
from agent_core.components.middleware.llm.base import LLMCallContext
from agent_core.components.middleware.llm.loop_detection import LoopDetectionMiddleware
from agent_core.components.middleware.status_report import StatusReportMiddleware
from agent_core.components.middleware.todo import TodoMiddleware
from agent_core.llm import LLMResponse
from agent_core.messages import text_of, user_msg
from agent_core.protocols import PhaseContext
from agent_core.runtime.registries import services


def _tool_response() -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=[
            {
                "function": {
                    "name": "web_search",
                    "arguments": '{"q":"same"}',
                }
            }
        ],
    )


@pytest.mark.asyncio
async def test_loop_detection_does_not_mix_tasks() -> None:
    middleware = LoopDetectionMiddleware(trigger_count=3)
    response = _tool_response()

    for task_id in ("task-a", "task-b", "task-c"):
        await middleware.after_llm(LLMCallContext(task_id=task_id), response)

    messages = [user_msg("continue")]
    result = await middleware.before_llm(LLMCallContext(task_id="task-d"), messages)

    assert result == messages


@pytest.mark.asyncio
async def test_loop_detection_still_triggers_within_one_scope() -> None:
    middleware = LoopDetectionMiddleware(trigger_count=3)
    ctx = LLMCallContext(task_id="task-a", role_id="solver", phase_id="solve")

    for _ in range(3):
        await middleware.after_llm(ctx, _tool_response())

    result = await middleware.before_llm(ctx, [user_msg("continue")])

    assert result[0]["role"] == "system"
    assert "loop detected" in text_of(result[0].get("content")).lower()


@pytest.mark.asyncio
async def test_anonymous_calls_do_not_share_loop_history() -> None:
    middleware = LoopDetectionMiddleware(trigger_count=2)
    for _ in range(2):
        await middleware.after_llm(LLMCallContext(), _tool_response())

    messages = [user_msg("continue")]
    assert await middleware.before_llm(LLMCallContext(), messages) == messages


@pytest.mark.asyncio
async def test_todo_uses_polymorphic_working_memory_summary() -> None:
    class ArtifactMemory(WorkingMemory):
        def one_line_summary(self) -> str:
            return "7 artifacts ready"

    memory = ArtifactMemory(task_id="task-a")
    token = current_working_memory.set(memory)
    try:
        result = await TodoMiddleware().before_llm(
            LLMCallContext(task_id="task-a", metadata={"turn": 3}),
            [user_msg("continue")],
        )
    finally:
        current_working_memory.reset(token)

    assert "7 artifacts ready" in "\n".join(text_of(msg.get("content")) for msg in result)


class _RecordingComm:
    def __init__(self) -> None:
        self.messages: list[Any] = []

    async def send(self, message: Any, *, mode: Any) -> None:
        self.messages.append(message)


@pytest.mark.asyncio
async def test_status_report_has_no_research_schema_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comm = _RecordingComm()
    monkeypatch.setattr(services, "get_optional", lambda service_type: comm)
    middleware = StatusReportMiddleware()
    ctx = PhaseContext(
        task_id="task-a",
        phase_id="solve",
        role_id="worker",
        metadata={"parent_agent_id": "parent"},
    )

    await middleware.after_phase(
        ctx,
        {"evidence_cards": [1, 2], "assertions": [1]},
    )

    content = comm.messages[0].content
    assert "evidence_count" not in content
    assert "assertion_count" not in content


@pytest.mark.asyncio
async def test_status_report_accepts_host_owned_result_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comm = _RecordingComm()
    monkeypatch.setattr(services, "get_optional", lambda service_type: comm)
    middleware = StatusReportMiddleware(
        result_summarizer=lambda result: {
            "artifact_count": len(result.get("artifacts", [])),
            "status": "spoofed",
        }
    )
    ctx = PhaseContext(
        task_id="task-a",
        phase_id="solve",
        role_id="worker",
        metadata={"parent_agent_id": "parent"},
    )

    await middleware.after_phase(ctx, {"artifacts": [1, 2, 3]})

    content = comm.messages[0].content
    assert content["artifact_count"] == 3
    assert content["status"] == "phase_completed"
