from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace

import pytest

from agent_core.components.agent_bus import (
    AgentBus,
    JobEntry,
    SubAgentSession,
    SubTask,
)
from agent_core.loop_types import AgentLoopResult
from agent_core.messages import assistant_msg, system_msg, user_msg
from agent_core.runtime.loop.message_trimmer import NullTrimmer


def _session(task_id: str = "root") -> SubAgentSession:
    return SubAgentSession(
        session_id=f"{task_id}::worker",
        task_id=task_id,
        name="worker",
        role_id="researcher",
        system_prompt="test",
        tools=[],
        llm=None,
        trimmer=NullTrimmer(),
    )
async def test_session_spawn_registers_job_before_eager_task_starts(
    monkeypatch,
) -> None:
    """The TUI's eager task factory must not outrun AgentBus bookkeeping."""
    async def fake_run_agent_loop(**kwargs):
        ctx = SimpleNamespace(
            turn=1,
            thinking="I should inspect the ClinVar source before generating data.",
            ai_text="I will query the source and validate the response.",
        )
        for observer in kwargs["observers"]:
            await observer.on_llm_delta(SimpleNamespace(
                turn=1, thinking_delta="I should inspect ", delta="",
            ))
            await observer.on_llm_delta(SimpleNamespace(
                turn=1, thinking_delta="the ClinVar source first.", delta="",
            ))
            await observer.on_llm_response(ctx)
            await observer.on_tool_call(ctx, {
                "name": "web_search", "args": {"query": "ClinVar API"},
            })
            await observer.on_tool_result(ctx, SimpleNamespace(
                name="web_search", result="found source", is_error=False,
            ))
        return AgentLoopResult(
            messages=[
                system_msg(kwargs["system_prompt"]),
                user_msg(kwargs["user_message"]),
                assistant_msg("finished"),
            ],
            final_content="finished",
            stopped_by="final_answer",
        )

    monkeypatch.setattr(
        "agent_core.components.agent_bus.bus.run_agent_loop",
        fake_run_agent_loop,
    )
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    loop.set_task_factory(asyncio.eager_task_factory)
    try:
        bus = AgentBus()
        session_id = await bus.create_session(
            task_id="root",
            name="worker",
            role_id="researcher",
            system_prompt="test",
            tools_override=[],
            llm_override=object(),
        )
        job_id = await bus.submit_task_to_session(session_id, "do work")
        outcome = await bus.wait_any_session_detailed("root", timeout=0)
    finally:
        loop.set_task_factory(previous_factory)

    assert outcome.reason == "ready"
    assert outcome.result is not None
    _, result = outcome.result
    assert result.success
    assert result.final_content == "finished"
    assert bus._jobs[job_id].status == "completed"
    assert bus._sessions[session_id].current_job_id is None
    snapshot = bus.describe_sessions_for_task("root")[0]
    assert [event["kind"] for event in snapshot["events"]] == [
        "thinking", "message", "tool_call", "tool_result",
    ]
    assert snapshot["events"][0]["detail"].startswith("I should inspect")
    assert len([e for e in snapshot["events"] if e["kind"] == "thinking"]) == 1


@pytest.mark.asyncio
async def test_wait_reconciles_cancelled_task_instead_of_fake_timeout() -> None:
    bus = AgentBus()
    session = _session()
    job_id = "root::worker::1"
    task = asyncio.create_task(asyncio.sleep(60))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    session.current_job_id = job_id
    session.total_task_count = 1
    bus._sessions[session.session_id] = session
    bus._jobs[job_id] = JobEntry(
        job_id=job_id,
        parent_task_id="root",
        item=SubTask(question="work", role_id="researcher"),
        task=task,
        status="running",
    )

    outcome = await bus.wait_any_session_detailed("root", timeout=1800)

    assert outcome.reason == "ready"
    assert outcome.elapsed_s < 1
    assert outcome.result is not None
    _, result = outcome.result
    assert not result.success
    assert result.error_class == "CancelledError"
    assert session.current_job_id is None
    assert bus._jobs[job_id].status == "aborted"


@pytest.mark.asyncio
async def test_wait_reports_actual_timeout_only_for_live_task() -> None:
    bus = AgentBus()
    session = _session()
    job_id = "root::worker::1"
    task = asyncio.create_task(asyncio.sleep(60))
    session.current_job_id = job_id
    session.total_task_count = 1
    bus._sessions[session.session_id] = session
    bus._jobs[job_id] = JobEntry(
        job_id=job_id,
        parent_task_id="root",
        item=SubTask(question="work", role_id="researcher"),
        task=task,
        status="running",
    )

    try:
        outcome = await bus.wait_any_session_detailed("root", timeout=0.02)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    assert outcome.reason == "timeout"
    assert outcome.result is None
    assert 0.01 <= outcome.elapsed_s < 1
    assert session.current_job_id == job_id


@pytest.mark.asyncio
async def test_wait_reports_unpublished_when_a_task_ends_empty_handed() -> None:
    """A task that finished during the wait is not "nothing to wait for".

    Real time elapsed, so telling the coordinator to discount the wait (what
    ``no_pending`` means) would be the same wrong-elapsed-time steering in the
    opposite direction.
    """
    bus = AgentBus()
    session = _session()
    job_id = "root::worker::1"

    async def _finish_without_publishing() -> None:
        await asyncio.sleep(0.02)
        # Mimic a wrapper that completes but leaves the session's bookkeeping
        # untouched, so reconciliation has nothing to hand back either.
        bus._jobs[job_id].status = "completed"
        session.current_job_id = None

    task = asyncio.create_task(_finish_without_publishing())
    session.current_job_id = job_id
    session.total_task_count = 1
    bus._sessions[session.session_id] = session
    bus._jobs[job_id] = JobEntry(
        job_id=job_id,
        parent_task_id="root",
        item=SubTask(question="work", role_id="researcher"),
        task=task,  # type: ignore[arg-type]
        status="running",
    )

    outcome = await bus.wait_any_session_detailed("root", timeout=5)

    assert outcome.reason == "unpublished"
    assert outcome.result is None
    assert outcome.elapsed_s >= 0.02


@pytest.mark.asyncio
async def test_session_snapshot_freezes_a_finished_worker_duration() -> None:
    """The UI reads ``elapsed_s`` verbatim, so it must stop growing.

    ``active`` is what tells a renderer whether the number is still counting;
    without it a finished worker's row ticks up for as long as the fan-in
    blocks.
    """
    async def fake_run_agent_loop(**kwargs):
        return AgentLoopResult(
            messages=[
                system_msg(kwargs["system_prompt"]),
                user_msg(kwargs["user_message"]),
                assistant_msg("done"),
            ],
            final_content="done",
            stopped_by="final_answer",
        )

    bus = AgentBus()
    session_id = await bus.create_session(
        task_id="root",
        name="market_research",
        role_id="agent_team_sub",
        system_prompt="test",
        tools_override=[],
        llm_override=object(),
    )

    unassigned = bus.describe_sessions_for_task("root")[0]
    assert unassigned["status"] == "unassigned"
    assert unassigned["active"] is False
    assert unassigned["elapsed_s"] == 0.0
    assert unassigned["role_id"] == "agent_team_sub"

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            "agent_core.components.agent_bus.bus.run_agent_loop",
            fake_run_agent_loop,
        )
        await bus.submit_task_to_session(session_id, "do work")
        assert await bus.wait_any_session("root", timeout=5) is not None

    finished = bus.describe_sessions_for_task("root")[0]
    assert finished["status"] == "idle"
    assert finished["active"] is False
    frozen = finished["elapsed_s"]

    await asyncio.sleep(0.05)
    assert bus.describe_sessions_for_task("root")[0]["elapsed_s"] == frozen


@pytest.mark.asyncio
async def test_wait_distinguishes_no_pending_work_from_timeout() -> None:
    bus = AgentBus()
    session = _session()
    bus._sessions[session.session_id] = session

    outcome = await bus.wait_any_session_detailed("root", timeout=1800)

    assert outcome.reason == "no_pending"
    assert outcome.result is None
    assert outcome.elapsed_s < 1
