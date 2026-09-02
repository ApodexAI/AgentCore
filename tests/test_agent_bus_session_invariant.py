"""Tests for AgentBus session invariants:

* ``force_finalizer`` mutates raw_result before bus absorbs into session.
* Closed boundary always has a clean final assistant message.
* Aborted boundary also has a clean final assistant message.
* ``observers_builder`` receives ``task_index``.
* ``session_task_submitted`` SSE payload includes ``is_reuse`` + ``role_id``.
"""

from __future__ import annotations

import contextlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_core.components.agent_bus import (
    AgentBus,
    SubAgentRuntimeSpec,
)
from agent_core.loop_types import AgentLoopResult
from agent_core.messages import (
    assistant_msg,
    is_assistant_msg,
    text_of,
    user_msg,
)
from agent_core.runtime.loop.message_trimmer import (
    NullTrimmer,
    find_final_assistant,
)
from agent_core.runtime.registries import services as registry


def _tool_call(call_id: str, name: str, arguments: str = "{}") -> dict:
    """Build an OpenAI-wire tool_call dict."""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


class _FakeEventStore:
    def __init__(self) -> None:
        self.append = AsyncMock()
        self.get_events = AsyncMock(return_value=[])


@pytest.fixture(autouse=True)
def clear_registry():
    registry.clear()
    yield
    registry.clear()


async def _make_bus_with_session(
    *,
    name: str = "agent_a",
    task_id: str = "task-1",
    role_id: str = "researcher",
    system_prompt: str = "You are a researcher.",
    trimmer=None,
    runtime_spec: SubAgentRuntimeSpec | None = None,
):
    bus = AgentBus()
    sid = await bus.create_session(
        task_id=task_id,
        name=name,
        role_id=role_id,
        system_prompt=system_prompt,
        llm_override=MagicMock(),
        tools_override=[],
        trimmer=trimmer,
        runtime_spec=runtime_spec,
    )
    return bus, sid


# ── force_finalizer ───────────────────────────────────────────────────────


class TestForceFinalizer:
    @pytest.mark.asyncio
    async def test_appends_to_session_messages_and_last_report(self, monkeypatch):
        """force_finalizer mutating raw_result.messages + final_content
        must land in session.messages + session.last_report."""

        async def fake_loop(**kwargs):
            im = list(kwargs.get("initial_messages") or [])
            um = kwargs.get("user_message", "")
            return AgentLoopResult(
                messages=[*im, user_msg(um), assistant_msg("thinking", tool_calls=[_tool_call("c1", "search")])],
                final_content="",  # no clean final from the loop
                stopped_by="max_turns",
                metadata={},
            )

        monkeypatch.setattr(
            "agent_core.components.agent_bus.bus.run_agent_loop", fake_loop,
        )

        async def force_finalizer(raw_result, _item):
            raw_result.messages.append(assistant_msg("FORCED"))
            raw_result.final_content = "FORCED"
            return raw_result

        spec = SubAgentRuntimeSpec(force_finalizer=force_finalizer)
        bus, sid = await _make_bus_with_session(runtime_spec=spec)

        await bus.submit_task_to_session(sid, "task 1")
        await bus.wait_any_session("task-1", timeout=2)

        session = bus.get_session(sid)
        assert session.last_report == "FORCED"
        # The forced assistant message must be the boundary's final message.
        start, end = session.task_boundaries[0]
        assert end is not None
        assert is_assistant_msg(session.messages[end])
        assert text_of(session.messages[end].get("content")) == "FORCED"
        # The trimmer's invariant helper finds it.
        ai = find_final_assistant(session.messages, start + 1, end)
        assert ai is not None and text_of(ai.get("content")) == "FORCED"


# ── Closed-boundary invariant ─────────────────────────────────────────────


class TestClosedBoundaryInvariant:
    @pytest.mark.asyncio
    async def test_synth_stub_when_loop_only_has_tool_calls(self, monkeypatch):
        """No clean final assistant message from the loop AND no
        force_finalizer → bus appends a stub assistant message so the
        boundary still has one."""

        async def fake_loop(**kwargs):
            im = list(kwargs.get("initial_messages") or [])
            um = kwargs.get("user_message", "")
            return AgentLoopResult(
                messages=[*im, user_msg(um), assistant_msg("", tool_calls=[_tool_call("c1", "search")])],
                final_content="",
                stopped_by="max_turns",
                metadata={},
            )

        monkeypatch.setattr(
            "agent_core.components.agent_bus.bus.run_agent_loop", fake_loop,
        )

        bus, sid = await _make_bus_with_session()
        await bus.submit_task_to_session(sid, "task 1")
        await bus.wait_any_session("task-1", timeout=2)

        session = bus.get_session(sid)
        start, end = session.task_boundaries[0]
        assert end is not None
        # Stub must exist at end, with the stopped_by reason mentioned.
        assert is_assistant_msg(session.messages[end])
        assert "max_turns" in text_of(session.messages[end].get("content"))
        # find_final_assistant locates it.
        assert find_final_assistant(session.messages, start + 1, end) is not None

    @pytest.mark.asyncio
    async def test_no_synth_stub_when_loop_has_clean_final(self, monkeypatch):
        """If the loop already produced a clean final assistant message, bus
        does NOT append a synthetic one."""

        async def fake_loop(**kwargs):
            im = list(kwargs.get("initial_messages") or [])
            um = kwargs.get("user_message", "")
            return AgentLoopResult(
                messages=[*im, user_msg(um), assistant_msg("answer here")],
                final_content="answer here",
                stopped_by="terminal_tool",
                metadata={},
            )

        monkeypatch.setattr(
            "agent_core.components.agent_bus.bus.run_agent_loop", fake_loop,
        )

        bus, sid = await _make_bus_with_session()
        await bus.submit_task_to_session(sid, "task 1")
        await bus.wait_any_session("task-1", timeout=2)

        session = bus.get_session(sid)
        # Boundary length is exactly 2 (user_msg + assistant_msg), no stub.
        start, end = session.task_boundaries[0]
        assert end - start == 1
        assert text_of(session.messages[end].get("content")) == "answer here"

    @pytest.mark.asyncio
    async def test_last_report_falls_back_to_previous_when_empty(self, monkeypatch):
        """When raw_result.final_content is empty and force_finalizer
        doesn't fix it, last_report keeps its previous value (so
        cross-agent <attach/> never goes blank mid-session)."""

        responses = iter([
            AgentLoopResult(
                messages=[user_msg("t1"), assistant_msg("A1")],
                final_content="A1",
                metadata={},
            ),
            AgentLoopResult(
                messages=[
                    user_msg("t2"),
                    assistant_msg("", tool_calls=[_tool_call("c1", "x")]),
                ],
                final_content="",
                stopped_by="max_turns",
                metadata={},
            ),
        ])

        async def fake_loop(**kwargs):
            base = next(responses)
            im = list(kwargs.get("initial_messages") or [])
            return AgentLoopResult(
                messages=im + base.messages,
                final_content=base.final_content,
                stopped_by=base.stopped_by,
                metadata=base.metadata,
            )

        monkeypatch.setattr(
            "agent_core.components.agent_bus.bus.run_agent_loop", fake_loop,
        )

        bus, sid = await _make_bus_with_session(trimmer=NullTrimmer())
        await bus.submit_task_to_session(sid, "t1")
        await bus.wait_any_session("task-1", timeout=2)
        assert bus.get_session(sid).last_report == "A1"

        await bus.submit_task_to_session(sid, "t2")
        await bus.wait_any_session("task-1", timeout=2)
        # Empty new final → keep "A1".
        assert bus.get_session(sid).last_report == "A1"

    @pytest.mark.asyncio
    async def test_last_report_falls_back_when_final_is_whitespace_only(
        self, monkeypatch,
    ):
        """Whitespace-only final_content must NOT clobber last_report."""

        responses = iter([
            AgentLoopResult(
                messages=[user_msg("t1"), assistant_msg("A1")],
                final_content="A1",
                metadata={},
            ),
            AgentLoopResult(
                messages=[
                    user_msg("t2"),
                    assistant_msg("  \n  "),
                ],
                final_content="  \n  ",  # whitespace only
                stopped_by="terminal_tool",
                metadata={},
            ),
        ])

        async def fake_loop(**kwargs):
            base = next(responses)
            im = list(kwargs.get("initial_messages") or [])
            return AgentLoopResult(
                messages=im + base.messages,
                final_content=base.final_content,
                stopped_by=base.stopped_by,
                metadata=base.metadata,
            )

        monkeypatch.setattr(
            "agent_core.components.agent_bus.bus.run_agent_loop", fake_loop,
        )

        bus, sid = await _make_bus_with_session(trimmer=NullTrimmer())
        await bus.submit_task_to_session(sid, "t1")
        await bus.wait_any_session("task-1", timeout=2)
        assert bus.get_session(sid).last_report == "A1"

        await bus.submit_task_to_session(sid, "t2")
        await bus.wait_any_session("task-1", timeout=2)
        # Whitespace-only new final → keep "A1", don't overwrite.
        assert bus.get_session(sid).last_report == "A1"


# ── Aborted boundary invariant ────────────────────────────────────────────


class TestAbortedBoundaryInvariant:
    @pytest.mark.asyncio
    async def test_cancelled_task_still_has_clean_final(self, monkeypatch):
        """When the loop coroutine is cancelled, the bus closes the
        boundary via close_session_boundary_aborted — that helper must
        also append an abort stub so trimmer doesn't lose the task."""

        import asyncio

        gate = asyncio.Event()
        cancel_token = asyncio.Event()

        async def gated_loop(**kwargs):
            cancel_token.set()
            await gate.wait()  # block forever; we'll cancel
            return AgentLoopResult(
                messages=[], final_content="", metadata={},
            )

        monkeypatch.setattr(
            "agent_core.components.agent_bus.bus.run_agent_loop", gated_loop,
        )

        bus, sid = await _make_bus_with_session()
        await bus.submit_task_to_session(sid, "task 1")
        await cancel_token.wait()

        # Cancel the in-flight task via the bus's tracked task object.
        session = bus.get_session(sid)
        job = bus._jobs[session.current_job_id].task
        job.cancel()
        # Drain the cancellation
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(job, timeout=2)

        # Boundary must be closed with at least one clean assistant message.
        start, end = session.task_boundaries[0]
        assert end is not None
        assert find_final_assistant(session.messages, start + 1, end) is not None
        # Stub mentions abort.
        assert any(
            is_assistant_msg(m)
            and "aborted" in text_of(m.get("content")).lower()
            for m in session.messages[start + 1: end + 1]
        )

        # Cleanup so pytest doesn't warn.
        gate.set()


# ── observers_builder receives task_index ─────────────────────────────────


class TestObserversBuilderTaskIndex:
    @pytest.mark.asyncio
    async def test_builder_called_with_incrementing_task_index(self, monkeypatch):
        async def fake_loop(**kwargs):
            im = list(kwargs.get("initial_messages") or [])
            um = kwargs.get("user_message", "")
            return AgentLoopResult(
                messages=[*im, user_msg(um), assistant_msg("ok")],
                final_content="ok",
                metadata={},
            )

        monkeypatch.setattr(
            "agent_core.components.agent_bus.bus.run_agent_loop", fake_loop,
        )

        seen: list[int] = []

        def observers_builder(_job_id, _item, task_index):
            seen.append(task_index)
            return []

        spec = SubAgentRuntimeSpec(observers_builder=observers_builder)
        bus, sid = await _make_bus_with_session(runtime_spec=spec)

        for prompt in ("t1", "t2", "t3"):
            await bus.submit_task_to_session(sid, prompt)
            await bus.wait_any_session("task-1", timeout=2)

        assert seen == [1, 2, 3]


# ── SSE payload extension ─────────────────────────────────────────────────


class TestSessionTaskSubmittedPayload:
    @pytest.mark.asyncio
    async def test_payload_has_is_reuse_and_role_id(self, monkeypatch):
        async def fake_loop(**kwargs):
            im = list(kwargs.get("initial_messages") or [])
            um = kwargs.get("user_message", "")
            return AgentLoopResult(
                messages=[*im, user_msg(um), assistant_msg("ok")],
                final_content="ok",
                metadata={},
            )

        monkeypatch.setattr(
            "agent_core.components.agent_bus.bus.run_agent_loop", fake_loop,
        )

        event_store = _FakeEventStore()
        from agent_core.protocols import EventSink

        registry.register(EventSink, event_store)

        bus, sid = await _make_bus_with_session()

        await bus.submit_task_to_session(sid, "t1")
        await bus.wait_any_session("task-1", timeout=2)
        await bus.submit_task_to_session(sid, "t2")
        await bus.wait_any_session("task-1", timeout=2)

        submitted = [
            call.kwargs["payload"]
            for call in event_store.append.await_args_list
            if call.kwargs.get("payload", {}).get("trace_type")
            == "session_task_submitted"
        ]
        assert len(submitted) == 2
        assert submitted[0]["is_reuse"] is False
        assert submitted[0]["role_id"] == "researcher"
        assert submitted[0]["task_count"] == 1
        assert submitted[1]["is_reuse"] is True
        assert submitted[1]["role_id"] == "researcher"
        assert submitted[1]["task_count"] == 2
