"""A dispatched session task must be running by the time dispatch returns.

``submit_task_to_session`` is fire-and-return: a host tool hands back a job id
and its invocation scope closes. Nothing on the dispatch path is guaranteed to
suspend — ``_emit_session_task_submitted`` returns without awaiting when no
sink is installed, and a sink whose ``append`` never awaits does not yield
either — so establishing the task takes an explicit yield.

Without it the job is still ``"submitted"`` when the caller moves on, and a
host that reports job state right after dispatching (a task board, a status
line) describes a worker that has not begun as though it were merely queued.
The window is also the one ``_mark_job_aborted`` exists to paper over: it
releases the guard reservation and substitutes a stand-in result precisely
because a task can be cancelled before ``_run_and_finalize`` reaches its
``finally``. Establishing the task first is the narrower fix — every
cancellation then lands inside the job wrapper on its own.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from agent_core.components.agent_bus import AgentBus
from agent_core.models.agent_definition import AgentDefinition
from agent_core.runtime.registries import services as registry
from agent_core.runtime.registries.agents import AgentRegistry
from agent_core.runtime.resources.manager import ResourceManager

TASK_ID = "t-handoff"


@pytest.fixture(autouse=True)
def _clear_registry():
    registry.clear()
    yield
    registry.clear()


def _register_researcher() -> None:
    reg = AgentRegistry()
    reg.register(AgentDefinition(
        role_id="researcher", display_name="researcher", system_prompt="p",
        allowed_tools=[], color="#000", icon="agent",
    ))
    registry.register(AgentRegistry, reg)
    resource_mgr = MagicMock(spec=ResourceManager)
    resource_mgr.get_llm.return_value = MagicMock()
    resource_mgr.get_tools_for_role.return_value = []
    registry.register(ResourceManager, resource_mgr)


async def _session(bus: AgentBus) -> str:
    return await bus.create_session(
        task_id=TASK_ID, name="lit", role_id="researcher",
        system_prompt="You are a researcher.",
        llm_override=MagicMock(), tools_override=[],
    )


def _parked_loop(entered: asyncio.Event):
    async def loop(**kwargs):
        entered.set()
        await asyncio.sleep(3600)
    return loop


@pytest.mark.asyncio
async def test_dispatch_leaves_the_job_running_not_merely_submitted(monkeypatch):
    """``entry.status`` flips to ``"running"`` inside ``_run_and_finalize``.

    Reading ``"submitted"`` here means the coroutine had not started — the
    condition the trailing yield exists to rule out.
    """
    entered = asyncio.Event()
    monkeypatch.setattr(
        "agent_core.components.agent_bus.bus.run_agent_loop",
        _parked_loop(entered),
    )
    _register_researcher()
    bus = AgentBus()
    session_id = await _session(bus)

    job_id = await bus.submit_task_to_session(session_id, "q")

    assert bus.get_job_status(job_id) == "running"
    assert entered.is_set()


@pytest.mark.asyncio
async def test_cancelling_a_round_never_lets_a_queued_task_reach_the_llm(
    monkeypatch,
):
    """Tearing down a round must not first start what it is tearing down.

    Cancelling a running job runs its ``finally``, which drains the session
    queue. Interleaved with the cancellation pass that dispatches the next
    queued task — one LLM round-trip in, then cancelled again — so the queue
    has to be cleared first.
    """
    gate = asyncio.Event()
    reached_llm: list[str] = []

    async def loop(**kwargs):
        reached_llm.append(str(kwargs.get("user_message")))
        await gate.wait()

    monkeypatch.setattr(
        "agent_core.components.agent_bus.bus.run_agent_loop", loop,
    )
    _register_researcher()
    bus = AgentBus()
    session_id = await _session(bus)

    jobs = [
        await bus.submit_task_to_session(session_id, prompt)
        for prompt in ("T1", "T2", "T3")
    ]

    cancelled = await bus.cancel_for_parent(TASK_ID)

    assert reached_llm == ["T1"]
    # Every job this call tore down is counted, including the running one
    # that absorbed its own cancellation.
    assert cancelled == 3
    assert [bus.get_job_status(job) for job in jobs] == ["aborted"] * 3
    assert not bus.get_session(session_id).pending_tasks
