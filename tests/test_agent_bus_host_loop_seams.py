"""A host's loop hooks must reach sub-agents, not only its own agents.

``AgentBus`` runs every sub-agent through :func:`run_agent_loop`. A host that
wraps that engine installs its own ``AgentLoopHooks`` and tool-call parser in
*its* wrapper — per-tool timeout floors, execution scopes for cost and trace
attribution, overflow spill files, session affinity, how a hallucinated tool
name is answered. If the bus called the engine with neither, those would apply
to the agents the host launches directly and silently not to the sub-agents
the bus spawns.

The asymmetry has no runtime symptom: the loop still runs, it just runs with
all-default no-op hooks, so sub-agent tool results get truncated with no spill
file, sub-agent cost lands on the parent's role, and per-tool timeout floors
vanish. These two seams prevent it, and both dispatch paths — ``submit`` and
the session queue, including its deferred drain — have to pass them through.

The seams are deliberately NOT a replacement for ``run_agent_loop`` itself:
that module-global name is what tests (here and in every host) monkeypatch to
keep a loop out of a unit test, so shadowing it would quietly turn those into
real loop runs.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent_core.components.agent_bus import (
    AgentBus,
    SubTask,
    configure_default_runtime_hooks,
    configure_default_tool_call_parser,
)
from agent_core.components.agent_bus import bus as bus_module
from agent_core.loop_types import AgentLoopResult
from agent_core.messages import assistant_msg, user_msg
from agent_core.models.agent_definition import AgentDefinition
from agent_core.runtime.loop.agent_loop import AgentLoopHooks
from agent_core.runtime.registries import services as registry
from agent_core.runtime.registries.agents import AgentRegistry
from agent_core.runtime.resources.manager import ResourceManager


@pytest.fixture(autouse=True)
def _clean_globals():
    registry.clear()
    yield
    registry.clear()
    configure_default_runtime_hooks(None)
    configure_default_tool_call_parser(None)


def _registry_with(*role_ids: str) -> None:
    reg = AgentRegistry()
    for role_id in role_ids:
        reg.register(AgentDefinition(
            role_id=role_id, display_name=role_id, system_prompt="p",
            allowed_tools=[], color="#000", icon="agent",
        ))
    registry.register(AgentRegistry, reg)
    resource_mgr = MagicMock(spec=ResourceManager)
    resource_mgr.get_llm.return_value = MagicMock()
    resource_mgr.get_tools_for_role.return_value = []
    registry.register(ResourceManager, resource_mgr)


def _recording_engine(calls: list[dict[str, Any]]):
    async def engine(**kwargs: Any) -> AgentLoopResult:
        calls.append(kwargs)
        um = str(kwargs.get("user_message", ""))
        return AgentLoopResult(
            messages=[
                *list(kwargs.get("initial_messages") or []),
                user_msg(um),
                assistant_msg(f"r:{um}"),
            ],
            final_content=f"r:{um}",
            metadata={},
        )
    return engine


def test_unconfigured_resolvers_select_the_engine_defaults():
    assert bus_module._runtime_hooks() is None
    assert bus_module._tool_call_parser() is None


def test_none_restores_the_engine_defaults():
    hooks = AgentLoopHooks()
    configure_default_runtime_hooks(lambda: hooks)
    assert bus_module._runtime_hooks() is hooks
    configure_default_runtime_hooks(None)
    assert bus_module._runtime_hooks() is None


def test_hooks_are_resolved_at_call_time_not_at_configure_time():
    """A host below the loop package in its layer stack imports lazily."""
    resolved: list[int] = []

    def resolver() -> Any:
        resolved.append(1)
        return AgentLoopHooks()

    configure_default_runtime_hooks(resolver)
    assert resolved == []
    bus_module._runtime_hooks()
    bus_module._runtime_hooks()
    assert len(resolved) == 2


@pytest.mark.asyncio
async def test_submit_dispatch_passes_the_host_seams_through():
    _registry_with("researcher")
    calls: list[dict[str, Any]] = []
    hooks, parser = AgentLoopHooks(), object()
    configure_default_runtime_hooks(lambda: hooks)
    configure_default_tool_call_parser(lambda: parser)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(bus_module, "run_agent_loop", _recording_engine(calls))
        bus = AgentBus()
        job_id = await bus.submit("t1", SubTask(question="q1", role_id="researcher"))
        collected = await bus.collect([job_id], timeout=5)

    assert [r.final_content for r in collected.completed] == ["r:q1"]
    assert [c["runtime_hooks"] for c in calls] == [hooks]
    assert [c["parser"] for c in calls] == [parser]


@pytest.mark.asyncio
async def test_session_dispatch_passes_the_host_seams_through():
    _registry_with("researcher")
    calls: list[dict[str, Any]] = []
    hooks, parser = AgentLoopHooks(), object()
    configure_default_runtime_hooks(lambda: hooks)
    configure_default_tool_call_parser(lambda: parser)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(bus_module, "run_agent_loop", _recording_engine(calls))
        bus = AgentBus()
        session_id = await bus.create_session(
            task_id="t1", name="lit", role_id="researcher",
            system_prompt="You are a researcher.",
            llm_override=MagicMock(), tools_override=[],
        )
        await bus.submit_task_to_session(session_id, "S1")
        # A second task dispatches later, from the drain path's call stack.
        await bus.submit_task_to_session(session_id, "S2")

        deadline = asyncio.get_running_loop().time() + 5
        while len(calls) < 2 and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.01)

    assert [c["user_message"] for c in calls] == ["S1", "S2"]
    assert [c["runtime_hooks"] for c in calls] == [hooks, hooks]
    assert [c["parser"] for c in calls] == [parser, parser]


@pytest.mark.asyncio
async def test_a_monkeypatched_engine_still_keeps_a_loop_out_of_a_unit_test():
    """The seams must not shadow the name hosts' tests stub."""
    _registry_with("researcher")
    calls: list[dict[str, Any]] = []
    configure_default_runtime_hooks(lambda: AgentLoopHooks())

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(bus_module, "run_agent_loop", _recording_engine(calls))
        bus = AgentBus()
        job_id = await bus.submit("t1", SubTask(question="q1", role_id="researcher"))
        await bus.collect([job_id], timeout=5)

    assert len(calls) == 1
