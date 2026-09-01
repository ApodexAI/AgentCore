from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from agent_core.events import EventType
from agent_core.execution_context import (
    ExecutionScope,
    build_execution_scope,
    chain_fallback_active,
    chain_fallback_scope,
    ensure_trace_metadata,
    get_current_execution_scope,
    get_current_tool_budget,
    get_current_tool_call_id,
    reset_current_execution_scope,
    reset_current_tool_budget,
    reset_current_tool_call_id,
    set_current_execution_scope,
    set_current_tool_budget,
    set_current_tool_call_id,
)
from agent_core.runtime.events import EventBus
from agent_core.runtime.tool_permission import (
    ToolPermissionContext,
    from_config_map,
    from_execution_policy,
)
from agent_core.types import TaskStatus, new_event_id, new_prompt_id, new_session_id, new_step_id


def test_identity_shapes_and_string_event_contract() -> None:
    assert len(new_event_id()) == 16
    assert len(new_session_id()) == 12
    assert len(new_prompt_id()) == 12
    assert len(new_step_id()) == 10
    assert TaskStatus.RUNNING == "running"
    assert EventType.TOOL_RESULT == "tool_result"


def test_execution_scope_build_and_trace_metadata() -> None:
    state = {"execution_context": {"session_id": "session", "custom": 1}}
    scope = build_execution_scope(task_id="t", phase_id="p", role_id="r", state=state)
    assert scope.metadata == {"session_id": "session", "custom": 1, "agent_id": "r"}
    assert scope.metadata is not state["execution_context"]

    ensure_trace_metadata(scope.metadata, default_step_id="step")
    first_prompt = scope.metadata["prompt_id"]
    ensure_trace_metadata(scope.metadata, refresh_prompt_id=True)
    assert scope.metadata["session_id"] == "session"
    assert scope.metadata["step_id"] == "step"
    assert scope.metadata["prompt_id"] != first_prompt


def test_execution_context_tokens_restore_nested_values() -> None:
    assert get_current_execution_scope() is None
    scope_token = set_current_execution_scope(ExecutionScope(task_id="task"))
    call_token = set_current_tool_call_id("call")
    budget_token = set_current_tool_budget(12.5)
    try:
        assert get_current_execution_scope() == ExecutionScope(task_id="task")
        assert get_current_tool_call_id() == "call"
        assert get_current_tool_budget() == 12.5
        assert not chain_fallback_active()
        with chain_fallback_scope():
            assert chain_fallback_active()
        assert not chain_fallback_active()
    finally:
        reset_current_tool_budget(budget_token)
        reset_current_tool_call_id(call_token)
        reset_current_execution_scope(scope_token)

    assert get_current_execution_scope() is None
    assert get_current_tool_call_id() == ""
    assert get_current_tool_budget() is None


@pytest.mark.asyncio
async def test_execution_context_is_isolated_between_tasks() -> None:
    async def read(call_id: str) -> str:
        token = set_current_tool_call_id(call_id)
        try:
            await asyncio.sleep(0)
            return get_current_tool_call_id()
        finally:
            reset_current_tool_call_id(token)

    assert await asyncio.gather(read("a"), read("b")) == ["a", "b"]


@pytest.mark.asyncio
async def test_event_bus_dispatches_typed_and_global_handlers() -> None:
    bus = EventBus()
    seen: list[tuple[str, int]] = []

    async def handler(event: dict[str, object]) -> None:
        seen.append((str(event["event_type"]), int(event["value"])))

    bus.subscribe(EventType.TOOL_RESULT, handler)
    bus.subscribe_all(handler)
    await bus.publish(EventType.TOOL_RESULT, {"value": 3})
    assert seen == [("tool_result", 3), ("tool_result", 3)]


@pytest.mark.asyncio
async def test_event_bus_isolates_handler_failures() -> None:
    bus = EventBus()
    completed = False

    async def broken(_event: dict[str, object]) -> None:
        raise RuntimeError("boom")

    async def healthy(_event: dict[str, object]) -> None:
        nonlocal completed
        completed = True

    bus.subscribe("event", broken)
    bus.subscribe("event", healthy)
    await bus.publish("event", {})
    assert completed


def test_tool_permission_merge_is_fail_closed() -> None:
    profile = ToolPermissionContext.from_iterables(
        allow_names={"bash", "web_search"}, deny_prefixes=("internal_",)
    )
    request = ToolPermissionContext.from_iterables(
        allow_names={"bash", "python"}, deny_names={"bash"}
    )
    merged = profile.merge(request)
    assert merged.allow_names == frozenset({"bash"})
    assert not merged.allows("bash")
    assert not merged.allows("internal_admin")


def test_config_map_requires_bare_booleans(caplog: pytest.LogCaptureFixture) -> None:
    context = from_config_map({"bash": True, "web_search": False, "python": "false"})
    assert context.allow_names == frozenset({"bash"})
    assert context.deny_names == frozenset({"web_search"})
    assert "ignored" in caplog.text


@dataclass
class _Policy:
    allow_tools: tuple[str, ...] = ("bash",)
    deny_tools: tuple[str, ...] = ("web_search",)
    deny_tool_prefixes: tuple[str, ...] = ("internal_",)


def test_execution_policy_is_structurally_consumed() -> None:
    context = from_execution_policy(_Policy())
    assert context.allows("bash")
    assert not context.allows("web_search")
    assert not context.allows("internal_admin")
