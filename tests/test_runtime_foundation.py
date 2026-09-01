from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import cast

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
    normalize_execution_context,
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
        seen.append((str(event["event_type"]), cast("int", event["value"])))

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


@dataclass
class _NullPolicy:
    """The common shape where every constraint is an optional tuple."""

    allow_tools: tuple[str, ...] | None = None
    deny_tools: tuple[str, ...] | None = None
    deny_tool_prefixes: tuple[str, ...] | None = None


def test_execution_policy_tolerates_none_valued_attributes() -> None:
    context = from_execution_policy(_NullPolicy())
    assert context.is_empty()
    assert context.allows("bash")


def test_direct_construction_normalizes_case() -> None:
    context = ToolPermissionContext(
        allow_names=frozenset({"Bash", "WEB_SEARCH"}),
        deny_names=frozenset({"Web_Search"}),
        deny_prefixes=("Internal_",),
    )
    assert context.allows("Bash")
    assert context.allows("bash")
    assert not context.allows("web_search")
    assert not context.allows("INTERNAL_admin")


def test_permission_filter_accepts_any_iterable() -> None:
    context = ToolPermissionContext.from_iterables(deny_names=["bash"])
    assert context.filter(frozenset({"bash", "python"})) == {"python"}
    assert context.filter(["bash", "python"]) == {"python"}


def test_config_map_true_entry_is_an_exhaustive_allowlist() -> None:
    """A single ``True`` denies every unlisted tool; ``False`` alone does not."""
    enabling = from_config_map({"bash": True})
    assert enabling.allows("bash")
    assert not enabling.allows("web_search")

    disabling = from_config_map({"web_search": False})
    assert disabling.allows("bash")
    assert not disabling.allows("web_search")


def test_execution_scope_metadata_is_deeply_independent() -> None:
    persisted: dict[str, object] = {"labels": {"run": "first"}, "steps": [1]}
    state = {"execution_context": persisted}
    scope = build_execution_scope(task_id="t", phase_id="p", role_id="r", state=state)

    labels = cast("dict[str, object]", scope.metadata["labels"])
    labels["run"] = "second"
    steps = cast("list[object]", scope.metadata["steps"])
    steps.append(2)

    assert persisted == {"labels": {"run": "first"}, "steps": [1]}


def test_execution_scope_role_id_overrides_stale_persisted_agent_id() -> None:
    state = {"execution_context": {"agent_id": "previous_role"}}
    scope = build_execution_scope(
        task_id="t", phase_id="phase_2", role_id="current_role", state=state
    )
    assert scope.metadata["agent_id"] == "current_role"
    assert scope.metadata["agent_id"] == scope.role_id

    # With no authoritative role to apply, a persisted value is left alone.
    kept = build_execution_scope(
        task_id="t", phase_id="phase_2", role_id="", state={"execution_context": {"agent_id": "x"}}
    )
    assert kept.metadata["agent_id"] == "x"


def test_normalize_execution_context_survives_uncopyable_values() -> None:
    lock = threading.Lock()
    normalized = normalize_execution_context({"lock": lock, "count": 1})
    assert normalized["count"] == 1
    assert normalized["lock"] is lock


@pytest.mark.asyncio
async def test_event_payload_is_not_shared_between_handlers() -> None:
    bus = EventBus()
    observed: list[dict[str, object]] = []

    async def mutator(event: dict[str, object]) -> None:
        await asyncio.sleep(0)
        event.pop("value", None)
        event["injected"] = True

    async def observer(event: dict[str, object]) -> None:
        await asyncio.sleep(0)
        observed.append(dict(event))

    bus.subscribe("event", mutator)
    bus.subscribe("event", observer)
    await bus.publish("event", {"value": 3})

    assert observed == [{"event_type": "event", "value": 3}]


@pytest.mark.asyncio
async def test_published_event_type_wins_over_payload() -> None:
    bus = EventBus()
    seen: list[str] = []

    async def handler(event: dict[str, object]) -> None:
        seen.append(str(event["event_type"]))

    bus.subscribe(EventType.TOOL_RESULT, handler)
    await bus.publish(EventType.TOOL_RESULT, {"event_type": "spoofed"})
    assert seen == ["tool_result"]


@pytest.mark.asyncio
async def test_event_bus_subscriptions_can_be_removed() -> None:
    bus = EventBus()
    calls = 0

    async def handler(_event: dict[str, object]) -> None:
        nonlocal calls
        calls += 1

    bus.subscribe("event", handler)
    bus.subscribe_all(handler)
    await bus.publish("event", {})
    assert calls == 2

    assert bus.unsubscribe("event", handler)
    assert not bus.unsubscribe("event", handler)
    assert bus.unsubscribe_all(handler)
    assert not bus.unsubscribe_all(handler)
    await bus.publish("event", {})
    assert calls == 2

    bus.subscribe("event", handler)
    bus.clear()
    await bus.publish("event", {})
    assert calls == 2
