"""Phase 4 D13 surgery 1 — ``AgentBus(event_sink=...)`` injection.

Lets the SDK assembly hand a custom EventSink straight to the bus
without going through the global service registry. The legacy
zero-arg constructor still works (falls back to
``registry.get_optional(EventStore)``).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from agent_core.components.agent_bus.bus import AgentBus


class _FakeSink:
    """Minimal EventSink Protocol stub recording every append."""

    def __init__(self) -> None:
        self.appended: list[dict[str, Any]] = []

    async def append(
        self,
        task_id: Any = "",
        event_type: Any = None,
        payload: dict[str, Any] | None = None,
        agent_role: str = "system",
    ) -> None:
        self.appended.append({
            "task_id": task_id,
            "event_type": event_type,
            "payload": payload or {},
            "agent_role": agent_role,
        })

    def replay(self, task_id: str):  # pragma: no cover — unused in this test
        async def _empty():
            if False:
                yield None
        return _empty()


def test_default_constructor_falls_back_to_registry() -> None:
    """Zero-arg construction still works via registry lookup."""
    bus = AgentBus()
    # ``_event_sink()`` returns whatever ``registry.get_optional`` returns,
    # which is None in a clean test environment.
    assert bus._event_sink() is None


def test_injected_sink_is_returned_directly() -> None:
    """Injected sink bypasses the registry entirely."""
    sink = _FakeSink()
    bus = AgentBus(event_sink=sink)
    assert bus._event_sink() is sink


def test_injected_sink_used_when_registry_also_has_one() -> None:
    """Injected sink wins over registry when both are present."""
    sink = _FakeSink()
    other = _FakeSink()
    bus = AgentBus(event_sink=sink)

    with patch(
        "agent_core.components.agent_bus.bus.registry.get_optional",
        return_value=other,
    ):
        assert bus._event_sink() is sink


def test_injected_agent_registry_returns_directly() -> None:
    """D13.3 — injected AgentRegistry bypasses the registry."""
    from agent_core.runtime.registries.agents import AgentRegistry
    reg = AgentRegistry()
    bus = AgentBus(agent_registry=reg)
    assert bus._agent_registry(required=False) is reg
    assert bus._agent_registry(required=True) is reg


def test_injected_resource_manager_returns_directly() -> None:
    """D13.3 — injected ResourceManager bypasses the registry."""
    sentinel = object()
    bus = AgentBus(resource_manager=sentinel)
    assert bus._resource_manager(required=False) is sentinel
    assert bus._resource_manager(required=True) is sentinel


@pytest.mark.asyncio
async def test_session_task_lifecycle_uses_injected_sink() -> None:
    """End-to-end: lifecycle helpers route through the injected sink."""
    from agent_core.components.agent_bus.models import SubAgentResult, SubAgentSession
    from agent_core.components.agent_bus.runtime import (
        emit_session_task_completed,
        emit_session_task_submitted,
    )
    from agent_core.runtime.loop.message_trimmer import NullTrimmer

    sink = _FakeSink()
    session = SubAgentSession(
        session_id="s1", task_id="root", name="alice",
        role_id="researcher", system_prompt="", tools=[], llm=None,
        trimmer=NullTrimmer(),
    )

    await emit_session_task_submitted(
        session, "j1", "do the thing", event_sink=sink,
    )
    await emit_session_task_completed(
        session, "j1",
        SubAgentResult(
            question="q", role_id="researcher",
            success=True, final_content="done",
        ),
        event_sink=sink,
    )

    assert len(sink.appended) == 2
    assert sink.appended[0]["payload"]["trace_type"] == "session_task_submitted"
    assert sink.appended[1]["payload"]["trace_type"] == "session_task_completed"
