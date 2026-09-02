"""Durable result recovery must not fail silently on a bad event store.

``AgentBus._reconcile_from_messages`` is deliberately soft — no AgentComm, no
messages, or a transient read failure all fall back to the in-memory path. That
softness used to swallow a *permanent* fault too: a store with no total-order
ordinal raised ``ValueError`` out of ``int(event.id)``, the blanket
``except Exception`` logged it at debug level, and the whole L1-0 recovery path
was dead with nothing in the logs at default verbosity.

So the two behaviours pinned here are complementary: recovery still works
end-to-end against a conforming store, and a non-conforming one is reported at
ERROR while still degrading to ``None`` rather than taking the run down.
"""

from __future__ import annotations

import logging

import pytest

from agent_core.components.agent_bus.agent_comm import AgentComm
from agent_core.components.agent_bus.bus import AgentBus
from agent_core.models.agent_message import AgentMessage
from agent_core.runtime.events import EventBus
from agent_core.runtime.registries import services as registry
from tests.test_agent_comm_store_contract import OpaqueIdStore, ReferenceStore

RESULT = "subtask_result"


@pytest.fixture(autouse=True)
def _clear_registry():
    registry.clear()
    yield
    registry.clear()


def _result_msg(task_id: str, job_id: str) -> AgentMessage:
    return AgentMessage(
        task_id=task_id,
        from_agent="worker",
        to_agent="lead",
        message_type=RESULT,
        content={
            "correlation_id": job_id,
            "session_id": "s1",
            "result": {
                "question": "q", "role_id": "researcher",
                "success": True, "final_content": "done",
            },
        },
    )


async def _bus_with_store(store) -> AgentBus:
    comm = AgentComm(store, EventBus())
    registry.register(AgentComm, comm)
    await comm.send(_result_msg("t-rec", "t-rec.job.1"))
    bus = AgentBus(event_sink=store)
    bus._note_result_recipient("t-rec", {"parent_agent_id": "lead"})
    return bus


@pytest.mark.asyncio
async def test_conforming_store_recovers_the_result():
    bus = await _bus_with_store(ReferenceStore())

    recovered = await bus._reconcile_from_messages("t-rec")

    assert recovered is not None
    session_id, result = recovered
    assert session_id == "s1"
    assert result.final_content == "done"
    assert result.success is True


@pytest.mark.asyncio
async def test_recovered_result_is_not_delivered_twice():
    bus = await _bus_with_store(ReferenceStore())

    assert await bus._reconcile_from_messages("t-rec") is not None
    assert await bus._reconcile_from_messages("t-rec") is None


@pytest.mark.asyncio
async def test_ordinal_less_store_is_reported_at_error_not_swallowed(caplog):
    bus = await _bus_with_store(OpaqueIdStore())

    with caplog.at_level(logging.ERROR, logger="agent_core.components.agent_bus.bus"):
        recovered = await bus._reconcile_from_messages("t-rec")

    # Still soft: the run continues on the in-memory path.
    assert recovered is None
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, "a permanent store fault must not be logged at debug level"
    assert "ordinal" in errors[0].getMessage()
