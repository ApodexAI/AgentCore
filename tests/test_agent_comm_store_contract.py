"""The ``_AgentCommEventStore`` contract that ``AgentComm.consume`` rests on.

``AgentComm`` is the durable half of inter-agent messaging: the cursor in
``consume`` is what makes result recovery exactly-once across a restart. That
cursor is an *integer* — ``EventReader.after_id`` — while ``KernelEvent.id`` is
an opaque ``EventId`` string. So the store owes AgentComm a total-order
ordinal, and this module pins that debt from both sides:

* what a conforming store looks like, and the three ordering guarantees the
  ``consume`` docstring promises callers (order, exactly-once under ``limit``,
  per-``message_type`` cursor isolation);
* that a non-conforming store fails loudly and actionably instead of raising
  ``ValueError`` from ``int()`` — which the bus used to swallow at debug level,
  leaving durable recovery silently dead;
* that a plain ``EventSink`` is *not* an AgentComm store. ``EventSink.append``
  is flat ``(task_id, event_type, payload, agent_role)`` and cannot carry
  ``from_agent`` / ``to_agent`` / ``message_type`` / ``correlation_id``, which
  are exactly the fields the reader queries on.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_core.components.agent_bus.agent_comm import (
    AgentComm,
    DeliveryMode,
    EventStoreContractError,
    _AgentCommEventStore,
    event_ordinal,
)
from agent_core.events import EventType
from agent_core.models.agent_message import AgentMessage
from agent_core.models.event import KernelEvent
from agent_core.protocols import EventSink
from agent_core.runtime.events import EventBus
from agent_core.types import EventId, TaskId, new_event_id

RESULT = "subtask_result"
STATUS = "status_update"


# ──────────────────────────────────────────────────────────────────────────
# Reference implementations
# ──────────────────────────────────────────────────────────────────────────


class ReferenceStore:
    """Minimal conforming store: stamps a monotonic ``seq`` on append."""

    def __init__(self) -> None:
        self.events: list[KernelEvent] = []
        self._next_seq = 1

    async def append(self, event: KernelEvent) -> KernelEvent:
        stamped = event.model_copy(update={
            "id": EventId(new_event_id()),
            "seq": self._next_seq,
        })
        self._next_seq += 1
        self.events.append(stamped)
        return stamped

    def replay(self, task_id: str):  # pragma: no cover — unused here
        async def _empty():
            if False:
                yield None
        return _empty()

    async def get_events(
        self,
        task_id: Any,
        event_type: Any = None,
        after_id: int = 0,
        limit: int | None = None,
    ) -> list[KernelEvent]:
        found = [
            e for e in self.events
            if str(e.task_id) == str(task_id) and (e.seq or 0) > after_id
        ]
        return found[:limit] if limit is not None else found

    async def get_events_for_agent(
        self,
        to_agent: str,
        after_id: int = 0,
        limit: int = 50,
        *,
        task_id: str | Any | None = None,
        message_type: str | None = None,
    ) -> list[KernelEvent]:
        found = [
            e for e in self.events
            if e.to_agent == to_agent
            and (e.seq or 0) > after_id
            and (task_id is None or str(e.task_id) == str(task_id))
            and (message_type is None or e.message_type == message_type)
        ]
        return found[:limit]


class DecimalIdStore(ReferenceStore):
    """Legacy shape: the integer primary key *is* the event id, no ``seq``."""

    async def append(self, event: KernelEvent) -> KernelEvent:
        stamped = event.model_copy(update={
            "id": EventId(str(self._next_seq)), "seq": None,
        })
        self._next_seq += 1
        self.events.append(stamped)
        return stamped

    async def get_events_for_agent(
        self, to_agent, after_id=0, limit=50, *, task_id=None, message_type=None,
    ) -> list[KernelEvent]:
        found = [
            e for e in self.events
            if e.to_agent == to_agent
            and int(str(e.id)) > after_id
            and (task_id is None or str(e.task_id) == str(task_id))
            and (message_type is None or e.message_type == message_type)
        ]
        return found[:limit]


class OpaqueIdStore(ReferenceStore):
    """Non-conforming: keeps ``new_event_id()`` hex and stamps no ordinal."""

    async def append(self, event: KernelEvent) -> KernelEvent:
        stamped = event.model_copy(update={
            "id": EventId(new_event_id()), "seq": None,
        })
        self.events.append(stamped)
        return stamped

    async def get_events_for_agent(
        self, to_agent, after_id=0, limit=50, *, task_id=None, message_type=None,
    ) -> list[KernelEvent]:
        return [
            e for e in self.events
            if e.to_agent == to_agent
            and (task_id is None or str(e.task_id) == str(task_id))
            and (message_type is None or e.message_type == message_type)
        ]


class _FlatSink:
    """A valid ``EventSink`` — and therefore *not* an AgentComm store."""

    async def append(
        self,
        task_id: Any = "",
        event_type: Any = None,
        payload: dict[str, Any] | None = None,
        agent_role: str = "system",
    ) -> None:
        return None

    def replay(self, task_id: str):  # pragma: no cover
        async def _empty():
            if False:
                yield None
        return _empty()


def _msg(
    to: str, *, task: str = "t1", mtype: str = RESULT, job: str = "j1",
) -> AgentMessage:
    return AgentMessage(
        task_id=task,
        from_agent="worker",
        to_agent=to,
        message_type=mtype,
        content={"correlation_id": job, "result": {"final_content": job}},
    )


# ──────────────────────────────────────────────────────────────────────────
# event_ordinal: what counts as a total order
# ──────────────────────────────────────────────────────────────────────────


def test_seq_is_the_ordinal_when_present():
    event = KernelEvent(
        task_id=TaskId("t1"), event_type=EventType.AGENT_MESSAGE,
        id=EventId("ae8dfb7969004eea"), seq=7,
    )
    assert event_ordinal(event) == 7


def test_decimal_id_is_accepted_as_a_fallback_ordinal():
    """Stores that expose their integer primary key as the id keep working."""
    event = KernelEvent(
        task_id=TaskId("t1"), event_type=EventType.AGENT_MESSAGE,
        id=EventId("42"),
    )
    assert event_ordinal(event) == 42


def test_seq_wins_over_a_decimal_id():
    event = KernelEvent(
        task_id=TaskId("t1"), event_type=EventType.AGENT_MESSAGE,
        id=EventId("42"), seq=7,
    )
    assert event_ordinal(event) == 7


@pytest.mark.parametrize("raw_id", ["", "ae8dfb7969004eea", "evt-1", "1.5", "-3"])
def test_opaque_id_without_seq_is_a_contract_error(raw_id):
    """Actionable error, not ``ValueError`` from ``int()``.

    ``new_event_id()`` hex is the trap: neither numeric nor monotonic, and it
    is the default id generator in this package — so a host can build a
    plausible-looking store that cannot order a cursor at all.
    """
    event = KernelEvent(
        task_id=TaskId("t1"), event_type=EventType.AGENT_MESSAGE,
        id=EventId(raw_id),
    )
    with pytest.raises(EventStoreContractError) as excinfo:
        event_ordinal(event)
    assert "KernelEvent.seq" in str(excinfo.value)


def test_default_event_id_generator_cannot_order_a_cursor():
    """Pins the reason ``seq`` exists at all."""
    event = KernelEvent(
        task_id=TaskId("t1"), event_type=EventType.AGENT_MESSAGE,
        id=new_event_id(),
    )
    with pytest.raises(EventStoreContractError):
        event_ordinal(event)


# ──────────────────────────────────────────────────────────────────────────
# The store protocol: which shape of ``append`` AgentComm requires
# ──────────────────────────────────────────────────────────────────────────


def test_reference_store_satisfies_the_agent_comm_contract():
    assert isinstance(ReferenceStore(), _AgentCommEventStore)


def test_flat_event_sink_is_not_an_agent_comm_store():
    """``EventSink`` is a telemetry sink, not a message store.

    It is structurally a valid ``EventSink`` and must stay one — the bus uses
    that shape for session lifecycle events — but AgentComm needs a
    whole-event ``append`` that can carry the routing fields.
    """
    sink = _FlatSink()
    assert isinstance(sink, EventSink)
    assert not isinstance(sink, _AgentCommEventStore)


@pytest.mark.asyncio
async def test_send_round_trips_the_routing_fields():
    """The fields ``EventSink.append`` cannot express are the ones queried on."""
    store = ReferenceStore()
    comm = AgentComm(store, EventBus())

    persisted = await comm.send(_msg("critic"), DeliveryMode.QUEUE)

    assert persisted.to_agent == "critic"
    assert persisted.from_agent == "worker"
    assert persisted.message_type == RESULT
    assert persisted.correlation_id == "j1"
    assert event_ordinal(persisted) == 1


# ──────────────────────────────────────────────────────────────────────────
# The three guarantees the consume() docstring makes to callers
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("store_cls", [ReferenceStore, DecimalIdStore])
async def test_awaited_sends_come_back_in_send_order(store_cls):
    comm = AgentComm(store_cls(), EventBus())
    for i in range(5):
        await comm.send(_msg("critic", job=f"j{i}"))

    events = await comm.consume("critic", task_id="t1", message_type=RESULT)

    assert [e.payload["content"]["correlation_id"] for e in events] == [
        f"j{i}" for i in range(5)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("store_cls", [ReferenceStore, DecimalIdStore])
async def test_limit_truncation_delivers_each_message_exactly_once(store_cls):
    """The remainder arrives on the next call, still in order, never twice."""
    comm = AgentComm(store_cls(), EventBus())
    for i in range(5):
        await comm.send(_msg("critic", job=f"j{i}"))

    seen: list[str] = []
    for _ in range(3):
        batch = await comm.consume(
            "critic", task_id="t1", limit=2, message_type=RESULT,
        )
        seen += [e.payload["content"]["correlation_id"] for e in batch]

    assert seen == [f"j{i}" for i in range(5)]
    assert await comm.consume(
        "critic", task_id="t1", limit=2, message_type=RESULT,
    ) == []


@pytest.mark.asyncio
async def test_typed_cursors_do_not_carry_each_others_mail_past_unread():
    """A shared cursor would let the result reader consume status updates."""
    comm = AgentComm(ReferenceStore(), EventBus())
    await comm.send(_msg("critic", mtype=STATUS, job="s0"))
    await comm.send(_msg("critic", mtype=RESULT, job="r0"))
    await comm.send(_msg("critic", mtype=STATUS, job="s1"))

    results = await comm.consume("critic", task_id="t1", message_type=RESULT)
    statuses = await comm.consume("critic", task_id="t1", message_type=STATUS)

    assert [e.payload["content"]["correlation_id"] for e in results] == ["r0"]
    assert [e.payload["content"]["correlation_id"] for e in statuses] == ["s0", "s1"]


@pytest.mark.asyncio
async def test_reset_cursor_replays_the_selected_stream():
    comm = AgentComm(ReferenceStore(), EventBus())
    await comm.send(_msg("critic", job="r0"))

    first = await comm.consume("critic", task_id="t1", message_type=RESULT)
    assert await comm.consume("critic", task_id="t1", message_type=RESULT) == []

    comm.reset_cursor("critic", "t1", message_type=RESULT)
    replayed = await comm.consume("critic", task_id="t1", message_type=RESULT)

    assert [e.id for e in replayed] == [e.id for e in first]


# ──────────────────────────────────────────────────────────────────────────
# Non-conforming stores: loud, and the cursor stays put
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_consume_rejects_an_ordinal_less_store_actionably():
    """Previously a bare ``ValueError`` the bus swallowed at debug level."""
    comm = AgentComm(OpaqueIdStore(), EventBus())
    await comm.send(_msg("critic"))

    with pytest.raises(EventStoreContractError):
        await comm.consume("critic", task_id="t1", message_type=RESULT)


@pytest.mark.asyncio
async def test_misordered_batch_leaves_the_cursor_untouched():
    """Advancing to max(ids) would permanently skip the lower ids."""
    store = ReferenceStore()
    comm = AgentComm(store, EventBus())
    await comm.send(_msg("critic", job="r0"))
    await comm.send(_msg("critic", job="r1"))
    store.events.reverse()

    with pytest.raises(RuntimeError, match="ascending"):
        await comm.consume("critic", task_id="t1", message_type=RESULT)

    store.events.reverse()
    recovered = await comm.consume("critic", task_id="t1", message_type=RESULT)
    assert [e.payload["content"]["correlation_id"] for e in recovered] == ["r0", "r1"]
