"""AgentComm — inter-agent communication service.

Sends typed AgentMessages, persists them as KernelEvents,
and publishes to EventBus for real-time streaming.

Phase D (Issue #24): DeliveryMode (TRIGGER/QUEUE) for message routing.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from enum import StrEnum
from typing import Protocol, runtime_checkable

from agent_core.events import EventType
from agent_core.models.agent_message import AgentMessage
from agent_core.models.event import KernelEvent
from agent_core.protocols import EventReader
from agent_core.runtime.events import EventBus
from agent_core.types import TaskId

logger = logging.getLogger(__name__)


class EventStoreContractError(RuntimeError):
    """The event store violated the :class:`_AgentCommEventStore` contract.

    Distinct from a transient read failure on purpose: a caller that treats
    read errors as soft (``AgentBus`` result recovery does) must still
    surface a misconfigured store loudly, because every future read will
    fail the same way.
    """


def event_ordinal(event: KernelEvent) -> int:
    """Return the store-assigned total-order ordinal of ``event``.

    Prefers the explicit :attr:`KernelEvent.seq`. Falls back to a decimal
    ``id`` so a store that exposes its integer primary key as the event id
    keeps working unchanged.

    Anything else cannot carry a total order — an unset id, or the opaque
    hex of :func:`agent_core.types.new_event_id`, whose values are neither
    numeric nor monotonic. Those are rejected with an actionable
    :class:`EventStoreContractError` rather than the bare ``ValueError``
    ``int()`` would raise, because the fix is a store change and the
    message has to say so.
    """
    if event.seq is not None:
        return event.seq
    raw = str(event.id)
    if raw.isdecimal():
        return int(raw)
    raise EventStoreContractError(
        "Event store must stamp every appended event with a total-order "
        "ordinal: set KernelEvent.seq to a monotonically increasing integer "
        "(a decimal KernelEvent.id is also accepted). Got "
        f"id={raw!r}, seq=None — an opaque id such as new_event_id() cannot "
        "order a durable cursor, so AgentComm.consume cannot advance safely."
    )


@runtime_checkable
class _AgentCommEventStore(EventReader, Protocol):
    """AgentComm needs a whole-event append plus cursored reads
    (:class:`~agent_core.protocols.EventReader`) — combined here because
    Python lacks an intersection type. Public Protocols stay minimal in
    ``core.protocols``.

    ``append`` takes a :class:`KernelEvent`, *not* the flattened
    ``(task_id, event_type, payload, agent_role)`` of
    :class:`~agent_core.protocols.EventSink`: an inter-agent message carries
    ``from_agent`` / ``to_agent`` / ``message_type`` / ``correlation_id`` as
    first-class fields that the reader queries on, and ``EventSink`` cannot
    express them. A host that implements only ``EventSink`` is a valid
    telemetry sink for ``AgentBus`` but is *not* a valid AgentComm store.

    The combination has a stronger semantic contract than either half:
    ``append`` retains the event and returns it stamped with a total-order
    ordinal (see :func:`event_ordinal`), and sequential awaited appends
    receive increasing ordinals. That ordinal — not ``timestamp`` — is the
    total order, and ``EventReader``'s ``after_id`` cursor is expressed in
    it. Concurrent appends have no defined relative order, only that every
    reader observes the same resulting order.

    Ordering is not guaranteed across live writer processes sharing one
    JSONL path. Nor is losslessness guaranteed for a capacity-bounded
    in-memory store, which evicts its oldest retained events.
    """

    async def append(self, event: KernelEvent) -> KernelEvent: ...


class DeliveryMode(StrEnum):
    """Message delivery mode.

    TRIGGER: persist + hot queue + EventBus broadcast.
        Use when receiver must act immediately (e.g., assertion → critic).
    QUEUE: persist + hot queue only (no broadcast).
        Use for status updates the receiver pulls when ready.
    """

    TRIGGER = "trigger"
    QUEUE = "queue"


class AgentComm:
    """Sends inter-agent messages and logs them as events.

    All messages are persisted to EventStore (truth source).
    Hot queues are optional in-memory acceleration.
    DeliveryMode controls whether EventBus broadcast fires.
    """

    def __init__(
        self,
        event_store: _AgentCommEventStore,
        event_bus: EventBus,
    ) -> None:
        self._event_store = event_store
        self._event_bus = event_bus
        # Each message type gets an independent cursor. A specialized
        # consumer (for example AgentBus result recovery) must not advance
        # the generic inbox cursor or consume status/control messages that
        # belong to another component.
        self._cursors: dict[tuple[str, str | None, str | None], int] = {}
        self._hot_queues: dict[str, asyncio.Queue[KernelEvent]] = {}

    async def send(
        self,
        msg: AgentMessage,
        mode: DeliveryMode = DeliveryMode.QUEUE,
    ) -> KernelEvent:
        """Send an agent message.

        Always persists to EventStore (truth source).
        TRIGGER mode additionally broadcasts via EventBus for immediate wakeup.
        """
        event = KernelEvent(
            task_id=TaskId(msg.task_id),
            event_type=EventType.AGENT_MESSAGE,
            from_agent=msg.from_agent,
            to_agent=msg.to_agent,
            message_type=msg.message_type,
            correlation_id=msg.content.get("correlation_id"),
            payload={
                "message_id": msg.id,
                "from_agent": msg.from_agent,
                "to_agent": msg.to_agent,
                "message_type": msg.message_type,
                "content": msg.content,
                "parent_id": msg.parent_id,
                "correlation_id": msg.content.get("correlation_id"),
                "delivery_mode": mode.value,
            },
        )

        # 1. Persist (always — EventStore is truth source)
        persisted = await self._event_store.append(event)

        # 2. Hot queue (always — non-durable acceleration)
        queue = self._hot_queues.get(msg.to_agent)
        if queue is not None:
            try:
                queue.put_nowait(persisted)
            except asyncio.QueueFull:
                logger.debug(
                    "Hot queue for %s is full; consumer uses EventStore",
                    msg.to_agent,
                )

        # 3. Broadcast (TRIGGER only — immediate wakeup signal)
        if mode == DeliveryMode.TRIGGER:
            await self._event_bus.publish(
                event.event_type, event.payload,
            )

        logger.debug(
            "Agent message %s: %s → %s [%s] mode=%s",
            msg.id, msg.from_agent, msg.to_agent,
            msg.message_type, mode.value,
        )
        return persisted

    async def consume(
        self,
        agent_id: str,
        *,
        task_id: str | None = None,
        limit: int = 50,
        message_type: str | None = None,
    ) -> list[KernelEvent]:
        """Consume undelivered messages for an agent using an idempotent cursor.

        Cursor is per ``(agent_id, task_id, message_type)``. Typed consumers
        therefore cannot skip or discard unrelated inbox traffic. Restarting
        from 0 replays all messages in the selected stream.

        Ordering (L2-2): messages come back in send order within a task.
        That rests entirely on the stronger ``_AgentCommEventStore``
        contract above and the "Ordering" section of
        ``core.protocols.EventReader``. Three consequences callers rely on:

        * a sender that awaits each ``send`` sees its own messages here
          in exactly that order — never reordered, never with a later
          message ahead of an earlier one;
        * the cursor only moves forward, so each message reaches this
          ``(agent, task, message_type)`` reader exactly once even when
          ``limit`` truncates the batch — the remainder arrives on the
          next call, still in order;
        * ``message_type`` is part of the cursor key on purpose: a
          shared cursor would let one consumer's read carry another
          consumer's mail past unread.

        ``tests/test_agent_comm_store_contract.py`` pins all three, plus
        the ordinal requirement itself, against a reference store.
        """
        cursor_key = (agent_id, task_id, message_type)
        after_id = self._cursors.get(cursor_key, 0)
        # Preserve compatibility with third-party EventReader implementations
        # written before typed inbox queries existed.
        if message_type is None:
            events = await self._event_store.get_events_for_agent(
                agent_id,
                after_id=after_id,
                limit=limit,
                task_id=task_id,
            )
        else:
            events = await self._event_store.get_events_for_agent(
                agent_id,
                after_id=after_id,
                limit=limit,
                task_id=task_id,
                message_type=message_type,
            )
        if events:
            ids = [event_ordinal(event) for event in events]
            if any(current <= previous for previous, current in itertools.pairwise(ids)):
                # Advancing to max(ids) can permanently skip lower ids that
                # a misordered, limited query left outside this batch. Keep
                # the cursor untouched and fail loudly instead of returning
                # a partial stream with silent data loss.
                logger.error(
                    "Event store returned %d events for %s outside strict "
                    "ascending id order (%s); cursor remains at %d",
                    len(ids), agent_id, ids, after_id,
                )
                raise RuntimeError(
                    "EventReader must return events in strictly ascending id order"
                )
            self._cursors[cursor_key] = ids[-1]
        return events

    def reset_cursor(
        self,
        agent_id: str,
        task_id: str | None = None,
        *,
        message_type: str | None = None,
    ) -> None:
        """Reset one typed cursor, or every cursor for an agent/task pair."""
        if message_type is not None:
            self._cursors.pop((agent_id, task_id, message_type), None)
            return
        for key in list(self._cursors):
            if key[:2] == (agent_id, task_id):
                self._cursors.pop(key, None)

    def hot_queue_for(
        self, agent_id: str,
    ) -> asyncio.Queue[KernelEvent]:
        """Return the non-durable hot queue for an agent."""
        return self._hot_queues.setdefault(
            agent_id, asyncio.Queue(maxsize=256),
        )
