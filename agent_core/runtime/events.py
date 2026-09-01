"""Async pub/sub for portable runtime events."""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

Handler = Callable[[dict[str, Any]], Awaitable[None]]


def _coerce_key(event_type: str | Enum) -> str:
    if isinstance(event_type, Enum):
        value = getattr(event_type, "value", None)
        if isinstance(value, str):
            return value
        return str(value)
    return str(event_type)


class EventBus:
    """Small failure-isolated async event bus.

    Subscriptions are held strongly and never expire on their own. A bus shared
    across tasks must be paired with :meth:`unsubscribe` (or :meth:`clear`) when
    a subscriber goes away, or every finished task leaves its handler — and the
    state that handler closes over — reachable forever.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._global_handlers: list[Handler] = []

    def subscribe(self, event_type: str | Enum, handler: Handler) -> None:
        self._handlers[_coerce_key(event_type)].append(handler)

    def subscribe_all(self, handler: Handler) -> None:
        self._global_handlers.append(handler)

    def unsubscribe(self, event_type: str | Enum, handler: Handler) -> bool:
        """Remove one typed subscription. Returns whether it was registered."""
        key = _coerce_key(event_type)
        registered = self._handlers.get(key)
        if not registered:
            return False
        try:
            registered.remove(handler)
        except ValueError:
            return False
        if not registered:
            del self._handlers[key]
        return True

    def unsubscribe_all(self, handler: Handler) -> bool:
        """Remove one catch-all subscription. Returns whether it was registered."""
        try:
            self._global_handlers.remove(handler)
        except ValueError:
            return False
        return True

    def clear(self) -> None:
        """Drop every subscription."""
        self._handlers.clear()
        self._global_handlers.clear()

    async def publish(self, event_type: str | Enum, payload: dict[str, Any]) -> None:
        key = _coerce_key(event_type)
        handlers = [*self._handlers.get(key, ()), *self._global_handlers]
        if not handlers:
            return
        if "event_type" in payload and payload["event_type"] != key:
            logger.warning(
                "Event payload carried event_type=%r for a %r publish; the published type wins.",
                payload["event_type"],
                key,
            )
        # The published type is authoritative, so it is applied last.
        event_data = {**payload, "event_type": key}
        # Each handler gets its own top-level copy: handlers run concurrently
        # and interleave at every await, so a handler that mutates the payload
        # must not change what its peers observe.
        await asyncio.gather(*(self._safe_call(handler, dict(event_data)) for handler in handlers))

    @staticmethod
    async def _safe_call(handler: Handler, data: dict[str, Any]) -> None:
        try:
            await handler(data)
        except Exception:
            logger.exception("Event handler failed: %s", getattr(handler, "__name__", repr(handler)))


__all__ = ["EventBus", "Handler"]
