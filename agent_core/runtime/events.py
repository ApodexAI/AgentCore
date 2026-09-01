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
    """Small failure-isolated async event bus."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._global_handlers: list[Handler] = []

    def subscribe(self, event_type: str | Enum, handler: Handler) -> None:
        self._handlers[_coerce_key(event_type)].append(handler)

    def subscribe_all(self, handler: Handler) -> None:
        self._global_handlers.append(handler)

    async def publish(self, event_type: str | Enum, payload: dict[str, Any]) -> None:
        key = _coerce_key(event_type)
        handlers = [*self._handlers.get(key, ()), *self._global_handlers]
        if not handlers:
            return
        event_data = {"event_type": key, **payload}
        await asyncio.gather(*(self._safe_call(handler, event_data) for handler in handlers))

    @staticmethod
    async def _safe_call(handler: Handler, data: dict[str, Any]) -> None:
        try:
            await handler(data)
        except Exception:
            logger.exception("Event handler failed: %s", getattr(handler, "__name__", repr(handler)))


__all__ = ["EventBus", "Handler"]
