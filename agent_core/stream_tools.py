"""Shared wire-format assembly for streamed native tool calls."""
from __future__ import annotations

from typing import Any

from agent_core.messages import ToolCall


class StreamToolCalls:
    def __init__(self) -> None:
        self._slots: dict[int, ToolCall] = {}

    def feed(self, deltas: list[dict[str, Any]]) -> None:
        for delta in deltas:
            index = delta.get("index") or 0
            slot = self._slots.setdefault(index, {
                "id": "", "type": "function", "function": {"name": "", "arguments": ""},
            })
            if delta.get("id"):
                slot["id"] = delta["id"]
            if delta.get("name"):
                slot["function"]["name"] += delta["name"]
            if delta.get("arguments"):
                slot["function"]["arguments"] += delta["arguments"]

    def complete(self) -> list[ToolCall]:
        # Keep provider order by index; unexecutable nameless slots never reach
        # history or middleware fingerprints. Argument validation stays with
        # the caller's existing repair/parser path.
        return [self._slots[i] for i in sorted(self._slots) if self._slots[i]["function"]["name"]]

    @property
    def dropped_count(self) -> int:
        return sum(not slot["function"]["name"] for slot in self._slots.values())
