"""JSON-safe models for the run-local context journal."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class ContextScope:
    """Product-assigned journal isolation keys."""

    session_id: str
    run_id: str
    agent_id: str = ""

    def __post_init__(self) -> None:
        if not self.session_id or not self.run_id:
            raise ValueError("session_id and run_id must be non-empty")


@dataclass(frozen=True)
class BlobRef:
    digest: str
    size: int
    media_type: str = "application/octet-stream"


@dataclass(frozen=True)
class JournalEntry:
    scope: ContextScope
    kind: str
    payload: dict[str, Any] = field(default_factory=dict[str, Any])
    blobs: dict[str, BlobRef] = field(default_factory=dict[str, BlobRef])
    entry_id: str = field(default_factory=lambda: uuid4().hex)
    sequence: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("journal entry kind must be non-empty")


@dataclass(frozen=True)
class Note:
    note_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


__all__ = ["BlobRef", "ContextScope", "JournalEntry", "Note"]
