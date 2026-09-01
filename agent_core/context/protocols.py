"""Storage contracts for durable context."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from agent_core.context.models import BlobRef, ContextScope, JournalEntry


@runtime_checkable
class BlobStore(Protocol):
    async def put(
        self, data: bytes, *, media_type: str = "application/octet-stream"
    ) -> BlobRef: ...

    async def get(self, ref: BlobRef) -> bytes: ...

    async def delete(self, digest: str) -> None: ...


@runtime_checkable
class JournalStore(Protocol):
    async def append(self, entry: JournalEntry) -> JournalEntry: ...

    async def list_entries(
        self,
        scope: ContextScope,
        *,
        after_sequence: int = 0,
        kinds: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[JournalEntry]: ...

    async def delete_scope(self, scope: ContextScope) -> set[str]:
        """Delete a scope and return blob digests no longer referenced."""
        ...


__all__ = ["BlobStore", "JournalStore"]
