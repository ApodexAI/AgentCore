"""High-level durable context orchestration."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from agent_core.context.models import BlobRef, ContextScope, JournalEntry, Note
from agent_core.context.notes import NOTE_DELETED, NOTE_UPSERTED, note_payload, project_notes
from agent_core.context.protocols import BlobStore, JournalStore


class ContextManager:
    """Compose a journal and blob store without owning product lifecycle policy."""

    def __init__(self, journal: JournalStore, blobs: BlobStore) -> None:
        self.journal = journal
        self.blobs = blobs

    async def append(
        self,
        scope: ContextScope,
        kind: str,
        payload: dict[str, Any] | None = None,
        *,
        attachments: dict[str, bytes] | None = None,
        media_types: dict[str, str] | None = None,
    ) -> JournalEntry:
        refs: dict[str, BlobRef] = {}
        for name, data in (attachments or {}).items():
            media_type = (media_types or {}).get(name, "application/octet-stream")
            refs[name] = await self.blobs.put(data, media_type=media_type)
        return await self.journal.append(
            JournalEntry(scope=scope, kind=kind, payload=dict(payload or {}), blobs=refs)
        )

    async def history(
        self,
        scope: ContextScope,
        *,
        after_sequence: int = 0,
        kinds: tuple[str, ...] | None = None,
        limit: int | None = None,
    ) -> list[JournalEntry]:
        return await self.journal.list_entries(
            scope,
            after_sequence=after_sequence,
            kinds=kinds,
            limit=limit,
        )

    async def upsert_note(
        self,
        scope: ContextScope,
        note_id: str,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> JournalEntry:
        note = Note(
            note_id=note_id,
            text=text,
            metadata=dict(metadata or {}),
            updated_at=datetime.now(UTC),
        )
        return await self.append(scope, NOTE_UPSERTED, note_payload(note))

    async def delete_note(self, scope: ContextScope, note_id: str) -> JournalEntry:
        return await self.append(scope, NOTE_DELETED, {"note_id": note_id})

    async def notes(self, scope: ContextScope) -> list[Note]:
        entries = await self.history(scope, kinds=(NOTE_UPSERTED, NOTE_DELETED))
        return project_notes(entries)

    async def delete_scope(self, scope: ContextScope) -> None:
        unreferenced = await self.journal.delete_scope(scope)
        for digest in unreferenced:
            await self.blobs.delete(digest)


__all__ = ["ContextManager"]
