"""Journal event helpers for durable notes."""

from __future__ import annotations

from datetime import datetime
from typing import Any, cast

from agent_core.context.models import JournalEntry, Note

NOTE_UPSERTED = "note.upserted"
NOTE_DELETED = "note.deleted"


def note_payload(note: Note) -> dict[str, Any]:
    return {
        "note_id": note.note_id,
        "text": note.text,
        "metadata": dict(note.metadata),
        "updated_at": note.updated_at.isoformat(),
    }


def project_notes(entries: list[JournalEntry]) -> list[Note]:
    notes: dict[str, Note] = {}
    for entry in entries:
        note_id = str(entry.payload.get("note_id") or "")
        if not note_id:
            continue
        if entry.kind == NOTE_DELETED:
            notes.pop(note_id, None)
        elif entry.kind == NOTE_UPSERTED:
            metadata = entry.payload.get("metadata")
            notes[note_id] = Note(
                note_id=note_id,
                text=str(entry.payload.get("text") or ""),
                metadata=(
                    dict(cast("dict[str, Any]", metadata)) if isinstance(metadata, dict) else {}
                ),
                updated_at=datetime.fromisoformat(str(entry.payload["updated_at"])),
            )
    return sorted(notes.values(), key=lambda note: (note.updated_at, note.note_id))


__all__ = ["NOTE_DELETED", "NOTE_UPSERTED", "note_payload", "project_notes"]
