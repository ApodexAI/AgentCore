from __future__ import annotations

import stat

import pytest

from agent_core.context import ContextManager, ContextScope, FileBlobStore, SQLiteJournalStore
from agent_core.context.projection import bounded_tail, latest_by_kind


@pytest.mark.asyncio
async def test_journal_round_trip_scope_isolation_and_sequence(tmp_path) -> None:
    manager = ContextManager(
        SQLiteJournalStore(tmp_path / "journal.db"),
        FileBlobStore(tmp_path / "blobs"),
    )
    first = ContextScope("session", "run-a", "agent")
    second = ContextScope("session", "run-b", "agent")
    one = await manager.append(first, "turn", {"value": 1})
    two = await manager.append(first, "turn", {"value": 2})
    await manager.append(second, "turn", {"value": 3})

    history = await manager.history(first)
    assert [entry.payload["value"] for entry in history] == [1, 2]
    assert one.sequence < two.sequence
    assert await manager.history(first, after_sequence=one.sequence) == [two]


@pytest.mark.asyncio
async def test_blobs_are_content_addressed_verified_and_reference_counted(tmp_path) -> None:
    blob_store = FileBlobStore(tmp_path / "blobs")
    manager = ContextManager(SQLiteJournalStore(tmp_path / "journal.db"), blob_store)
    first = ContextScope("session", "run-a")
    second = ContextScope("session", "run-b")
    entry_a = await manager.append(first, "artifact", attachments={"report": b"same"})
    entry_b = await manager.append(second, "artifact", attachments={"report": b"same"})
    ref = entry_a.blobs["report"]
    assert entry_b.blobs["report"] == ref
    assert await blob_store.get(ref) == b"same"
    mode = blob_store._path(ref.digest).stat().st_mode
    assert mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) == 0

    await manager.delete_scope(first)
    assert await blob_store.get(ref) == b"same"
    await manager.delete_scope(second)
    assert not blob_store._path(ref.digest).exists()


@pytest.mark.asyncio
async def test_notes_are_event_sourced(tmp_path) -> None:
    manager = ContextManager(
        SQLiteJournalStore(tmp_path / "journal.db"), FileBlobStore(tmp_path / "blobs")
    )
    scope = ContextScope("session", "run")
    await manager.upsert_note(scope, "decision", "first", metadata={"source": "tool"})
    await manager.upsert_note(scope, "decision", "revised")
    await manager.upsert_note(scope, "keep", "visible")
    await manager.delete_note(scope, "decision")
    notes = await manager.notes(scope)
    assert [(note.note_id, note.text) for note in notes] == [("keep", "visible")]


@pytest.mark.asyncio
async def test_projection_helpers_are_deterministic(tmp_path) -> None:
    manager = ContextManager(
        SQLiteJournalStore(tmp_path / "journal.db"), FileBlobStore(tmp_path / "blobs")
    )
    scope = ContextScope("session", "run")
    for kind in ("a", "b", "a"):
        await manager.append(scope, kind)
    entries = await manager.history(scope)
    assert latest_by_kind(entries)["a"] == entries[-1]
    assert bounded_tail(entries, max_entries=2) == entries[-2:]
