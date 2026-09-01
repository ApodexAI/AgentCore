"""SQLite implementation of the append-only context journal."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from agent_core.context.models import BlobRef, ContextScope, JournalEntry

_SCHEMA = """
CREATE TABLE IF NOT EXISTS context_journal (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id TEXT NOT NULL UNIQUE,
    session_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    blobs_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_context_scope_sequence
ON context_journal(session_id, run_id, agent_id, sequence);
CREATE TABLE IF NOT EXISTS context_journal_blobs (
    entry_id TEXT NOT NULL,
    digest TEXT NOT NULL,
    PRIMARY KEY(entry_id, digest),
    FOREIGN KEY(entry_id) REFERENCES context_journal(entry_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_context_blob_digest
ON context_journal_blobs(digest);
"""


class SQLiteJournalStore:
    """Concurrent-safe SQLite journal using one short connection per operation."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self._setup_lock = asyncio.Lock()
        self._ready = False

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    async def setup(self) -> None:
        if self._ready:
            return
        async with self._setup_lock:
            if self._ready:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(self._setup_sync)
            self._ready = True

    def _setup_sync(self) -> None:
        with self._connect() as connection:
            connection.executescript(_SCHEMA)

    async def append(self, entry: JournalEntry) -> JournalEntry:
        await self.setup()
        sequence = await asyncio.to_thread(self._append_sync, entry)
        return JournalEntry(
            scope=entry.scope,
            kind=entry.kind,
            payload=dict(entry.payload),
            blobs=dict(entry.blobs),
            entry_id=entry.entry_id,
            sequence=sequence,
            created_at=entry.created_at,
        )

    def _append_sync(self, entry: JournalEntry) -> int:
        blobs_json = json.dumps(
            {
                name: {
                    "digest": ref.digest,
                    "size": ref.size,
                    "media_type": ref.media_type,
                }
                for name, ref in entry.blobs.items()
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._connect() as connection:
            cursor = connection.execute(
                """INSERT INTO context_journal
                (entry_id, session_id, run_id, agent_id, kind, payload_json, blobs_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.entry_id,
                    entry.scope.session_id,
                    entry.scope.run_id,
                    entry.scope.agent_id,
                    entry.kind,
                    json.dumps(entry.payload, sort_keys=True, separators=(",", ":")),
                    blobs_json,
                    entry.created_at.isoformat(),
                ),
            )
            connection.executemany(
                "INSERT INTO context_journal_blobs(entry_id, digest) VALUES (?, ?)",
                ((entry.entry_id, ref.digest) for ref in entry.blobs.values()),
            )
            sequence = cursor.lastrowid
        if sequence is None:
            raise RuntimeError("SQLite did not assign a journal sequence")
        return int(sequence)

    async def list_entries(
        self,
        scope: ContextScope,
        *,
        after_sequence: int = 0,
        kinds: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[JournalEntry]:
        await self.setup()
        return await asyncio.to_thread(
            self._list_sync, scope, after_sequence, tuple(kinds or ()), limit
        )

    def _list_sync(
        self,
        scope: ContextScope,
        after_sequence: int,
        kinds: tuple[str, ...],
        limit: int | None,
    ) -> list[JournalEntry]:
        clauses = ["session_id = ?", "run_id = ?", "agent_id = ?", "sequence > ?"]
        params: list[object] = [scope.session_id, scope.run_id, scope.agent_id, after_sequence]
        if kinds:
            clauses.append("kind IN (" + ",".join("?" for _ in kinds) + ")")
            params.extend(kinds)
        sql = "SELECT * FROM context_journal WHERE " + " AND ".join(clauses)
        sql += " ORDER BY sequence ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(0, limit))
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [self._decode(row) for row in rows]

    @staticmethod
    def _decode(row: sqlite3.Row) -> JournalEntry:
        payload = cast("dict[str, Any]", json.loads(str(row["payload_json"])))
        raw_blobs = cast("dict[str, dict[str, Any]]", json.loads(str(row["blobs_json"])))
        blobs = {
            name: BlobRef(
                digest=str(value["digest"]),
                size=int(value["size"]),
                media_type=str(value.get("media_type") or "application/octet-stream"),
            )
            for name, value in raw_blobs.items()
        }
        return JournalEntry(
            scope=ContextScope(
                session_id=str(row["session_id"]),
                run_id=str(row["run_id"]),
                agent_id=str(row["agent_id"]),
            ),
            kind=str(row["kind"]),
            payload=payload,
            blobs=blobs,
            entry_id=str(row["entry_id"]),
            sequence=int(row["sequence"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
        )

    async def delete_scope(self, scope: ContextScope) -> set[str]:
        await self.setup()
        return await asyncio.to_thread(self._delete_scope_sync, scope)

    def _delete_scope_sync(self, scope: ContextScope) -> set[str]:
        where = "session_id = ? AND run_id = ? AND agent_id = ?"
        params = (scope.session_id, scope.run_id, scope.agent_id)
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT DISTINCT digest FROM context_journal_blobs
                WHERE entry_id IN (SELECT entry_id FROM context_journal WHERE """
                + where
                + ")",
                params,
            ).fetchall()
            candidates = {str(row["digest"]) for row in rows}
            connection.execute("DELETE FROM context_journal WHERE " + where, params)
            unreferenced = {
                digest
                for digest in candidates
                if connection.execute(
                    "SELECT 1 FROM context_journal_blobs WHERE digest = ? LIMIT 1",
                    (digest,),
                ).fetchone()
                is None
            }
        return unreferenced


__all__ = ["SQLiteJournalStore"]
