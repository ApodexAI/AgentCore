"""Pure journal projection helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from agent_core.context.models import JournalEntry


def fold_entries[T](
    entries: Iterable[JournalEntry],
    initial: T,
    reducer: Callable[[T, JournalEntry], T],
) -> T:
    """Replay entries deterministically into a caller-owned projection."""
    state = initial
    for entry in entries:
        state = reducer(state, entry)
    return state


def latest_by_kind(entries: Iterable[JournalEntry]) -> dict[str, JournalEntry]:
    """Return the highest-sequence entry for every event kind."""
    latest: dict[str, JournalEntry] = {}
    for entry in entries:
        previous = latest.get(entry.kind)
        if previous is None or entry.sequence > previous.sequence:
            latest[entry.kind] = entry
    return latest


def bounded_tail(
    entries: Iterable[JournalEntry],
    *,
    max_entries: int,
) -> list[JournalEntry]:
    """Select the newest entries while preserving chronological order."""
    if max_entries <= 0:
        return []
    ordered = sorted(entries, key=lambda entry: entry.sequence)
    return ordered[-max_entries:]


__all__ = ["bounded_tail", "fold_entries", "latest_by_kind"]
