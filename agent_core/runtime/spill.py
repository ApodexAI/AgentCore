"""Session-isolated, content-addressed storage for oversized tool results.

The store owns persistence and preview shaping. Products own where the root is
mounted, which path is visible to an agent sandbox, and when a session ends.
No product execution context, tool registry, or sandbox implementation is
imported here.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, Literal

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_AGGREGATE_RESULT_CHARS",
    "SpillStore",
    "TruncationMode",
    "budgeted_preview",
    "scope_component",
    "truncate_preview",
]

TruncationMode = Literal["middle", "head"]

DEFAULT_AGGREGATE_RESULT_CHARS = 200_000
_MIN_AGGREGATE_KEEP = 2_000
_MIN_PREVIEW_CHARS = 500
_MIN_SIDE_CHARS = 200
_SPILL_MIN_CHARS = 1_500
_SPILL_SEPARATOR = "\n---\n\n"


def scope_component(scope: str) -> str:
    """Map an arbitrary session identifier to one safe, stable component."""

    return hashlib.sha256(scope.encode("utf-8")).hexdigest()[:16] if scope else ""


def _elision(removed: int) -> str:
    return f"\n… {removed:,} chars elided …\n"


def _head_end(text: str, budget: int) -> int:
    if budget >= len(text):
        return len(text)
    newline = text.rfind("\n", budget // 2, budget)
    return newline if newline > 0 else budget


def _tail_start(text: str, budget: int) -> int:
    start = max(len(text) - budget, 0)
    if start == 0:
        return 0
    newline = text.find("\n", start, start + max(budget // 2, 1))
    return newline + 1 if newline != -1 else start


def truncate_preview(
    text: str,
    budget: int,
    *,
    mode: TruncationMode = "middle",
) -> str:
    """Return a bounded head-only or head-and-tail preview."""

    if budget <= 0:
        return ""
    if len(text) <= budget:
        return text
    if mode == "head":
        return text[: _head_end(text, budget)]

    room = budget - len(_elision(len(text)))
    if room < 2 * _MIN_SIDE_CHARS:
        return text[: _head_end(text, budget)]
    head_end = _head_end(text, room // 2)
    tail_start = _tail_start(text, room - room // 2)
    if tail_start <= head_end:
        return text[: _head_end(text, budget)]
    return text[:head_end] + _elision(tail_start - head_end) + text[tail_start:]


def _spill_footer(ref: str, *, full_len: int, note: str = "") -> str:
    suffix = f" {note}" if note else ""
    if not ref:
        return (
            f"\n\n[... only part of this {full_len:,}-char result is shown; the "
            f"remainder is not readable from this backend.{suffix}]"
        )
    directory = ref.rsplit("/", 1)[0]
    return (
        f"\n\n[... only part of this {full_len:,}-char result is shown (head and "
        f"tail; the gap is marked above). Full content is saved read-only at "
        f"{ref}. Only if the elided middle is required, read that path with "
        f"whichever tool you have — read_file, `cat` via bash, or "
        f'grep_search(pattern="...", path="{directory}"). It is read-only; '
        f"do not write there.{suffix}]"
    )


def budgeted_preview(
    body: str,
    *,
    cap: int,
    ref: str,
    full_len: int | None = None,
    note: str = "",
    mode: TruncationMode = "middle",
) -> str:
    """Build a preview plus recovery pointer within one character budget."""

    total = len(body) if full_len is None else full_len
    footer = _spill_footer(ref, full_len=total, note=note)
    preview_budget = max(cap - len(footer), _MIN_PREVIEW_CHARS)
    return truncate_preview(body, preview_budget, mode=mode) + footer


class SpillStore:
    """One conversation's read-only recovery store.

    ``root`` is the physical host directory containing all scoped stores.
    ``visible_root`` is the path exposed inside the agent runtime, such as
    ``/spill``. Passing ``None`` keeps persistence available to the host but
    avoids advertising an unreadable path to the model.
    """

    _created_stores: ClassVar[set[Path]] = set()

    def __init__(
        self,
        root: str | Path,
        scope: str,
        *,
        visible_root: str | None = None,
        document_builder: Callable[[str, str, str], str] | None = None,
    ) -> None:
        if not scope:
            raise ValueError("spill scope must not be empty")
        self.root = Path(root).expanduser().resolve()
        self.scope = scope
        self.component = scope_component(scope)
        self.directory = self.root / self.component
        self.visible_root = visible_root.rstrip("/") if visible_root else None
        self._document_builder = document_builder or self._document

    @property
    def visible_directory(self) -> str:
        if self.visible_root is None:
            return ""
        return f"{self.visible_root}/{self.component}"

    def _ensure_directory(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.directory.is_symlink():
            raise OSError(f"spill scope directory must not be a symlink: {self.directory}")
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            self.directory.resolve().relative_to(self.root)
        except ValueError as exc:
            raise OSError("spill scope escaped its configured root") from exc
        with contextlib.suppress(OSError):
            self.directory.chmod(0o755)
        self._created_stores.add(self.directory)

    def ensure(self) -> Path:
        """Create this scope's store and return its physical directory."""

        self._ensure_directory()
        return self.directory

    @staticmethod
    def _document(tool_name: str, spill_id: str, result: str) -> str:
        return (
            f"# {tool_name} — spilled tool result\n\n"
            f"- id: `{spill_id}`\n"
            f"- captured: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n"
            f"- length: {len(result):,} chars"
            f"{_SPILL_SEPARATOR}{result}"
        )

    def write(
        self,
        tool_name: str,
        body: str,
        *,
        require_visible: bool = False,
    ) -> tuple[Path, str] | None:
        """Persist one body atomically and return physical and visible paths."""

        if require_visible and self.visible_root is None:
            return None
        self._ensure_directory()
        digest = hashlib.sha256(
            tool_name.encode("utf-8") + b"\x00" + body.encode("utf-8", "replace"),
        ).hexdigest()[:16]
        path = self.directory / f"{digest}.md"
        if path.is_symlink():
            logger.warning("Refusing symlink at spill destination %s", path)
            return None
        if not path.exists():
            tmp = path.with_name(f".{digest}.{uuid.uuid4().hex[:8]}.tmp")
            try:
                tmp.write_text(
                    self._document_builder(tool_name, digest, body),
                    encoding="utf-8",
                )
                os.replace(tmp, path)
                with contextlib.suppress(OSError):
                    path.chmod(0o444)
            except OSError as exc:
                logger.warning("Failed to spill %s result: %s", tool_name, exc)
                with contextlib.suppress(OSError):
                    tmp.unlink()
                return None
        ref = f"{self.visible_directory}/{path.name}" if self.visible_directory else ""
        return path, ref

    def spill_compacted_body(
        self,
        tool_name: str,
        body: str,
        *,
        min_chars: int = _SPILL_MIN_CHARS,
    ) -> str | None:
        """Persist a body before compaction removes its inline representation."""

        if len(body) < min_chars:
            return None
        spilled = self.write(tool_name, body, require_visible=True)
        return spilled[1] if spilled else None

    def overflow(
        self,
        tool_name: str,
        result: str,
        *,
        cap: int,
        mode: TruncationMode = "middle",
    ) -> str:
        """Spill and preview one result when it exceeds ``cap``."""

        if cap <= 0 or len(result) <= cap:
            return result
        spilled = self.write(tool_name, result)
        ref = spilled[1] if spilled else ""
        return budgeted_preview(result, cap=cap, ref=ref, mode=mode)

    def enforce_aggregate_budget(
        self,
        results: list[str],
        *,
        tool_names: list[str] | None = None,
        cap: int = DEFAULT_AGGREGATE_RESULT_CHARS,
        mode: TruncationMode = "middle",
    ) -> list[str]:
        """Bound all results in one turn, spilling every shortened body first."""

        total = sum(len(result) for result in results)
        if cap <= 0 or total <= cap:
            return results
        names = list(tool_names or [])
        adjusted = list(results)
        for idx, result in sorted(
            enumerate(results),
            key=lambda item: len(item[1]),
            reverse=True,
        ):
            if total <= cap:
                break
            excess = total - cap
            result_cap = max(_MIN_AGGREGATE_KEEP, len(result) - excess)
            if result_cap >= len(result):
                continue
            name = names[idx] if idx < len(names) else "tool"
            spilled = (
                self.write(name, result, require_visible=True)
                if len(result) >= _SPILL_MIN_CHARS
                else None
            )
            replacement = budgeted_preview(
                result,
                cap=result_cap,
                ref=spilled[1] if spilled else "",
                note="Cut further to fit the per-turn tool-result budget.",
                mode=mode,
            )
            total -= len(result) - len(replacement)
            adjusted[idx] = replacement
        return adjusted

    def _physical_path(self, ref: str | Path) -> Path | None:
        raw = str(ref)
        if self.visible_directory and (
            raw == self.visible_directory or raw.startswith(self.visible_directory + "/")
        ):
            relative = raw[len(self.visible_directory) :].lstrip("/")
            candidate = self.directory / relative
        else:
            candidate = Path(raw)
        try:
            resolved = candidate.resolve()
            resolved.relative_to(self.directory.resolve())
        except (OSError, ValueError):
            return None
        return resolved

    def contains_path(self, path: str | Path) -> bool:
        """Whether a physical or agent-visible path belongs to this store."""

        return self._physical_path(path) is not None

    def read(self, ref: str | Path) -> str | None:
        """Read the captured body, rejecting paths outside this scope."""

        path = self._physical_path(ref)
        if path is None or not path.is_file():
            return None
        return self._read_path(path)

    @staticmethod
    def _read_path(path: Path) -> str | None:
        try:
            text = path.read_text(encoding="utf-8")
            if _SPILL_SEPARATOR in text:
                return text.split(_SPILL_SEPARATOR, 1)[1]
            data = json.loads(text)
            content = data.get("content")
            return str(content) if content is not None else None
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Failed to read spill file %s: %s", path, exc)
            return None

    @classmethod
    def read_created(cls, path: str | Path) -> str | None:
        """Read a physical file only when its store was created by this process."""

        try:
            resolved = Path(path).resolve()
        except OSError:
            return None
        if resolved.parent not in cls._created_stores or not resolved.is_file():
            return None
        return cls._read_path(resolved)

    def body_names_file(self, body: str) -> bool:
        roots = [str(self.directory), self.visible_directory]
        return any(root and f"{root.rstrip('/')}/" in body for root in roots)

    def cleanup(self) -> int:
        """Remove files created for this scope, never traversing subdirectories."""

        if not self.directory.is_dir():
            return 0
        count = 0
        for entry in self.directory.iterdir():
            if not entry.is_file() and not entry.is_symlink():
                continue
            try:
                entry.unlink()
                count += 1
            except OSError:
                pass
        with contextlib.suppress(OSError):
            self.directory.rmdir()
        self._created_stores.discard(self.directory)
        return count

    @classmethod
    def cleanup_process(cls) -> int:
        """Remove only stores created by this process."""

        count = 0
        for directory in list(cls._created_stores):
            if directory.is_dir():
                for entry in directory.iterdir():
                    if not entry.is_file() and not entry.is_symlink():
                        continue
                    with contextlib.suppress(OSError):
                        entry.unlink()
                        count += 1
                with contextlib.suppress(OSError):
                    directory.rmdir()
            cls._created_stores.discard(directory)
        return count
