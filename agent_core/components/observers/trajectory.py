# pyright: reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportMissingTypeArgument=false, reportUnnecessaryIsInstance=false
"""Write workflow-neutral trajectories in JSON and/or JSONL formats.

JSON snapshots are atomically replaced; JSONL events are flushed incrementally
so partial runs remain readable without retaining full payloads in memory.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import IO, Any, Literal

from agent_core.loop_types import (
    AgentLoopResult,
    BaseObserver,
    CompactionEvent,
    ContextCompactionContext,
    Intervention,
    LoopConfig,
    ToolResult,
    TurnContext,
)
from agent_core.messages import Message
from agent_core.runtime.loop.compact import COMPACTION_SEQ_KEY
from agent_core.tool_content import redacted_for_trace, redacted_tool_result_content

_FORMATS: tuple[str, ...] = ("json", "jsonl")
_DEFAULT_FORMATS: tuple[str, ...] = _FORMATS
# Normal completion markers (Anthropic / OpenAI-compatible). They carry no
# information and appear on almost every turn, so trajectories omit them.
_ORDINARY_FINISH_REASONS = frozenset({"end_turn", "stop"})
_DIAGNOSTIC_KEYS = ("usage", "thinking_blocks", "finish_reason", "stop_details")


def _json_default(value: Any) -> Any:
    """Degrade a non-JSON value instead of failing every later snapshot.

    Loop history and tool results may carry host metadata (datetimes, SDK
    objects, bytes). A trace is diagnostic, so a lossy rendering beats losing
    the whole document to one ``TypeError``.
    """
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"
    if isinstance(value, (set, frozenset)):
        return list(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        with contextlib.suppress(Exception):
            return model_dump(mode="json")
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        with contextlib.suppress(Exception):
            return isoformat()
    return str(value)


_STREAM_ENCODER = json.JSONEncoder(
    ensure_ascii=False, separators=(",", ":"), default=_json_default,
)


def _line_digest(line: bytes) -> bytes:
    """Digest of one rendered (redacted, clipped) spool line."""
    return hashlib.blake2b(line, digest_size=16).digest()


def _lineage(message: Any) -> list[Any]:
    """Objects a shallow copy of *message* still shares with it.

    Compaction and thinking resets copy the message dict but keep its values
    (or, for a rebuilt content list, its blocks). A message the model produced
    separately never shares these objects, even when its text is identical.
    Falsy values are excluded: ``""`` and ``()`` are interpreter singletons.
    """
    if not isinstance(message, dict):
        return []
    shared: list[Any] = []
    for key in ("content", "tool_calls", "reasoning_content"):
        value = message.get(key)
        if value:
            shared.append(value)
            if isinstance(value, list):
                shared.extend(item for item in value if item)
    return shared


_DEFAULT_FORMAT_ENV_VARS = (
    "AGENT_CORE_TRAJECTORY_FORMATS",
    "FRONTIER_AGENT_TRAJECTORY_FORMATS",
    "MIROHARNESS_TRAJECTORY_FORMATS",
    "MIRO_TRAJECTORY_FORMATS",
    "SWARM_TRAJECTORY_FORMATS",
)


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


# ── Trajectory memory guardrails ──────
#
# One instance of this observer exists per agent (main + every sub-agent).
# History is streamed one message at a time to disk, and two bounds keep
# each public JSON snapshot cheap:
#
#   _BODY_MAX_CHARS — per tool-result body kept in the JSON envelope.
#     Uncapped, a single entry could hold a full ``TOOL_RESULT_MAX_CHARS``
#     (150_000) payload; × turns × agents that dominated worker RAM. When
#     JSONL is enabled (the default), that stream still records every body
#     untruncated. JSON-only configurations intentionally keep only the cap.
#   _COALESCE_N / _COALESCE_MS — snapshot batching, mirroring the
#     equivalent defaults in the external worker-trace observer. Every
#     history-changing event (LLM input/response, tool result, turn end)
#     counts once. Bounds the O(n²) whole-document copy to one per batch.
#     The copy is streamed without a second full string, and only messages
#     new or changed since the previous snapshot are re-rendered.
#
# 0 disables the respective bound.
_BODY_MAX_CHARS: int = _env_int("SWARM_TRAJECTORY_BODY_MAX_CHARS", 16384)
_COALESCE_N: int = _env_int("SWARM_TRAJECTORY_COALESCE_N", 20)
_COALESCE_MS: float = float(_env_int("SWARM_TRAJECTORY_COALESCE_MS", 0))


def _clip(text: str, limit: int) -> str:
    """Head-slice *text* to *limit* chars, marking what was dropped."""
    if limit <= 0 or len(text) <= limit:
        return text
    return (
        f"{text[:limit]}\n"
        f"... [truncated {len(text) - limit} of {len(text)} chars]"
    )


def _resolve_formats(
    arg: Iterable[str] | None,
    env_vars: Iterable[str] = _DEFAULT_FORMAT_ENV_VARS,
) -> set[str]:
    """Resolve enabled formats from an argument, environment, or defaults."""
    if arg is not None:
        return {f for f in arg if f in _FORMATS}
    env = ""
    for name in env_vars:
        env = os.getenv(name, "")
        if env:
            break
    if env:
        return {f.strip() for f in env.split(",") if f.strip() in _FORMATS}
    return set(_DEFAULT_FORMATS)


class TrajectoryFileObserver(BaseObserver):
    """Save conversation snapshots (JSON) and execution events (JSONL).

    JSON follows the loop's current history, including rewrites and synthetic
    replies, with diagnostic annotations, image redaction and the configured
    tool-body cap. JSONL retains original events even after history is pruned.
    """

    # These hooks do synchronous file I/O without yielding, so a background
    # task never moved that work off the event loop; it only reordered it.
    # Awaiting them preserves event order even when a synthetic result needs
    # no async tool execution, where a background on_llm_response could run
    # AFTER it. Coalescing limits snapshot work; streamed value digests let
    # unchanged spool prefixes survive while still detecting nested edits.
    critical: bool = True

    def __init__(
        self,
        output_dir: Path,
        *,
        filename: str | None = None,
        formats: Iterable[str] | None = None,
        tools: list[Any] | None = None,
        model_name: str | None = None,
        system_prompt: str | None = None,
        user_message: str | None = None,
        format_env_vars: Iterable[str] = _DEFAULT_FORMAT_ENV_VARS,
        tool_schema_detail: Literal["full", "minimal"] = "full",
        include_start_tool_names: bool = True,
    ) -> None:
        """Args:
            output_dir: Directory where trajectory file(s) are written.
            filename: Optional stem (without extension). Falls back to the
                loop's ``task_id`` when omitted. Sub-agents pass their
                session name so each agent gets its own file.
            formats: Subset of ``("json", "jsonl")``. ``None`` → env var
                ``FRONTIER_AGENT_TRAJECTORY_FORMATS`` (or legacy
                ``SWARM_TRAJECTORY_FORMATS``) → both.
            tools: Optional list of native ``Tool`` instances / dict
                objects bound to this agent's LLM. Serialized to OpenAI
                ``{"type": "function", "function": {...}}`` schema and
                emitted as the JSON envelope's top-level ``tools`` field
                so a replay consumer can reconstruct the LLM call
                signature. Tool *names* (derived from this list) are
                also recorded in the JSONL ``start`` event for
                lightweight downstream consumers.
            model_name: The model id (e.g. ``"Qwen3-235B"``). Recorded
                on both formats so post-run renderers know what ran.
            system_prompt: System prompt seeded into the loop. Recorded
                on the JSONL ``start`` event so post-run renderers can
                reconstruct the full message history without re-reading
                the JSON envelope.
            user_message: Initial user message. Same purpose as
                ``system_prompt``.
        """
        self._dir = output_dir
        self._filename = filename
        self._formats = _resolve_formats(formats, format_env_vars)
        self._tools_schema = self._serialize_tools(
            tools or [], detail=tool_schema_detail,
        )
        self._tool_names = []
        if include_start_tool_names:
            self._tool_names = [
                (t.get("function", {}) or {}).get("name", "")
                for t in self._tools_schema
                if isinstance(t, dict)
            ]
            self._tool_names = [n for n in self._tool_names if n]
        self._model_name = model_name or ""
        self._system_prompt = system_prompt or ""
        self._user_message = user_message or ""
        self._task_id: str = ""
        self._role_id: str = ""
        self._max_turns: int = 0
        self._turns_used: int = 0
        self._tool_calls_count: int = 0
        self._stopped_by: str = ""

        # Borrow the loop's live history instead of reconstructing it from
        # execution events. The spool is refreshed only when a snapshot is due.
        # No second in-memory copy of the message bodies is retained.
        self._history: list[Message] | None = None
        # True when no loop history was offered (a host driving the hooks
        # directly), so the observer assembles the conversation itself.
        self._owns_history: bool = False
        # Diagnostic fields are separate from history: id(message) ->
        # [message, details], one entry per assistant in history. Identity
        # binds them first; a replacement inherits an entry only when it is a
        # shallow copy of exactly one vanished assistant (see ``_lineage``).
        # Entries without diagnostics still compete, so repeated text cannot
        # inherit another turn's details. Entries whose message left history
        # are released at the next snapshot.
        self._response_metadata: dict[int, list[Any]] = {}
        # One (message, reference signature, line digest, end offset) per
        # history entry already in the spool. A snapshot keeps the longest
        # unchanged prefix and re-renders from the first difference.
        self._spool_entries: list[tuple[Any, tuple[Any, ...], bytes, int]] = []
        # Reference signatures catch replaced messages and values. Edits nested
        # inside a kept container need the rendered content compared, which
        # costs a full re-render, so it runs only after a compaction or
        # rollback notice and on forced (terminal) snapshots.
        self._deep_check_due: bool = False
        self._compaction_seq: Any = None
        # Set by on_loop_end. A cancellation arriving after it (while a later
        # observer still runs) must not overwrite the terminal snapshot.
        self._ended: bool = False
        # Flush coalescing cursors (see ``_flush_json``). ``0.0`` rather than
        # ``time.monotonic()`` so the FIRST flush is always due: the envelope
        # then appears on disk immediately instead of only after the first
        # batch fills, which matters when the process is SIGKILLed (OOM) —
        # a stale-but-present file beats no file at all.
        self._pending_flushes: int = 0
        self._last_flush_at: float = 0.0

        # JSONL append state
        self._jsonl_handle: IO[str] | None = None

    # ── Path helpers ────────────────────────────────────────────────────

    @staticmethod
    def _safe(stem: str) -> str:
        return stem.replace("/", "__").replace(":", "_")

    @staticmethod
    def _serialize_tools(
        tools: list[Any], *, detail: Literal["full", "minimal"] = "full",
    ) -> list[dict]:
        """Convert native :class:`Tool` instances / dicts to OpenAI schema.

        Accepts a heterogeneous list (``Tool`` / already-serialized dict)
        and returns OpenAI
        ``{"type": "function", "function": {"name", "description",
        "parameters"}}`` entries. A wire-critical byte-exact schema pinned
        on ``Tool.metadata["openai_schema"]`` is preferred over recomputing
        via ``to_openai_schema()``. Best effort: tools that can't be
        introspected fall through to a name + description stub.
        """
        out: list[dict] = []
        for t in tools:
            if isinstance(t, dict):
                if "function" in t and "type" in t:
                    out.append(t)
                elif "name" in t:
                    out.append({"type": "function", "function": t})
                continue
            if detail == "full":
                pinned = (getattr(t, "metadata", None) or {}).get("openai_schema")
                if isinstance(pinned, dict):
                    out.append(pinned)
                    continue
                schema_fn: Callable[[], dict[str, Any]] | None = getattr(
                    t, "to_openai_schema", None,
                )
                if callable(schema_fn):
                    try:
                        out.append(schema_fn())
                        continue
                    except Exception:
                        pass
            out.append({
                "type": "function",
                "function": {
                    "name": getattr(t, "name", None) or t.__class__.__name__,
                    "description": getattr(t, "description", "") or "",
                    "parameters": (
                        getattr(t, "parameters", {}) or {}
                        if detail == "full"
                        else {}
                    ),
                },
            })
        return out

    def _path(self, ext: str) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        stem = self._filename or self._task_id or "trace"
        return self._dir / f"{self._safe(stem)}.{ext}"

    def _message_spool_path(self) -> Path:
        # Deliberately avoid a ``.jsonl`` suffix: trajectory consumers glob
        # that namespace and expect every match to use the public event schema.
        return self._path("messages.spool")

    # ── JSONL writer ────────────────────────────────────────────────────

    def _write_jsonl(self, record: dict) -> None:
        if "jsonl" not in self._formats:
            return
        if self._jsonl_handle is None:
            self._jsonl_handle = open(  # noqa: SIM115
                self._path("jsonl"), "a", encoding="utf-8",
            )
        record.setdefault("ts", time.time())
        for chunk in _STREAM_ENCODER.iterencode(record):
            self._jsonl_handle.write(chunk)
        self._jsonl_handle.write("\n")
        self._jsonl_handle.flush()

    def _close_jsonl(self) -> None:
        if self._jsonl_handle is not None:
            self._jsonl_handle.close()
            self._jsonl_handle = None

    # ── JSON envelope writer ────────────────────────────────────────────

    def _close_message_spool(self, *, cleanup: bool = False) -> None:
        self._spool_entries = []
        if cleanup:
            with contextlib.suppress(FileNotFoundError):
                self._message_spool_path().unlink()

    @staticmethod
    def _signature(message: Any, details: Any) -> tuple[Any, ...]:
        """Shallow structure of *message*, compared by reference identity.

        History edits replace values (``message["content"] = rebuilt``) or
        whole messages, so identity of each value and content block detects
        them without serialising anything. Holding the references keeps their
        ids from being reused before the next comparison.
        """
        if not isinstance(message, dict):
            return (message, details)
        content = message.get("content")
        blocks = tuple(content) if isinstance(content, list) else ()
        return (tuple(message.items()), blocks, details)

    @staticmethod
    def _same(a: Any, b: Any) -> bool:
        if isinstance(a, tuple) and isinstance(b, tuple):
            return len(a) == len(b) and all(
                TrajectoryFileObserver._same(x, y) for x, y in zip(a, b, strict=True)
            )
        return a is b

    def _resolve_annotations(self, history: list[Message]) -> dict[int, dict[str, Any]]:
        """Bind diagnostics to the history messages that carry them now."""
        present = {id(message): message for message in history}
        bound: dict[int, list[Any]] = {}
        unmatched: list[list[Any]] = []
        for key, entry in self._response_metadata.items():
            if present.get(key) is entry[0]:
                bound[key] = entry
            else:
                unmatched.append(entry)
        if unmatched:
            # Map each object a vanished assistant owned to that assistant.
            # The entry still references the old message, so these ids
            # cannot be reused while this lookup is alive.
            owners: dict[int, list[list[Any]]] = {}
            for entry in unmatched:
                for value in _lineage(entry[0]):
                    sources = owners.setdefault(id(value), [])
                    if entry not in sources:
                        sources.append(entry)
            if owners:
                claims: dict[int, list[Message]] = {}
                for message in history:
                    if (
                        not isinstance(message, dict)
                        or message.get("role") != "assistant"
                        or id(message) in bound
                    ):
                        continue
                    sources: list[list[Any]] = []
                    for value in _lineage(message):
                        for entry in owners.get(id(value), ()):
                            if entry not in sources:
                                sources.append(entry)
                    # A copy of several turns, or several copies of one turn,
                    # identifies nothing. Leave those diagnostics in JSONL.
                    if len(sources) == 1:
                        claims.setdefault(id(sources[0]), []).append(message)
                for entry in unmatched:
                    targets = claims.get(id(entry), [])
                    if len(targets) == 1:
                        entry[0] = targets[0]
                        bound[id(targets[0])] = entry
        retained: dict[int, list[Any]] = {}
        for message in history:
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            entry = bound.get(id(message)) or [message, {}]
            retained[id(message)] = entry
        self._response_metadata = retained
        return {key: entry[1] for key, entry in retained.items() if entry[1]}

    def _render(self, message: Any, details: dict[str, Any] | None) -> bytes:
        rendered = self._message_to_dict(message)
        if rendered is None:
            return b""
        if details:
            blocks = details.get("thinking_blocks")
            content = message.get("content")
            if (
                blocks and isinstance(content, list)
                and all(block in content for block in blocks)
            ):
                # Already preserved verbatim as native content blocks.
                details = {k: v for k, v in details.items() if k != "thinking_blocks"}
            rendered.update(details)
        try:
            text = _STREAM_ENCODER.encode(rendered)
        except (TypeError, ValueError, RecursionError) as exc:
            # Non-string keys or a reference cycle in host metadata. Keep the
            # message's place (and its tool pairing) instead of losing the
            # whole snapshot.
            fallback = {
                key: rendered[key] for key in ("role", "tool_call_id", "tool_calls")
                if key in rendered
            }
            fallback["content"] = str(rendered.get("content", ""))
            fallback["trace_error"] = f"unserializable message: {exc}"
            try:
                text = _STREAM_ENCODER.encode(fallback)
            except (TypeError, ValueError, RecursionError):
                fallback.pop("tool_calls", None)
                text = _STREAM_ENCODER.encode(fallback)
        return (text + "\n").encode("utf-8")

    def _sync_history(self, *, deep: bool = False) -> None:
        """Bring the spool up to the real history, including in-place edits.

        A length/delta cursor is insufficient: compaction, rollback, thinking
        resets and image eviction can replace or edit already-recorded messages.
        The unchanged prefix stays on disk; from the first difference onwards
        the spool is truncated and re-rendered. Each message is rendered whole
        before it is written, so a failure leaves a consistent prefix.

        Prefix messages are compared by reference signature. *deep* also
        re-renders them and compares the redacted line, which catches edits
        nested inside a container the message kept.
        """
        history = self._history
        if history is None:
            return
        details_for = self._resolve_annotations(history)
        entries = self._spool_entries
        path = self._message_spool_path()
        keep = 0
        first_line: bytes | None = None
        if path.exists():
            limit = min(len(entries), len(history))
            while keep < limit:
                message = history[keep]
                ref, signature, digest, _ = entries[keep]
                details = details_for.get(id(message))
                if ref is not message or not self._same(
                    signature, self._signature(message, details),
                ):
                    break
                if deep:
                    line = self._render(message, details)
                    if _line_digest(line) != digest:
                        first_line = line
                        break
                keep += 1
        offset = entries[keep - 1][3] if keep else 0
        del entries[keep:]
        # Open even for an empty history so a rollback to [] clears the spool.
        with path.open("r+b" if keep else "wb") as fh:
            fh.seek(offset)
            fh.truncate()
            for message in history[keep:]:
                details = details_for.get(id(message))
                line = first_line if first_line is not None else self._render(message, details)
                first_line = None
                fh.write(line)
                offset += len(line)
                entries.append((
                    message, self._signature(message, details), _line_digest(line), offset,
                ))

    def _write_envelope(self, path: Path) -> None:
        """Stream one compatible JSON envelope without materialising it."""
        fields: list[tuple[str, Any]] = [
            ("task_id", self._task_id),
            ("role_id", self._role_id),
            ("max_turns", self._max_turns),
            ("turns_used", self._turns_used),
            ("tool_calls_count", self._tool_calls_count),
            ("stopped_by", self._stopped_by),
            ("tools", self._tools_schema),
        ]
        if self._model_name:
            fields.append(("model_name", self._model_name))
        with path.open("w", encoding="utf-8") as fh:
            fh.write("{")
            for idx, (key, value) in enumerate(fields):
                if idx:
                    fh.write(",")
                for chunk in _STREAM_ENCODER.iterencode(key):
                    fh.write(chunk)
                fh.write(":")
                for chunk in _STREAM_ENCODER.iterencode(value):
                    fh.write(chunk)
            fh.write(',"messages":[')
            spool_path = self._message_spool_path()
            if spool_path.exists():
                with spool_path.open("r", encoding="utf-8") as spool:
                    first = True
                    for line in spool:
                        # A hard crash can leave one partial tail record.
                        # Complete spool records always end in a newline.
                        if not line.endswith("\n") or not line.strip():
                            continue
                        if not first:
                            fh.write(",")
                        first = False
                        fh.write(line.rstrip("\n"))
            fh.write("]}")

    def _count_change(self) -> None:
        """Count a history change toward the next coalesced snapshot."""
        if "json" in self._formats:
            self._pending_flushes += 1

    def _flush_json(self, *, force: bool = False) -> None:
        """Rewrite the JSON envelope, coalescing bursts unless *force*.

        The envelope is a single document, so every flush still copies all
        message chunks from the refreshed history spool. Coalescing bounds that
        O(n²) I/O to one copy per ``_COALESCE_N`` events or
        ``_COALESCE_MS`` milliseconds. The copy is streamed, so neither the
        message history nor its serialised twin is materialised in memory.
        ``on_loop_end`` forces a final flush; the public JSONL event stream
        remains the live-tail surface.
        """
        if "json" not in self._formats:
            return
        if not force:
            self._pending_flushes += 1
            now = time.monotonic()
            # "0 disables the respective bound" (module header). With every
            # bound disabled there is nothing left to coalesce, so every event
            # flushes — treating a disabled bound as "never due" instead meant
            # ``SWARM_TRAJECTORY_COALESCE_N=0`` froze the JSON snapshot after
            # the first flush, and a process killed before ``on_loop_end``
            # left only the ``start`` state on disk.
            coalescing = _COALESCE_N > 0 or _COALESCE_MS > 0
            due = (
                not coalescing
                or self._last_flush_at == 0.0
                or (
                    _COALESCE_N > 0
                    and self._pending_flushes >= _COALESCE_N
                )
                or (
                    _COALESCE_MS > 0
                    and (now - self._last_flush_at) * 1000.0 >= _COALESCE_MS
                )
            )
            if not due:
                return
        self._pending_flushes = 0
        self._last_flush_at = time.monotonic()
        deep = force or self._deep_check_due
        self._sync_history(deep=deep)
        if deep:
            self._deep_check_due = False
        path = self._path("json")
        tmp = path.with_suffix(path.suffix + ".tmp")
        self._write_envelope(tmp)
        os.replace(tmp, path)

    def _message_to_dict(self, m: Any) -> dict | None:
        """Pass through native OpenAI-shaped dict messages.

        Loop history is now a list of plain OpenAI-wire dicts
        (``{"role", "content", "tool_calls"?, "tool_call_id"?,
        "reasoning_content"?}``), so emitting them onto the JSON envelope
        is a copy. Anything that isn't a role-bearing dict is dropped.
        """
        if isinstance(m, dict) and m.get("role"):
            # ``redacted_for_trace`` swaps an inline image's base64 for its
            # size. Verbatim, one 1080p screenshot writes ~137 KB here for
            # every turn it stays in history.
            message = dict(redacted_for_trace(m))
            if message.get("role") == "tool":
                content = message.get("content")
                if isinstance(content, str):
                    message["content"] = _clip(content, _BODY_MAX_CHARS)
                elif isinstance(content, list):
                    message["content"] = [
                        {**block, "text": _clip(block["text"], _BODY_MAX_CHARS)}
                        if isinstance(block, dict) and isinstance(block.get("text"), str)
                        else block
                        for block in content
                    ]
            return message
        return None

    # ── Lifecycle hooks ─────────────────────────────────────────────────

    # ── Recovery handle ─────────────────────────────────────────────────
    #
    # The JSONL stream is the only place a site-3 truncation's discarded content
    # survives (the post-processor at ``agent_loop.py:762`` persists nothing, and
    # runs AFTER ``notify_tool_result``, so the recorded body predates the cut).
    # ``recover_result`` needs to find this file, and the file is not
    # sandbox-visible, so the path is published rather than mounted.
    #
    # Published into ``ExecutionScope.metadata`` and NOT into a contextvar.
    # Non-critical hooks may run in ``asyncio.create_task``; a contextvar
    # ``.set()`` inside such a task mutates only that task's own context. Scope
    # metadata works for both awaited and background observers, and is seen
    # by the loop and every nested tool-call task. Same channel
    # ``agent_loop.py`` already uses for ``current_turn``.
    SCOPE_KEY = "trajectory_jsonl"

    def _publish_jsonl_path(self) -> None:
        """Advertise this agent's JSONL file, or withdraw the key when there is
        none to advertise.

        Absence of the key is the signal that no handle can be minted, so a
        jsonl-disabled configuration degrades to today's behaviour instead of
        minting handles that resolve to nothing.
        """
        from agent_core.execution_context import (
            get_current_execution_scope,
        )
        scope = get_current_execution_scope()
        if scope is None:
            return
        if "jsonl" not in self._formats:
            scope.metadata.pop(self.SCOPE_KEY, None)
            return
        # Deliberately not ``_path()``: that mkdirs (see ``_path``), and a
        # "where is my trajectory" answer must not create directories.
        stem = self._safe(self._filename or self._task_id or "trace")
        scope.metadata[self.SCOPE_KEY] = str(self._dir / f"{stem}.jsonl")

    def _withdraw_jsonl_path(self) -> None:
        from agent_core.execution_context import (
            get_current_execution_scope,
        )
        scope = get_current_execution_scope()
        if scope is not None:
            scope.metadata.pop(self.SCOPE_KEY, None)

    async def on_loop_start(self, config: LoopConfig) -> None:
        self._task_id = config.task_id
        self._role_id = config.role_id
        self._max_turns = config.max_turns
        self._ended = False
        self._deep_check_due = False
        self._compaction_seq = None
        # After ``_task_id`` is set: it is the stem fallback when no explicit
        # ``filename`` was passed.
        self._publish_jsonl_path()

        start_record: dict = {
            "t": "start",
            "task_id": config.task_id,
            "role_id": config.role_id,
            "max_turns": config.max_turns,
        }
        if self._model_name:
            start_record["model_name"] = self._model_name
        if self._system_prompt:
            start_record["system_prompt"] = self._system_prompt
        if self._user_message:
            start_record["user_message"] = self._user_message
        if self._tool_names:
            start_record["tool_names"] = self._tool_names
        self._write_jsonl(start_record)
        self._flush_json()

    async def on_before_llm(self, ctx: TurnContext) -> Intervention | None:
        # Unlike on_llm_input's request copy, this list stays live across
        # injections, compaction, early exits and cancellation. The loop
        # rewrites it in place, so binding once per run is enough.
        self._bind_history(ctx.messages)
        self._note_compaction(ctx)
        return None

    async def on_llm_input(self, ctx: TurnContext) -> None:
        # Injections from on_before_llm have now been applied. Transient request
        # addenda belong to the request, not to durable conversation history.
        self._note_compaction(ctx)
        self._flush_json()

    async def on_turn_end(self, ctx: TurnContext) -> Intervention | None:
        self._note_compaction(ctx)
        self._flush_json()
        return None

    async def on_context_compacted(self, ctx: ContextCompactionContext) -> None:
        # Rollbacks and context-limit pops announce themselves here.
        self._deep_check_due = True

    def _note_compaction(self, ctx: TurnContext) -> None:
        """Schedule a deep check when a compactor ran since the last look.

        A compactor may edit blocks in place, leaving every reference intact,
        and then the loop sends no ``on_context_compacted``. Its run counter in
        loop metadata advances regardless.
        """
        seq = (ctx.metadata or {}).get(COMPACTION_SEQ_KEY)
        if seq != self._compaction_seq:
            if self._compaction_seq is not None or seq:
                self._deep_check_due = True
            self._compaction_seq = seq

    def _bind_history(self, messages: list[Message] | None) -> None:
        if "json" not in self._formats or self._owns_history or messages is None:
            return
        if self._history is None:
            self._history = messages
            # Resumed histories can contain assistants without diagnostics;
            # include them when deciding whether a future copy is ambiguous.
            self._resolve_annotations(messages)

    @staticmethod
    def _assistant_from_ctx(ctx: TurnContext) -> Message:
        """Build the assistant entry for hosts that supply no loop history."""
        message: dict[str, Any] = {"role": "assistant", "content": ctx.ai_text or ""}
        if ctx.thinking:
            message["reasoning_content"] = ctx.thinking
        if ctx.tool_calls:
            message["tool_calls"] = [
                {
                    "id": tc.get("id") or f"call_{ctx.turn}_{idx}",
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": (
                            _STREAM_ENCODER.encode(tc.get("args", {}))
                            if isinstance(tc.get("args", {}), (dict, list))
                            else str(tc.get("args") or "")
                        ),
                    },
                }
                for idx, tc in enumerate(ctx.tool_calls)
            ]
        return message  # type: ignore[return-value]

    def _owned_tool_call_id(self, result: ToolResult) -> str:
        cid = getattr(result, "tool_call_id", "") or ""
        if cid or self._history is None:
            return cid
        answered = {m.get("tool_call_id") for m in self._history if m.get("role") == "tool"}
        for message in reversed(self._history):
            if message.get("role") == "assistant":
                for tc in message.get("tool_calls") or []:
                    if tc["id"] not in answered:
                        return tc["id"]
                break
        return ""

    async def on_llm_response(
        self, ctx: TurnContext,
    ) -> Intervention | None:
        # JSONL: append one event
        record: dict = {
            "t": "llm",
            "turn": ctx.turn,
            "content": ctx.ai_text,
            "tool_calls": [
                {"name": tc.get("name"), "args": tc.get("args", {})}
                for tc in (ctx.tool_calls or [])
            ],
        }
        if ctx.thinking:
            record["thinking"] = ctx.thinking
        if ctx.usage:
            record["usage"] = ctx.usage
        if ctx.thinking_blocks:
            # Native Anthropic thinking / OpenAI Responses reasoning: the
            # verbatim block list (thinking+signature / reasoning+
            # encrypted_content / text) so the (sub-)agent trajectory stays
            # replay-able with signatures / encrypted reasoning intact.
            record["thinking_blocks"] = ctx.thinking_blocks
        # Why the turn ended. Without it a trajectory cannot distinguish a
        # finished answer from one the output cap truncated, or from a request
        # a safety classifier declined — all three look like a turn whose text
        # stops, and a refusal looks like a turn that said nothing at all.
        # ``end_turn`` / ``stop`` are the uninformative common case and are
        # omitted to keep the line small; everything else (``length``, ``tool_use``,
        # ``refusal``, …) is recorded.
        if ctx.finish_reason and ctx.finish_reason not in _ORDINARY_FINISH_REASONS:
            record["finish_reason"] = ctx.finish_reason
        if ctx.stop_details:
            record["stop_details"] = ctx.stop_details
        self._write_jsonl(record)

        if "json" in self._formats:
            messages = ctx.messages or []
            self._bind_history(messages if messages else None)
            # The last assistant may already be followed by skipped/blocked
            # replies. Never infer its position from messages[-1]. Awaited
            # dispatch ensures this still refers to the response being observed.
            assistant = next(
                (m for m in reversed(messages)
                 if isinstance(m, dict) and m.get("role") == "assistant"),
                None,
            )
            if assistant is None and (self._history is None or self._owns_history):
                # No loop history at all: keep the pre-history-snapshot
                # contract and record the turn from the context itself.
                self._owns_history = True
                if self._history is None:
                    self._history = []
                assistant = self._assistant_from_ctx(ctx)
                self._history.append(assistant)
            if assistant is not None:
                details = {key: record[key] for key in _DIAGNOSTIC_KEYS if key in record}
                entry = self._response_metadata.get(id(assistant))
                if entry is not None and entry[0] is assistant:
                    entry[1] = details
                else:
                    self._response_metadata[id(assistant)] = [assistant, details]
            self._tool_calls_count += len(ctx.tool_calls or [])
            self._turns_used = ctx.turn
            self._flush_json()
        return None

    async def on_tool_result(
        self, ctx: TurnContext, result: ToolResult,
    ) -> ToolResult | None:
        # Results describe what the tool returned, before post-processing and
        # image capability gating. Only the real history can say what the model
        # received. Empty ids here mean the runtime itself supplied no id.
        jsonl_record: dict[str, Any] = {
            "t": "result",
            "turn": ctx.turn,
            "name": result.name,
            "tool_call_id": getattr(result, "tool_call_id", "") or "",
            "result": result.result,
            "error": result.is_error,
            "ms": result.duration_ms,
        }
        if result.images:
            # The result event predates message attachment/capability gating, so
            # this states what the TOOL returned, not that every profile saw it.
            jsonl_record["images"] = redacted_tool_result_content("", result.images)
        self._write_jsonl(jsonl_record)

        if "json" in self._formats:
            if self._owns_history and self._history is not None:
                body = _clip(
                    result.result if isinstance(result.result, str)
                    else ("" if result.result is None else str(result.result)),
                    _BODY_MAX_CHARS,
                )
                self._history.append({  # type: ignore[arg-type]
                    "role": "tool",
                    "tool_call_id": self._owned_tool_call_id(result),
                    "content": redacted_tool_result_content(
                        f"[error] {body}" if result.is_error else body,
                        result.images,
                    ),
                })
            # The loop appends the reply only after this hook, so a snapshot
            # here would show an unanswered call. Count it toward the next one.
            self._count_change()
        return None

    async def on_compaction(self, event: CompactionEvent) -> None:
        """Record what a compaction discarded and what it kept in its place.

        The summary is written whole. It is the one artifact that explains a
        rewrite the rest of the trajectory cannot show — the replaced turns are
        gone from the history by the time anything reads it — and it is bounded
        by the summariser's own output length, not by tool output, so it does
        not need the ``_BODY_MAX_CHARS`` guard that tool results do.
        """
        self._deep_check_due = True
        self._write_jsonl({
            "t": "compaction",
            "turn": event.turn,
            "seq": event.seq,
            "selected": event.selected,
            "tokens_before": event.tokens_before,
            "tokens_after": event.tokens_after,
            "relief_met": event.relief_met,
            "spill_refs": event.spill_refs,
            "attempts": event.attempts,
            "summary": event.summary,
            "rollback_reason": event.rollback_reason,
        })

    async def on_loop_end(self, result: AgentLoopResult) -> None:
        if "json" in self._formats and not self._owns_history:
            self._history = result.messages
        self._ended = True
        self._turns_used = max(self._turns_used, result.turns_used)
        self._tool_calls_count = result.tool_calls_count
        self._stopped_by = (
            getattr(result, "stopped_by", "") or self._stopped_by
        )

        self._write_jsonl({
            "t": "end",
            "turns": result.turns_used,
            "tool_calls": result.tool_calls_count,
            "stopped_by": result.stopped_by,
        })
        self._close_jsonl()
        # The file stays on disk and stays readable; withdrawing the key only
        # stops a handle being minted against a loop that has ended.
        self._withdraw_jsonl_path()
        # force=True: the terminal envelope must be complete regardless of
        # where the coalescing cursors happen to sit.
        snapshot_written = False
        try:
            self._flush_json(force=True)
            snapshot_written = True
        finally:
            # Always release the FD. Keep the forensic spool when the terminal
            # snapshot fails; remove it only after a successful atomic replace.
            self._close_message_spool(cleanup=snapshot_written)
            self._history = None
            self._owns_history = False
            self._response_metadata = {}

    async def on_loop_cancelled(self) -> None:
        """Flush current history and preserve crash-forensic sidecars."""
        self._close_jsonl()
        self._withdraw_jsonl_path()
        if self._ended:
            # on_loop_end already wrote the terminal snapshot and released the
            # history; re-flushing would replace it with an empty envelope.
            return
        try:
            self._flush_json(force=True)
        finally:
            self._close_message_spool()
            self._history = None
            self._owns_history = False
            self._response_metadata = {}
