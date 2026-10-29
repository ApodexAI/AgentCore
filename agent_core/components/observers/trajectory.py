# pyright: reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportMissingTypeArgument=false, reportUnnecessaryIsInstance=false
"""Write workflow-neutral trajectories in JSON and/or JSONL formats.

JSON snapshots are atomically replaced; JSONL events are flushed incrementally
so partial runs remain readable without retaining full payloads in memory.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import IO, Any, Literal

from agent_core.loop_types import (
    AgentLoopResult,
    BaseObserver,
    CompactionEvent,
    Intervention,
    LoopConfig,
    ToolResult,
    TurnContext,
)
from agent_core.messages import Message
from agent_core.tool_content import redacted_for_trace, redacted_tool_result_content

_FORMATS: tuple[str, ...] = ("json", "jsonl")
_DEFAULT_FORMATS: tuple[str, ...] = _FORMATS
# Normal completion markers (Anthropic / OpenAI-compatible). They carry no
# information and appear on almost every turn, so trajectories omit them.
_ORDINARY_FINISH_REASONS = frozenset({"end_turn", "stop"})
_STREAM_ENCODER = json.JSONEncoder(ensure_ascii=False, separators=(",", ":"))
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
#     equivalent defaults in the external worker-trace observer.
#     Bounds the O(n²) whole-document copy to
#     one per batch. The copy is streamed without a second full string.
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

    # These hooks do synchronous file I/O without yielding. Awaiting them
    # preserves event order even when a synthetic result needs no async tool
    # execution; a background on_llm_response can otherwise run AFTER it.
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
        # Diagnostic fields are separate from history. Identity checks avoid
        # annotating a replacement message after compaction/rollback. Discarded
        # references are released at the next coalesced snapshot.
        self._response_metadata: dict[int, tuple[Message, dict[str, Any]]] = {}
        self._message_spool_handle: IO[str] | None = None
        self._message_count: int = 0
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

    def _append_message(self, message: dict) -> None:
        """Append a message; the snapshot writer flushes the complete spool."""
        if self._message_spool_handle is None:
            self._message_spool_handle = open(  # noqa: SIM115
                self._message_spool_path(), "w", encoding="utf-8",
            )
        for chunk in _STREAM_ENCODER.iterencode(message):
            self._message_spool_handle.write(chunk)
        self._message_spool_handle.write("\n")
        self._message_count += 1

    def _close_message_spool(self, *, cleanup: bool = False) -> None:
        if self._message_spool_handle is not None:
            self._message_spool_handle.close()
            self._message_spool_handle = None
        if cleanup:
            with contextlib.suppress(FileNotFoundError):
                self._message_spool_path().unlink()

    def _sync_history(self) -> None:
        """Replace the spool with the real history, including in-place edits.

        A length/delta cursor is insufficient: compaction, rollback, thinking
        resets and image eviction can replace or edit already-recorded messages.
        Streaming at the coalesced flush boundary keeps those changes faithful
        without a full history copy or an O(n) rewrite on every tool callback.
        """
        if self._history is None:
            return
        self._close_message_spool()
        self._message_count = 0
        retained: dict[int, tuple[Message, dict[str, Any]]] = {}
        # Open even for an empty history so a rollback to [] clears the spool.
        self._message_spool_handle = open(  # noqa: SIM115
            self._message_spool_path(), "w", encoding="utf-8",
        )
        for message in self._history:
            rendered = self._message_to_dict(message)
            if rendered is None:
                continue
            annotation = self._response_metadata.get(id(message))
            if annotation is not None and annotation[0] is message:
                rendered.update(annotation[1])
                retained[id(message)] = annotation
            self._append_message(rendered)
        self._message_spool_handle.flush()
        self._response_metadata = retained

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
        self._sync_history()
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
        if "json" in self._formats:
            # Unlike on_llm_input's request copy, this list stays live across
            # injections, compaction, early exits and cancellation.
            self._history = ctx.messages
        return None

    async def on_llm_input(self, ctx: TurnContext) -> None:
        # Injections from on_before_llm have now been applied. Transient request
        # addenda belong to the request, not to durable conversation history.
        self._flush_json()

    async def on_turn_end(self, ctx: TurnContext) -> Intervention | None:
        if "json" in self._formats:
            self._history = ctx.messages
            self._flush_json()
        return None

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
            self._history = ctx.messages
            # The last assistant may already be followed by skipped/blocked
            # replies. Never infer its position from messages[-1]. Awaited
            # dispatch ensures this still refers to the response being observed.
            assistant = next(
                (m for m in reversed(ctx.messages) if m.get("role") == "assistant"),
                None,
            )
            if assistant is not None:
                details = {key: record[key] for key in (
                    "usage", "thinking_blocks", "finish_reason", "stop_details",
                ) if key in record}
                self._response_metadata[id(assistant)] = (assistant, details)
            self._tool_calls_count += len(ctx.tool_calls)
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

        return None

    async def on_compaction(self, event: CompactionEvent) -> None:
        """Record what a compaction discarded and what it kept in its place.

        The summary is written whole. It is the one artifact that explains a
        rewrite the rest of the trajectory cannot show — the replaced turns are
        gone from the history by the time anything reads it — and it is bounded
        by the summariser's own output length, not by tool output, so it does
        not need the ``_BODY_MAX_CHARS`` guard that tool results do.
        """
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
        if "json" in self._formats:
            self._history = result.messages
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
            self._response_metadata.clear()

    async def on_loop_cancelled(self) -> None:
        """Flush current history and preserve crash-forensic sidecars."""
        self._close_jsonl()
        self._withdraw_jsonl_path()
        try:
            self._flush_json(force=True)
        finally:
            self._close_message_spool()
            self._history = None
            self._response_metadata.clear()
