"""Compact message history with an LLM and a bounded rollback fallback.

If summarization fails, the compactor keeps only the system message, a
placeholder, and the latest user query so the next turn fits its budget.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any, cast

from agent_core.messages import (
    Message,
    text_of,
    user_msg,
)
from agent_core.runtime.loop.compact import (
    SPILL_MANIFEST_HEADER,
    compress_tool_results,
    estimate_tokens,
    partition_for_compaction,
    tool_names_by_call_id,
)
from agent_core.runtime.loop.compact import (
    compact_messages as _string_slice_compact,
)
from agent_core.runtime.loop.summary_prompt import (
    compaction_prompt,
    format_conversation_for_summary,
)
from agent_core.runtime.retriable import (
    is_context_length_error,
    is_transient_network,
)

__all__ = [
    "CompactionEventEmitter",
    "LLMSummaryCompactor",
    "SummaryPromptBuilder",
    "is_transient_summary_error",
]

logger = logging.getLogger(__name__)

# Strong references to in-flight fire-and-forget event emits. asyncio only holds
# a weak reference to a running task, so without this a compaction event could be
# garbage collected before it is delivered.
_PENDING_EMITS: set[asyncio.Task[Any]] = set()


# Substrings that identify a *deterministic* summariser failure — retrying one
# of these burns the retry budget on a guaranteed second failure. The dominant
# case is the summariser being handed more than it can read, which is exactly
# what happens on the runs that need compaction most.
_DETERMINISTIC_ERROR_MARKERS = (
    "context length",
    "context_length",
    "maximum context",
    "too many tokens",
    "too large",
    "invalid_request",
    "invalid request",
    "bad request",
    "unauthorized",
    "forbidden",
    "not found",
    "model_not_found",
)

# Substrings that identify a *transient* failure worth one more attempt.
_TRANSIENT_ERROR_MARKERS = (
    "timeout",
    "timed out",
    "connection",
    "temporarily",
    "unavailable",
    "overloaded",
    "rate limit",
    "rate_limit",
    "too many requests",
)

# Status codes, kept apart from the substrings above because a bare number is
# not a substring worth matching: provider messages carry request ids, durations
# and quota hints, so "Rate limit reached — try again in 400ms" used to classify
# a textbook retriable 429 as a permanent 400 and skip its whole retry budget.
# Matched only as a standalone token, and only after the exception's own status
# field — which is authoritative when the provider exposes one.
_DETERMINISTIC_STATUS_CODES = frozenset({400, 401, 403, 404, 413, 422})
_TRANSIENT_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504, 529})
_STATUS_ATTRS = ("status_code", "status", "http_status")
# Neither side of a code may abut a word character, a dot or a percent sign:
# ``400ms``, ``req_400abc``, ``0.400`` and ``400%`` are not status codes.
_CODE_PATTERNS = {
    code: re.compile(rf"(?<![\w.]){code}(?![\w.%])")
    for code in _DETERMINISTIC_STATUS_CODES | _TRANSIENT_STATUS_CODES
}


def _status_code(exc: BaseException) -> int | None:
    """The HTTP status an exception carries, if it names one at all."""

    candidates: list[Any] = [getattr(exc, attr, None) for attr in _STATUS_ATTRS]
    response = getattr(exc, "response", None)
    if response is not None:
        candidates.extend(getattr(response, attr, None) for attr in _STATUS_ATTRS)
    for value in candidates:
        if isinstance(value, bool):
            continue
        if isinstance(value, str):
            value = value.strip()
            if not value.isdigit():
                continue
            value = int(value)
        if isinstance(value, int) and 100 <= value <= 599:
            return value
    return None


def _mentions_status(text: str, codes: frozenset[int]) -> bool:
    return any(_CODE_PATTERNS[code].search(text) for code in codes)


# Backoff between transient summary retries: ``min(base * attempt, cap)``.
# Module constants for the same reason ``_runaway.py`` keeps its own — the values
# are a judgement about endpoint hiccups, not a per-call parameter, and a test
# has to be able to shorten them without waiting out a real backoff.
_RETRY_BACKOFF_BASE_S = 2.0
_RETRY_BACKOFF_CAP_S = 5.0


def is_transient_summary_error(exc: BaseException) -> bool:
    """Whether a summariser failure is worth retrying.

    Retrying a deterministic error — a 4xx, and above all "context too large" —
    is guaranteed to fail again while consuming the retry budget. A single
    summariser call can block for the full LLM timeout (600s in the shipped
    profiles), so getting this wrong costs minutes, not milliseconds. That
    asymmetry is why the classification exists at all rather than a flat retry
    count.

    Unrecognised errors are treated as transient: one extra attempt is cheaper
    than losing the whole research history to the fallback layout. The bias is
    deliberate and bounded on the other side by ``retry_total_timeout_s``.
    """
    if isinstance(exc, (asyncio.TimeoutError, ConnectionError, TimeoutError)):
        return True
    if isinstance(exc, asyncio.CancelledError):
        return False
    # Keep this ordering aligned with the main LLM retry path. Context overflow
    # is deterministic even when a provider's message mentions transport-like
    # details. Conversely, OpenAI-compatible gateways sometimes wrap an
    # upstream 5xx/timeout in an HTTP 400 response (``new_api_error`` /
    # ``bad_response_status_code``); the wrapper status must not suppress the
    # transient signal carried in the structured body.
    if is_context_length_error(exc):
        return False
    if is_transient_network(exc):
        return True
    # A structured status beats any reading of the message text.
    status = _status_code(exc)
    if status is not None:
        if status in _TRANSIENT_STATUS_CODES:
            return True
        if status in _DETERMINISTIC_STATUS_CODES:
            return False
    text = f"{type(exc).__name__}: {exc}".lower()
    for marker in _DETERMINISTIC_ERROR_MARKERS:
        if marker in text:
            return False
    for marker in _TRANSIENT_ERROR_MARKERS:
        if marker in text:
            return True
    # Codes last, and transient first among them: a rate limit is a 4xx, so a
    # message naming both ("429 ... retry after 400ms") must not read as
    # permanent.
    if _mentions_status(text, _TRANSIENT_STATUS_CODES):
        return True
    return not _mentions_status(text, _DETERMINISTIC_STATUS_CODES)


# Type alias for the optional event emitter callback. Signature matches
# what the SDK's event sink expects: a payload dict; the caller wraps
# the actual transport (event_store.append / stdout JSONL / no-op).
CompactionEventEmitter = Callable[[dict[str, Any]], Awaitable[None] | None]
SummaryPromptBuilder = Callable[[list[Message]], str]


def _last_human_message(messages: list[Message]) -> Message | None:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg
    return None


class LLMSummaryCompactor:
    """Compactor that delegates summarization to an LLM call.

    Implements the ``MessageCompactor`` Protocol with an *async* compact
    method. The agent loop awaits the result transparently — see the
    awaitable-aware call site in ``agent_loop.py`` step 10.

    Failure handling
    ----------------
    - **Summary LLM raises**: log, emit ``compaction`` event with
      ``rolled_back=True``, return rollback layout
      ``[system, "[Compaction failed — earlier turns dropped]", last_user]``.
    - **Summary returns empty / whitespace**: same as raise — rollback
      with placeholder text.
    - **Successful summary**: emit ``compaction`` event with
      ``rolled_back=False``, return ``[system, HumanMessage(summary), recent]``.

    Args:
        summary_llm: any object with ``async def chat(messages)``
            returning an object with ``.content`` (see ``_generate_summary``).
            ``None`` makes the
            compactor delegate to the deterministic string-slice path
            (no LLM call, no rollback) — useful when no summarizer is
            available but you still want a Protocol-conforming compactor.
        emit_event: optional callback that receives the ``compaction``
            event payload. The SDK's stdout JSONL emitter wires this in;
            other callers pass ``None`` and the event is just logged.
        failure_fallback: ``"panic"`` preserves the legacy tiny rollback;
            ``"deterministic"`` keeps a bounded string-slice summary instead.
        max_transient_retries: extra attempts for a *transient* summariser
            failure. **Defaults to zero**, so every existing direct caller
            (``apodex.session``, ``apodex.task_runner``, ``session_history``)
            keeps its historical single-attempt behaviour. Those callers do not
            pass ``retry_total_timeout_s``, which would make any retry they
            inherited unbounded in time — the reason this is opt-in rather
            than on by default. Deterministic failures never retry.
        retry_total_timeout_s: wall-clock ceiling across ALL attempts, backoff
            included. Enforced with ``asyncio.wait_for`` around each attempt,
            not merely checked between them: a summariser call carries its own
            (much larger) transport timeout, so a between-attempts check is not
            a ceiling. ``None`` disables the bound, which is only safe when
            ``max_transient_retries`` is zero.
    """

    def __init__(
        self,
        *,
        summary_llm: Any = None,
        emit_event: CompactionEventEmitter | None = None,
        failure_fallback: str = "panic",
        max_transient_retries: int = 0,
        retry_total_timeout_s: float | None = None,
        prompt_builder: SummaryPromptBuilder = compaction_prompt,
    ) -> None:
        self._summary_llm = summary_llm
        self._emit_event = emit_event
        self._failure_fallback = failure_fallback
        self._max_transient_retries = max(0, int(max_transient_retries))
        self._retry_total_timeout_s = retry_total_timeout_s
        self._prompt_builder = prompt_builder
        if self._max_transient_retries and not retry_total_timeout_s:
            # Not fatal — a caller may genuinely own the outer deadline — but it
            # is the shape of the defect this parameter pair exists to prevent,
            # so it must not pass silently.
            logger.warning(
                "LLMSummaryCompactor: max_transient_retries=%d without "
                "retry_total_timeout_s — retries are unbounded in time",
                self._max_transient_retries,
            )

    async def compact(
        self,
        messages: list[Message],
        keep_recent: int,
        *,
        compress_all_tool_results: bool = False,
        preserve_tool_names: frozenset[str] = frozenset(),
    ) -> list[Message]:
        # No LLM configured → behave as the string-slice fallback. This
        # keeps the SDK construction simple: callers can pass
        # ``LLMSummaryCompactor()`` and trust it never crashes on a
        # missing summarizer.
        source_messages = (
            compress_tool_results(messages) if compress_all_tool_results else messages
        )
        if self._summary_llm is None:
            return self._string_slice(source_messages, keep_recent)

        sys_msgs, middle, recent = self._partition(source_messages, keep_recent)
        if not middle:
            return source_messages  # nothing to summarize

        # Compute once; reused by both the success and rollback paths so
        # we don't re-walk the full message list per emit.
        tokens_before = estimate_tokens(source_messages)

        summary_text, attempts, failure_reason, failure_error = (
            await self._generate_summary_with_retry(
                middle, preserve_tool_names=preserve_tool_names,
            )
        )

        if failure_reason:
            logger.warning(
                "LLMSummaryCompactor: %s after %d attempt(s) (%s) — rolling back",
                failure_reason, attempts, failure_error or "no detail",
            )
            return await self._rollback(
                source_messages,
                sys_msgs,
                len(middle),
                keep_recent=keep_recent,
                tokens_before=tokens_before,
                reason=failure_reason,
                error=failure_error,
                attempts=attempts,
            )

        if not summary_text:
            logger.warning(
                "LLMSummaryCompactor: empty summary text — rolling back",
            )
            return await self._rollback(
                source_messages,
                sys_msgs,
                len(middle),
                keep_recent=keep_recent,
                tokens_before=tokens_before,
                reason="empty_summary",
                attempts=attempts,
            )

        summary_msg = user_msg(
            "[Compacted summary of earlier turns — older raw messages "
            "have been replaced by this rollup. Continue from here.]\n\n"
            + summary_text
        )
        new_messages: list[Message] = [*sys_msgs, summary_msg, *recent]
        await self._emit(
            {
                "rolled_back": False,
                "messages_before": len(messages),
                "messages_after": len(new_messages),
                "tokens_before": tokens_before,
                "tokens_after": estimate_tokens(new_messages),
                "compactor": "llm",
                "attempts": attempts,
                # The summary IS the compaction's product: every token the model
                # keeps of the replaced turns passes through here. Without it an
                # observer can report that a compaction happened and how much it
                # freed, but nothing about whether what survived was worth
                # keeping — which is the only question a reader of the record
                # actually has.
                "summary": summary_text,
            }
        )
        return new_messages

    @staticmethod
    def _partition(
        messages: list[Message],
        keep_recent: int,
    ) -> tuple[list[Message], list[Message], list[Message]]:
        return partition_for_compaction(messages, keep_recent)

    async def _generate_summary_with_retry(
        self,
        middle: list[Message],
        *,
        preserve_tool_names: frozenset[str] = frozenset(),
    ) -> tuple[str, int, str, str]:
        """Summarize with bounded retries → ``(text, attempts, reason, error)``.

        ``reason`` is empty on success. Only transient failures are retried, and
        the sequence is additionally bounded by ``retry_total_timeout_s`` so a
        stuck endpoint cannot stall the loop for ``retries × llm_timeout``.

        A permanent failure returns ``llm_error_permanent`` rather than
        ``llm_error``: both roll back identically, but a reader of the record
        needs to know the difference between "the endpoint hiccuped and we ran
        out of attempts" and "this summary can never succeed at this size",
        because only the second one means the relief target is unreachable.
        """
        deadline = (
            time.monotonic() + self._retry_total_timeout_s
            if self._retry_total_timeout_s is not None
            and self._retry_total_timeout_s > 0
            else None
        )
        attempts = 0
        last_error = ""
        while True:
            remaining = deadline - time.monotonic() if deadline is not None else None
            if remaining is not None and remaining <= 0:
                return (
                    "",
                    attempts,
                    "llm_error",
                    last_error or "summary retry deadline exceeded",
                )

            attempts += 1
            try:
                if remaining is None:
                    summary = await self._generate_summary(
                        middle, preserve_tool_names=preserve_tool_names,
                    )
                else:
                    # The deadline is a ceiling for the whole sequence, not a
                    # flag checked after an individual call has already spent
                    # its own transport timeout.
                    summary = await asyncio.wait_for(
                        self._generate_summary(
                            middle, preserve_tool_names=preserve_tool_names,
                        ),
                        timeout=remaining,
                    )
                return summary, attempts, "", ""
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error = str(exc) or type(exc).__name__
                if not is_transient_summary_error(exc):
                    logger.info(
                        "LLMSummaryCompactor: deterministic summariser error, "
                        "not retrying (%s)", exc,
                    )
                    return "", attempts, "llm_error_permanent", last_error
                exhausted = attempts > self._max_transient_retries
                out_of_time = deadline is not None and time.monotonic() >= deadline
                if exhausted or out_of_time:
                    return "", attempts, "llm_error", last_error
                # Short fixed backoff: the failure modes worth retrying here are
                # endpoint hiccups, and the call itself already took seconds.
                delay = min(_RETRY_BACKOFF_BASE_S * attempts, _RETRY_BACKOFF_CAP_S)
                if deadline is not None and deadline - time.monotonic() <= delay:
                    # Refuse a retry whose backoff alone would consume the rest
                    # of the budget. Returning now is faster and keeps the
                    # advertised ceiling strict rather than approximate.
                    return "", attempts, "llm_error", last_error
                await asyncio.sleep(delay)

    async def _generate_summary(
        self,
        to_summarize: list[Message],
        *,
        preserve_tool_names: frozenset[str] = frozenset(),
    ) -> str:
        id_to_name = (
            tool_names_by_call_id(to_summarize) if preserve_tool_names else {}
        )
        preserved_ids = frozenset(
            call_id
            for call_id, name in id_to_name.items()
            if name in preserve_tool_names
        )
        # Drop the spill index, the way ``compact_messages`` already does on the
        # deterministic path: it is a list of paths, there is nothing in it to
        # summarize, and ``TieredCompactor`` re-attaches the real one afterwards
        # from the refs it collected. Nothing breaks if a summary now quotes the
        # header — no code reads an index back out of prose — but spending
        # summarizer budget on it is still waste. The header check is the legacy
        # path for a history checkpointed before the field existed.
        conversation = format_conversation_for_summary(
            [
                message
                for message in to_summarize
                if not message.get("spill_refs")
                and SPILL_MANIFEST_HEADER not in text_of(message.get("content"))
            ],
            preserve_tool_result_ids=preserved_ids,
        )
        prompt = self._prompt_builder(to_summarize).format(conversation=conversation)
        resp = await self._summary_llm.chat([user_msg(prompt)])
        text: Any = getattr(resp, "content", None) or ""
        if isinstance(text, list):
            # Anthropic-style content blocks: list of {"type":"text","text":...}
            parts: list[str] = []
            for block in cast(list[object], text):
                if isinstance(block, dict):
                    value = cast(dict[str, Any], block).get("text", "")
                    parts.append(str(value))
                else:
                    parts.append(str(block))
            text = "".join(parts)
        if not isinstance(text, str):
            text = str(text)
        return text.strip()

    async def _rollback(
        self,
        messages: list[Message],
        sys_msgs: list[Message],
        dropped: int,
        *,
        keep_recent: int,
        tokens_before: int,
        reason: str,
        error: str | None = None,
        attempts: int = 1,
    ) -> list[Message]:
        """Use deterministic compaction when requested, else panic truncation.

        The legacy mode keeps only the system prefix, a failure marker, and the
        latest user message. Tiered compaction opts into deterministic slicing,
        whose size is verified by the caller before it is selected.
        """
        if self._failure_fallback == "deterministic":
            new_messages = _string_slice_compact(messages, keep_recent)
        else:
            last_user = _last_human_message(messages)
            placeholder = user_msg(
                f"[Compaction failed — {dropped} earlier turns dropped. "
                f"Reason: {reason}.]"
            )
            new_messages = [*sys_msgs, placeholder]
            if last_user is not None and last_user not in sys_msgs:
                new_messages.append(last_user)

        payload: dict[str, Any] = {
            "rolled_back": True,
            "rollback_reason": reason,
            "messages_before": len(messages),
            "messages_after": len(new_messages),
            "tokens_before": tokens_before,
            "tokens_after": estimate_tokens(new_messages),
            "compactor": "llm",
            "attempts": attempts,
        }
        if error:
            payload["error"] = error
        await self._emit(payload)
        return new_messages

    def _string_slice(
        self,
        messages: list[Message],
        keep_recent: int,
    ) -> list[Message]:
        new_messages = _string_slice_compact(messages, keep_recent)
        if keep_recent > 0:
            original_task = next(
                (
                    message
                    for message in messages
                    if message.get("role") == "user"
                    and not text_of(message.get("content")).startswith("[Compacted")
                ),
                None,
            )
            if original_task is not None and not any(
                message is original_task for message in new_messages
            ):
                insert_at = 0
                while (
                    insert_at < len(new_messages)
                    and new_messages[insert_at].get("role") == "system"
                ):
                    insert_at += 1
                new_messages = [
                    *new_messages[:insert_at],
                    original_task,
                    *new_messages[insert_at:],
                ]
        # Synchronous emit: schedule async path on a dummy loop only when
        # an emitter exists. Most callers pass ``emit_event=None`` (SDK
        # default), so the common case is a pure-sync no-op.
        if self._emit_event is not None:
            try:
                rv = self._emit_event(
                    {
                        "rolled_back": False,
                        "messages_before": len(messages),
                        "messages_after": len(new_messages),
                        "tokens_before": estimate_tokens(messages),
                        "tokens_after": estimate_tokens(new_messages),
                        "compactor": "string",
                    }
                )
                if inspect.isawaitable(rv):
                    # Best-effort fire-and-forget. The string-slice path
                    # is sync so we cannot reliably await; the caller
                    # should wire async emitters at the LLM-summary path.
                    with contextlib.suppress(RuntimeError):
                        # Keep a reference: a bare ensure_future can be garbage
                        # collected before it runs. Discarded via the callback
                        # once it settles.
                        task = asyncio.ensure_future(rv)
                        _PENDING_EMITS.add(task)
                        task.add_done_callback(_PENDING_EMITS.discard)
            except Exception:
                logger.debug("compaction event emitter raised", exc_info=True)
        return new_messages

    async def _emit(self, payload: dict[str, Any]) -> None:
        if self._emit_event is None:
            return
        try:
            rv = self._emit_event(payload)
            if inspect.isawaitable(rv):
                await rv
        except Exception:
            logger.debug("compaction event emitter raised", exc_info=True)
