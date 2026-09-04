"""State-aware context compaction for agent loops.

Rolling-summary helper that the caller invokes BEFORE each LLM call. On
each invocation:
  - If `messages` is under `threshold`, returns the input unchanged.
  - Otherwise, compacts the middle slice into a single rolling summary
    via `summary_llm`, keeping the system message and the last
    `keep_recent` turns. Returns the new list + `did_compact=True`.

Caller is expected to mutate its persistent message list when
`did_compact=True`:

    new_msgs, compacted = await compact_if_needed(messages, ...)
    if compacted:
        messages[:] = new_msgs

This is the state-aware counterpart to `SummarizationMiddleware`, which
runs every LLM call and re-summarizes the same persistent state from
scratch (stateless `before_llm`). State-aware avoids re-summarizing on
every call and keeps cost O(number-of-threshold-crossings) rather than
O(turns-past-threshold).

The summary prompt explicitly asks the summarizer to preserve entity
names, ruled-out candidates, consulted URLs, and verified facts — the
information types most often eroded by repeated summarization. Because
we use a rolling summary (not chained), each compaction passes the
previous summary back through summarization. The preservation prompt is
the safety net against drift.
"""

from __future__ import annotations

import logging
import re
from typing import Any, cast

from agent_core.messages import (
    Message,
    text_of,
    user_msg,
)
from agent_core.runtime.loop.summary_prompt import (
    COMPACTION_PROMPT as _COMPACTION_PROMPT,
)
from agent_core.runtime.loop.summary_prompt import (
    format_conversation_for_summary as _format_conversation_for_summary,
)

logger = logging.getLogger(__name__)


_MSG_OVERHEAD = 4
_CJK_RE = re.compile(r"[　-〿一-鿿぀-ゟ゠-ヿ]")


def _estimate_tokens_heuristic(messages: list[Message]) -> int:
    """Regex heuristic for mixed CJK/English text. ~10% accuracy.

    Used as fallback when tiktoken is unavailable.
    """
    total = 0
    for m in messages:
        text = text_of(m.get("content"))
        cjk_count = len(_CJK_RE.findall(text))
        other_count = len(text) - cjk_count
        total += cjk_count + (other_count // 4) + _MSG_OVERHEAD
    return total


def estimate_tokens(messages: list[Message]) -> int:
    """Estimate the token count of a message list.

    Prefers tiktoken cl100k_base (accurate for OpenAI-family models).
    Falls back to a CJK-aware character heuristic otherwise. The encoder
    loads on a daemon thread (see ``tokenizer.py``) so this never makes a
    synchronous network fetch on the event-loop thread — the historical
    cause of multi-minute loop wedges in egress-restricted containers.
    """
    from agent_core.runtime.loop.tokenizer import get_encoding_nonblocking
    encoder = get_encoding_nonblocking("cl100k_base")
    if encoder is not None:
        try:
            total = 0
            for m in messages:
                total += len(
                    encoder.encode(
                        text_of(m.get("content")), disallowed_special=()
                    )
                ) + _MSG_OVERHEAD
            return total
        except Exception:
            pass
    return _estimate_tokens_heuristic(messages)


async def _generate_summary(
    summary_llm: Any,
    to_summarize: list[Message],
) -> str:
    """Run the summarizer LLM to compress `to_summarize` into one block."""
    conversation = _format_conversation_for_summary(to_summarize)
    prompt = _COMPACTION_PROMPT.format(conversation=conversation)
    resp = await summary_llm.chat([user_msg(prompt)])
    text = getattr(resp, "content", None) or ""
    if isinstance(text, list):
        blocks = cast("list[Any]", text)
        text = "".join(
            str(cast("dict[str, Any]", c).get("text", ""))
            if isinstance(c, dict)
            else str(c)
            for c in blocks
        )
    return text.strip() if isinstance(text, str) else str(text)


async def compact_if_needed(
    messages: list[Message],
    *,
    threshold: int,
    keep_recent: int,
    summary_llm: Any,
    task_id: str = "",
) -> tuple[list[Message], bool]:
    """Compact `messages` if its estimated token count exceeds `threshold`.

    Returns `(new_messages, did_compact)`. When `did_compact=True` the
    caller MUST replace its persistent state with `new_messages`
    (typically `messages[:] = new_messages`) — this helper does NOT
    mutate the input list.

    Layout of the compacted output:
        [system_msg?, summary_msg, *recent_keep_recent_messages]

    The summary message is a `user` message containing a structured
    rollup that preserves entity names, ruled-out candidates, consulted
    URLs, and verified facts (see `_COMPACTION_PROMPT`).

    Args:
        messages: the agent's current persistent message list.
        threshold: token count above which compaction triggers.
        keep_recent: number of recent messages to keep raw (uncompressed).
        summary_llm: an :class:`LLMClient`-shaped LLM for the summary call.
        task_id: optional task id for logging.
    """
    token_est = estimate_tokens(messages)
    if token_est <= threshold or len(messages) <= keep_recent + 1:
        return messages, False

    # Split: optional leading system message + middle (to summarize) + tail
    sys_msgs = [m for m in messages[:1] if m.get("role") == "system"]
    rest = messages[len(sys_msgs):]

    split_idx = max(0, len(rest) - keep_recent)
    # Don't start the kept window on an orphan tool message (no matching
    # assistant message with tool_calls earlier in the kept window).
    while split_idx < len(rest) - 1 and rest[split_idx].get("role") == "tool":
        split_idx += 1

    to_summarize = rest[:split_idx]
    keep = rest[split_idx:]

    if not to_summarize:
        # Below-keep_recent middle — nothing to compact.
        return messages, False

    try:
        summary_text = await _generate_summary(summary_llm, to_summarize)
    except Exception as e:
        logger.warning(
            "compact_if_needed[task=%s]: summary LLM failed (%s), "
            "falling back to drop-old truncation",
            task_id, e,
        )
        # Fallback: drop the middle entirely with a placeholder note.
        # Better than crashing the agent loop on summarization failure.
        summary_text = (
            f"[Summary unavailable — {len(to_summarize)} earlier "
            f"messages dropped. Recent context preserved below.]"
        )

    summary_msg = user_msg(
        "[Compacted summary of earlier turns — older raw messages "
        "have been replaced by this rollup. Continue from here.]\n\n"
        + summary_text
    )

    new_messages = [*sys_msgs, summary_msg, *keep]
    new_token_est = estimate_tokens(new_messages)
    logger.info(
        "compact_if_needed[task=%s]: compacted %d→%d msgs "
        "(%d→%d tokens, threshold=%d)",
        task_id, len(messages), len(new_messages),
        token_est, new_token_est, threshold,
    )
    return new_messages, True
