"""Generic LLM output repair — fix structural noise in raw model output.

Master design Phase 3 PR-3.3 / §5.10. Distilled / open-weights chat models
occasionally emit malformed reasoning markup that downstream parsers
choke on. The two patterns observed in the wild:

- **Duplicated closing tags**: ``</think></think></think>`` instead of a
  single ``</think>``. Some R1-distilled models loop on the closing
  token when greedy decoding caps a long chain-of-thought.
- **Unclosed thinking blocks**: an opening ``<think>`` / ``<thinking>``
  with no matching close, usually because the model hit ``max_tokens``
  mid-CoT. The thinking content then bleeds into whatever consumes the
  message body.

Both are silent failures — the chat completion is *successful* by API
status but the markup is broken. This middleware runs after every LLM
call and rewrites the content using :func:`dataclasses.replace` so every
other ``LLMResponse`` field (tool_calls, reasoning_content, usage, …) is
preserved automatically.

Phase 4 will move this module to ``components/middleware/llm/`` per the
final folder structure design; the current location keeps the import
graph simple for the V1 light SDK.

Placement
---------
Wired as the **last** entry in
``build_default_research_llm_middleware_chain``. ``LLMMiddlewareChain``
runs ``after_llm`` hooks in *reverse* registration order (onion model),
so being last makes this middleware the innermost: it fires first in
``after_llm`` and every outer middleware (tracing, token accounting,
loop detection) sees the same canonical repaired content that the agent
loop / tool parser ultimately consume.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from typing import Any, cast

from agent_core.components.middleware.llm.base import (
    LLMCallContext,
    LLMMiddleware,
)
from agent_core.llm import LLMResponse

logger = logging.getLogger(__name__)

__all__ = ["OutputRepairMiddleware", "repair_output_text"]


# Match a closing tag followed by optional whitespace and another copy of
# the same closing tag. Iterated until the text stabilises so that runs
# of three or more collapse cleanly: ``</think></think></think>`` → one.
_DUP_CLOSE_RE = re.compile(
    r"</(think|thinking)>\s*</\1>",
    flags=re.IGNORECASE,
)

_OPEN_THINK_RE = re.compile(r"<(think|thinking)>", flags=re.IGNORECASE)
_CLOSE_THINK_RE = re.compile(r"</(think|thinking)>", flags=re.IGNORECASE)


def repair_output_text(text: str) -> str:
    """Apply the three repair rules to a single text segment.

    1. Collapse consecutive ``</think>`` / ``</thinking>`` runs to one.
    2. If opens > closes, append a matching close at the end.
    3. Strip trailing whitespace.

    Returns the input unchanged when no rule matched — callers can use
    identity comparison to detect a no-op.
    """
    if not text:
        return text

    # Hot-path early exit: most production LLMs (gpt-5 / gemini / claude)
    # never emit thinking tags. Skip the dedup loop + two findall passes
    # entirely when the text contains neither tag form, so the only work
    # left is a trailing-whitespace check.
    lowered = text.lower()
    if "<think" not in lowered and "</think" not in lowered:
        stripped = text.rstrip()
        return text if stripped == text else stripped

    repaired = text
    # Iterate until the dedup regex stops matching. A single ``re.sub``
    # only collapses one pair per region, so 3+ duplicates need a loop.
    while True:
        new = _DUP_CLOSE_RE.sub(r"</\1>", repaired)
        if new == repaired:
            break
        repaired = new

    opens = _OPEN_THINK_RE.findall(repaired)
    closes = _CLOSE_THINK_RE.findall(repaired)
    if len(opens) > len(closes):
        # Match the *first* unclosed open's tag style so we don't mix
        # ``<think>`` openers with ``</thinking>`` closers.
        tag = opens[len(closes)].lower()
        repaired = repaired + f"</{tag}>"

    return repaired.rstrip()


def _repair_content(content: Any) -> Any:
    """Apply ``repair_output_text`` to an ``LLMResponse.content`` payload.

    Content is either ``str`` (OpenAI-style) or a list of dict / string
    blocks (Anthropic-style). We repair text in each shape and forward
    unknown block types untouched.
    """
    if isinstance(content, str):
        return repair_output_text(content)

    if isinstance(content, list):
        repaired_blocks: list[Any] = []
        for block in cast("list[Any]", content):
            text_value = (
                cast("dict[str, Any]", block).get("text")
                if isinstance(block, dict)
                else None
            )
            if isinstance(text_value, str):
                mapping = cast("dict[str, Any]", block)
                new_text = repair_output_text(text_value)
                if new_text == text_value:
                    repaired_blocks.append(mapping)
                else:
                    repaired_blocks.append({**mapping, "text": new_text})
            elif isinstance(block, str):
                repaired_blocks.append(repair_output_text(block))
            else:
                repaired_blocks.append(block)
        return repaired_blocks

    return content


class OutputRepairMiddleware(LLMMiddleware):
    """Tail-end LLM middleware that fixes structural output noise.

    Rules (see ``repair_output_text``):
        1. Dedupe consecutive ``</think>`` / ``</thinking>`` tokens.
        2. Auto-close an unclosed ``<thinking>`` block.
        3. Trim trailing whitespace.

    Args:
        enabled: ``False`` short-circuits ``after_llm`` to a pass-through.
            Useful for benchmarks comparing raw vs repaired output.
    """

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = enabled

    @property
    def name(self) -> str:
        return "output_repair"

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def after_llm(
        self, ctx: LLMCallContext, response: LLMResponse,
    ) -> LLMResponse:
        repaired = _repair_content(response.content)
        if repaired is response.content or repaired == response.content:
            return response

        logger.debug(
            "OutputRepairMiddleware: repaired output (task=%s, call=%d)",
            ctx.task_id, ctx.call_index,
        )
        return replace(response, content=repaired)
