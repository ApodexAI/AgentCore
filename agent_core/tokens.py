"""Token estimation for messages.

The estimate itself lives in
:func:`agent_core.runtime.loop.context_budget.estimate_tokens`; this
module adds the message-shaped wrapper the loop and the finalization recovery
path import (both via ``loop.llm_client``, which re-exports them).

One estimator, deliberately. ``llm_client`` used to define a second one on its
own tiktoken loader, and the two disagreed by ~4x on the path that matters: its
fallback was ``len(text) // 4`` with no CJK awareness, while
``context_budget``'s counts each CJK character as a token. Measured on the
fallback path: 4.00x under on pure CJK, 2.58x under on realistic mixed Chinese
(digits and latin interleaved), 1.00x -- identical -- on English. The fallback is
not the rare path: ``get_encoding_nonblocking`` returns ``None`` for as long as
tiktoken is still loading on its daemon thread, i.e. the opening turns of every
process. So the context-overflow guard, the main consumer, was undercounting
Chinese tool results by 2.5-4x and failing to fire before the provider's context
limit -- and its 1.5x buffer does not cover that.

What that estimator did better is kept here: OpenAI-shaped ``tool_calls`` are
counted (see :func:`_tool_calls_text`) -- reading ``name`` / ``args`` off each
call measured a realistic search call at 8 tokens against an actual 39. What is NOT kept is its preference for
``o200k_base`` over ``cl100k_base``. Both are approximations for the models
actually in use, they differ by roughly a tenth on English, and the guard
already carries a 1.5x buffer and a 1000-token floor — not worth a second
encoder cache and a second warm-up window.
"""

from __future__ import annotations

import json
from typing import Any, cast

from agent_core.messages import text_of

# Per-message wire overhead (role, delimiters, the trailing separator). Same
# constant ``compact.estimate_tokens`` adds when it sums a whole history.
_PER_MESSAGE_OVERHEAD = 4


def estimate_text_tokens(text: str) -> int:
    """Estimate token count for a text string."""
    from agent_core.runtime.loop.context_budget import estimate_tokens

    return estimate_tokens(text)


def _tool_calls_text(tool_calls: Any) -> str:
    """Serialise ``tool_calls`` the way their token cost is actually incurred.

    Reading named keys off each call was the earlier approach and it silently
    measured nothing: it looked for the legacy flat ``name`` / ``args``
    shape, while the canonical shape in ``core.messages.ToolCall`` is OpenAI's
    ``{"id", "type", "function": {"name", "arguments"}}``. Every OpenAI-shaped
    tool call therefore contributed zero, and tool arguments are routinely the
    largest part of an assistant turn.

    JSON is both shape-agnostic and closer to the truth, since the punctuation
    it counts is punctuation that goes on the wire. ``ensure_ascii=False`` keeps
    CJK as CJK — escaping it to ``\\uXXXX`` would inflate the estimate ~6x
    against the very heuristic that exists to get CJK right.
    """
    try:
        return json.dumps(tool_calls, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(tool_calls)


def estimate_message_tokens(message: Any) -> int:
    """Estimate total tokens for a Message dict or object."""
    if not message:
        return 0
    if isinstance(message, dict):
        message_dict = cast(dict[str, Any], message)
        content = message_dict.get("content") or ""
        tool_calls = message_dict.get("tool_calls")
    else:
        content = getattr(message, "content", "") or ""
        tool_calls = getattr(message, "tool_calls", None)

    # ``text_of`` rather than ``str``: Anthropic returns content as a block list,
    # and its repr would charge the estimate for dict punctuation and the
    # thinking block's key names.
    tokens = estimate_text_tokens(text_of(content)) + _PER_MESSAGE_OVERHEAD
    if tool_calls:
        tokens += estimate_text_tokens(_tool_calls_text(tool_calls))
    return tokens
