"""Normalization of provider-specific completion-stop signals.

``LLMResponse.finish_reason`` is a provider-neutral field, but each transport
names the "output token cap was hit" case differently: OpenAI Chat says
``length``, Anthropic Messages says ``max_tokens``, and the OpenAI Responses API
reports ``status="incomplete"`` with ``incomplete_details.reason
="max_output_tokens"``.

``agent_core.runtime.loop._runaway`` tests for exactly ``"length"`` — it is the
only evidence that can distinguish a reply cut off mid-sentence from a complete
one, and there is deliberately no token-count fallback once visible text is
present. An unmapped marker therefore silently disables truncation recovery for
that transport, so every client funnels its stop signal through here.

Only the truncation markers are rewritten. Every other value is passed through
unchanged, because ``tool_use``/``end_turn``/``stop`` carry provider-meaningful
detail that hosts and tests read directly.
"""

from __future__ import annotations

from typing import Any

FINISH_REASON_LENGTH = "length"

# Values, lowercased, that mean "the output token cap stopped generation".
TRUNCATION_MARKERS: frozenset[str] = frozenset({
    "length",             # OpenAI Chat Completions (already normalized)
    "max_completion_tokens",
    "max_output_tokens",  # OpenAI Responses incomplete_details.reason
    "max_tokens",         # Anthropic Messages stop_reason
    "model_length",
    "output_limit",
})


def normalize_finish_reason(value: Any) -> str:
    """Map a transport's truncation marker to ``"length"``; pass others through."""
    text = str(value or "").strip()
    if not text:
        return ""
    if text.lower() in TRUNCATION_MARKERS:
        return FINISH_REASON_LENGTH
    return text


def responses_finish_reason(status: Any, incomplete_reason: Any = None) -> str:
    """Fold a Responses-API ``status`` plus ``incomplete_details.reason``.

    ``status`` alone is never ``"length"`` — it is ``completed`` / ``incomplete``
    / ``failed`` — so a gpt-5 turn stopped at ``max_output_tokens`` is invisible
    without the nested reason.
    """
    reason = normalize_finish_reason(incomplete_reason)
    if reason:
        return reason
    return normalize_finish_reason(status)


__all__ = [
    "FINISH_REASON_LENGTH",
    "TRUNCATION_MARKERS",
    "normalize_finish_reason",
    "responses_finish_reason",
]
