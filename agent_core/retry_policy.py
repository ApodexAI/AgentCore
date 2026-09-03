"""Retryable-error policy shared by provider wrappers and LLM middleware.

A leaf module on purpose. Both :mod:`agent_core.providers.fallback` and
:mod:`agent_core.components.middleware.llm.base` need this one table, and
having either import the other's package to get it would add a
``components -> providers -> runtime`` import edge for the sake of a
frozenset — an edge that turns into a partially-initialised-module
``ImportError`` the day anything under ``runtime/`` imports the middleware.
Two copies of one policy drift, so the table lives here and both import it.
"""

from __future__ import annotations

__all__ = ["LEGACY_RETRYABLE_KEYWORDS", "legacy_retryable"]

LEGACY_RETRYABLE_KEYWORDS = frozenset({
    "timeout", "timed out", "429", "500", "502", "503", "504", "529",
    "overloaded", "rate limit", "rate_limit", "server error",
    "connection reset", "connection error", "econnreset", "gateway timeout",
    "model_dump", "model_not_found",
})


def legacy_retryable(error: Exception) -> bool:
    """Match the historical two-model fallback wrapper's retry policy."""
    return isinstance(error, AttributeError) or any(
        keyword in str(error).lower() for keyword in LEGACY_RETRYABLE_KEYWORDS
    )
