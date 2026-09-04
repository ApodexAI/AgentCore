"""LLM retry middleware — exponential backoff for transient failures."""

from __future__ import annotations

import asyncio
import logging

from agent_core.components.middleware.llm.base import (
    LLMCallContext,
    LLMMiddleware,
)
from agent_core.retry_policy import legacy_retryable

logger = logging.getLogger(__name__)


class LLMRetryMiddleware(LLMMiddleware):
    """Transparent retry for transient LLM failures (timeout, 429, 5xx).

    Plugs into the ``on_llm_error`` hook added to ``LLMProxy``.
    When a call fails with a retryable error and the attempt count is
    below ``max_retries``, this middleware sleeps (exponential back-off
    with jitter) and returns ``True`` so the proxy re-issues the call.
    """

    def __init__(
        self,
        max_retries: int = 3,
        backoff_base: float = 0.5,
        backoff_max: float = 8.0,
    ) -> None:
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max

    @property
    def name(self) -> str:
        return "llm_retry"

    async def on_llm_error(
        self, ctx: LLMCallContext, error: Exception, attempt: int,
    ) -> bool:
        if attempt >= self._max_retries:
            return False
        if not legacy_retryable(error):
            return False

        import random
        delay = min(
            self._backoff_base * (2 ** attempt), self._backoff_max,
        ) + random.random() * 0.25
        logger.warning(
            "LLMRetryMiddleware: attempt %d/%d failed (%s), "
            "retrying in %.1fs",
            attempt + 1, self._max_retries,
            type(error).__name__, delay,
        )
        await asyncio.sleep(delay)
        return True
