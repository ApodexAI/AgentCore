"""RateLimitMiddleware — LLM-layer token bucket rate limiter.

Issue #24 Phase C: prevents parallel agents from overwhelming API quotas.

Uses a simple token bucket algorithm with per-provider limits.
Queues on limit (asyncio.sleep), never rejects.
after_llm corrects estimates with actual token usage.
"""

from __future__ import annotations

import asyncio
import logging
import time

from agent_core.components.middleware.llm.base import LLMCallContext, LLMMiddleware
from agent_core.llm import LLMResponse
from agent_core.messages import Message, text_of

logger = logging.getLogger(__name__)


class TokenBucket:
    """Simple token bucket rate limiter.

    Tracks two resources: requests per minute and tokens per minute.
    Refills continuously based on elapsed time.
    """

    def __init__(
        self,
        requests_per_min: int = 60,
        tokens_per_min: int = 100_000,
    ) -> None:
        if requests_per_min <= 0:
            raise ValueError("requests_per_min must be positive")
        if tokens_per_min <= 0:
            raise ValueError("tokens_per_min must be positive")
        self._rpm = float(requests_per_min)
        self._tpm = float(tokens_per_min)

        # Buckets start full
        self._request_tokens = self._rpm
        self._token_tokens = self._tpm
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        """Refill buckets based on elapsed time."""
        now = time.monotonic()
        elapsed_min = (now - self._last_refill) / 60.0
        self._request_tokens = min(
            self._rpm,
            self._request_tokens + elapsed_min * self._rpm,
        )
        self._token_tokens = min(
            self._tpm,
            self._token_tokens + elapsed_min * self._tpm,
        )
        self._last_refill = now

    async def acquire(self, estimated_tokens: int = 0) -> float:
        """Acquire rate limit capacity. Returns wait time in seconds.

        If bucket is empty, calculates required wait time and sleeps.
        Returns the time spent waiting (0 if no wait was needed).
        """
        total_wait = 0.0
        # A single request cannot reserve more than a full minute's token
        # capacity. Capping avoids an infinite wait for an oversized prompt;
        # the provider remains the authority on whether that request is valid.
        requested_tokens = min(max(estimated_tokens, 0), int(self._tpm))

        while True:
            async with self._lock:
                self._refill()

                req_wait = 0.0
                if self._request_tokens < 1.0:
                    deficit = 1.0 - self._request_tokens
                    req_wait = (deficit / self._rpm) * 60.0

                tok_wait = 0.0
                if requested_tokens > 0 and self._token_tokens < requested_tokens:
                    deficit = requested_tokens - self._token_tokens
                    tok_wait = (deficit / self._tpm) * 60.0

                wait_time = max(req_wait, tok_wait)
                if wait_time <= 0:
                    # Capacity is checked and reserved in one critical section;
                    # no other waiter can consume the refill between them.
                    self._request_tokens -= 1.0
                    if requested_tokens > 0:
                        self._token_tokens -= requested_tokens
                    return total_wait

            logger.info(
                "RateLimit: queuing %.1fs (req_wait=%.1f, tok_wait=%.1f)",
                wait_time, req_wait, tok_wait,
            )
            await asyncio.sleep(wait_time)
            total_wait += wait_time

    def adjust(self, actual_tokens: int, estimated_tokens: int) -> None:
        """Correct token bucket with actual usage.

        If we over-estimated, give back the difference.
        If under-estimated, consume the difference.
        """
        diff = estimated_tokens - actual_tokens
        if diff != 0:
            self._token_tokens = min(
                self._tpm, self._token_tokens + diff,
            )


class RateLimitMiddleware(LLMMiddleware):
    """LLM middleware: token bucket rate limiter.

    Prevents parallel agents from overwhelming LLM API quotas.
    Queues requests when limits are hit — never rejects.
    """

    def __init__(
        self,
        requests_per_min: int = 60,
        tokens_per_min: int = 100_000,
    ) -> None:
        self._bucket = TokenBucket(requests_per_min, tokens_per_min)
        self._estimate_key = "_rate_limit_estimated_tokens"

    @property
    def name(self) -> str:
        return "rate_limit"

    async def before_llm(
        self,
        ctx: LLMCallContext,
        messages: list[Message],
    ) -> list[Message]:
        """Acquire rate limit capacity before LLM call."""
        # Rough estimate: ~4 chars per token
        estimated = sum(
            len(str(text_of(m.get("content")))) for m in messages
        ) // 4
        ctx.metadata[self._estimate_key] = estimated

        wait = await self._bucket.acquire(estimated)
        if wait > 0:
            ctx.metadata["rate_limit_wait_s"] = round(wait, 2)

        return messages

    async def after_llm(
        self,
        ctx: LLMCallContext,
        response: LLMResponse,
    ) -> LLMResponse:
        """Correct token bucket with actual usage from response."""
        usage = (
            response.usage
            or response.response_metadata.get("token_usage")
            or response.response_metadata.get("usage")
            or {}
        )
        actual_total = (
            usage.get("total_tokens")
            or usage.get("prompt_tokens", 0)
            + usage.get("completion_tokens", 0)
            or usage.get("input_tokens", 0)
            + usage.get("output_tokens", 0)
        )
        estimated = ctx.metadata.get(self._estimate_key, 0)

        if actual_total and estimated:
            self._bucket.adjust(actual_total, estimated)

        return response
