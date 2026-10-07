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
import uuid

from agent_core.completion import reported_usage, usage_count
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

    def reservation_size(self, estimated_tokens: int) -> int:
        """Amount actually reserved, capped at one full minute's capacity."""
        return min(max(estimated_tokens, 0), int(self._tpm))

    async def acquire(self, estimated_tokens: int = 0) -> float:
        """Acquire rate limit capacity. Returns wait time in seconds.

        If bucket is empty, calculates required wait time and sleeps.
        Returns the time spent waiting (0 if no wait was needed).
        """
        total_wait = 0.0
        # A single request cannot reserve more than a full minute's token
        # capacity. Capping avoids an infinite wait for an oversized prompt;
        # the provider remains the authority on whether that request is valid.
        requested_tokens = self.reservation_size(estimated_tokens)

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
        self._reserved_key = f"_rate_limit_reserved_tokens_{uuid.uuid4().hex}"

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
        # acquire caps oversized reservations at one full token bucket. Correct
        # against what was reserved, not the uncapped prompt estimate.
        ctx.metadata[self._reserved_key] = self._bucket.reservation_size(estimated)

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
        usage = reported_usage(response)
        if usage is None:
            # Keep the original reservation when provider usage is unknown.
            return response
        actual_total = usage_count(usage, "total_tokens")
        if actual_total is None:
            inp = usage_count(usage, "prompt_tokens", "input_tokens")
            out = usage_count(usage, "completion_tokens", "output_tokens")
            if inp is None and out is None:
                return response
            actual_total = (inp or 0) + (out or 0)
        reserved = usage_count(ctx.metadata, self._reserved_key)
        if reserved is None:
            # Compatibility for contexts created by older before hooks.
            legacy_estimate = usage_count(ctx.metadata, self._estimate_key)
            if legacy_estimate is not None:
                reserved = self._bucket.reservation_size(legacy_estimate)
        if reserved is not None:
            self._bucket.adjust(actual_total, reserved)

        return response
