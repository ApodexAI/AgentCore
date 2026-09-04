from __future__ import annotations

import logging
import time
from typing import Any, cast

from agent_core.components.middleware.llm.base import (
    LLMCallContext,
    LLMMiddleware,
)
from agent_core.llm import LLMResponse
from agent_core.messages import Message, text_of

logger = logging.getLogger(__name__)


class LLMTracingMiddleware(LLMMiddleware):
    """Records duration and message counts for every LLM call."""

    def __init__(self, trace_logger: Any = None) -> None:
        self._trace = trace_logger
        self._timers: dict[int, float] = {}

    @property
    def name(self) -> str:
        return "llm_tracing"

    async def before_llm(
        self, ctx: LLMCallContext, messages: list[Message]
    ) -> list[Message]:
        self._timers[id(ctx)] = time.time()
        return messages

    async def after_llm(
        self, ctx: LLMCallContext, response: LLMResponse
    ) -> LLMResponse:
        start = self._timers.pop(id(ctx), time.time())
        # Use duration from metadata (stream) if available, otherwise calculate
        duration_ms = ctx.metadata.get("duration_ms")
        if duration_ms is None:
            duration_ms = int((time.time() - start) * 1000)

        if self._trace:
            try:
                # Native LLMResponse carries a flat ``usage`` dict; the
                # response_metadata is a thin ``{"id": ...}`` map.
                raw_rm: object = getattr(response, "response_metadata", None)
                rm: dict[str, Any] = (
                    cast("dict[str, Any]", raw_rm)
                    if isinstance(raw_rm, dict)
                    else {}
                )
                raw_usage: object = getattr(response, "usage", None)
                usage: Any = raw_usage or rm.get("token_usage") or rm.get("usage") or {}

                metadata = {
                    "role_id": ctx.role_id,
                    "call_index": ctx.call_index,
                    "duration_ms": duration_ms,
                    "usage": usage,
                }
                # A fallback chain stamps these on the response when the
                # call fell through to a secondary model. Surface them in
                # the trace metadata so observability can flag failover
                # runs without a separate IPC channel — master design §5.9.
                if "fallback_used" in rm:
                    metadata["fallback_used"] = rm["fallback_used"]
                if "model_actually_used" in rm:
                    metadata["model_actually_used"] = rm["model_actually_used"]
                if ctx.metadata.get("correlation_id"):
                    metadata["correlation_id"] = ctx.metadata["correlation_id"]

                output_text = text_of(response.content)
                await self._trace.log_llm_call(
                    task_id=ctx.task_id or "unknown",
                    agent_role_id=ctx.role_id,
                    action=f"llm_call:{ctx.call_index}",
                    input_preview="[LLM Request]",
                    output_preview=output_text[:4000] if output_text else "",
                    duration_ms=duration_ms,
                    session_id=ctx.metadata.get("session_id"),
                    prompt_id=ctx.metadata.get("prompt_id"),
                    step_id=ctx.metadata.get("step_id"),
                    metadata=metadata,
                )
            except Exception:
                logger.debug("LLMTracingMiddleware.after_llm logging failed", exc_info=True)
        return response
