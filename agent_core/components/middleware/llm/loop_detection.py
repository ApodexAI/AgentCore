from __future__ import annotations

import hashlib
import logging
from collections import deque

from agent_core.components.middleware.llm.base import (
    LLMCallContext,
    LLMMiddleware,
)
from agent_core.llm import LLMResponse
from agent_core.messages import Message, system_msg, text_of

logger = logging.getLogger(__name__)


class LoopDetectionMiddleware(LLMMiddleware):
    """Detects repeated tool call patterns and injects a strategy-switch hint.

    Tracks the last N tool calls (by name + args hash). If the same pattern
    appears `trigger_count` times consecutively, injects a system hint
    asking the LLM to try a different approach.
    """

    def __init__(self, pattern_window: int = 10, trigger_count: int = 3) -> None:
        self._window = pattern_window
        self._trigger = trigger_count
        self._history: deque[str] = deque(maxlen=pattern_window)
        self._injected: bool = False

    @property
    def name(self) -> str:
        return "loop_detection"

    def _hash_tool_calls(self, response: LLMResponse) -> str | None:
        """Create a fingerprint of tool calls in the response.

        Native wire tool_calls are ``{id, type, function: {name,
        arguments}}`` dicts, so the name lives under ``function.name``
        and the args are the JSON-encoded ``function.arguments`` string.
        """
        tool_calls = response.tool_calls
        if not tool_calls:
            return None
        sig = "|".join(
            f"{(tc.get('function') or {}).get('name', '')}:"
            f"{hashlib.md5(str((tc.get('function') or {}).get('arguments', '')).encode()).hexdigest()[:8]}"
            for tc in tool_calls
        )
        return sig

    async def after_llm(
        self, ctx: LLMCallContext, response: LLMResponse
    ) -> LLMResponse:
        sig = self._hash_tool_calls(response)
        if sig:
            self._history.append(sig)
            # Check for consecutive repetition
            if len(self._history) >= self._trigger:
                recent = list(self._history)[-self._trigger:]
                if len(set(recent)) == 1:
                    self._injected = True
                    logger.warning(
                        "LoopDetectionMiddleware: detected %d consecutive identical tool calls: %s",
                        self._trigger, sig,
                    )
        return response

    async def before_llm(
        self, ctx: LLMCallContext, messages: list[Message]
    ) -> list[Message]:
        if self._injected:
            self._injected = False
            self._history.clear()
            hint_text = (
                "\n\n[Loop detected] You have been repeating the same "
                "tool calls. Please try a different approach: use "
                "different search terms, try a different tool, or "
                "synthesize from what you already have."
            )
            # Merge into existing system message (some providers
            # require the system message to be first and only).
            messages = list(messages)
            for i, msg in enumerate(messages):
                if msg.get("role") == "system":
                    messages[i] = system_msg(
                        text_of(msg.get("content")) + hint_text,
                    )
                    return messages
            # No system message found — prepend as a system message
            return [system_msg(hint_text.strip()), *messages]
        return messages
