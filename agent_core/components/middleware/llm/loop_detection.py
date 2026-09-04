from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict, deque

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

    def __init__(
        self,
        pattern_window: int = 10,
        trigger_count: int = 3,
        max_scopes: int = 1024,
    ) -> None:
        if pattern_window <= 0:
            raise ValueError("pattern_window must be positive")
        if trigger_count <= 0:
            raise ValueError("trigger_count must be positive")
        if max_scopes <= 0:
            raise ValueError("max_scopes must be positive")
        self._window = pattern_window
        self._trigger = trigger_count
        self._max_scopes = max_scopes
        self._histories: OrderedDict[tuple[str, str, str], deque[str]] = OrderedDict()
        self._pending_hints: set[tuple[str, str, str]] = set()

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
            f"{hashlib.md5(str((tc.get('function') or {}).get('arguments', '')).encode(), usedforsecurity=False).hexdigest()[:8]}"
            for tc in tool_calls
        )
        return sig

    @staticmethod
    def _scope_key(ctx: LLMCallContext) -> tuple[str, str, str] | None:
        """Return a stable per-conversation key, or None when none exists."""
        task_or_session = ctx.task_id or str(ctx.metadata.get("session_id") or "")
        if not task_or_session:
            # Sharing anonymous history is worse than disabling the heuristic:
            # it would let unrelated callers contaminate one another.
            return None
        return task_or_session, ctx.role_id, ctx.phase_id

    def _history_for(self, key: tuple[str, str, str]) -> deque[str]:
        history = self._histories.get(key)
        if history is None:
            history = deque[str](maxlen=self._window)
            self._histories[key] = history
            if len(self._histories) > self._max_scopes:
                evicted, _ = self._histories.popitem(last=False)
                self._pending_hints.discard(evicted)
        else:
            self._histories.move_to_end(key)
        return history

    async def after_llm(
        self, ctx: LLMCallContext, response: LLMResponse
    ) -> LLMResponse:
        key = self._scope_key(ctx)
        if key is None:
            return response
        sig = self._hash_tool_calls(response)
        if sig:
            history = self._history_for(key)
            history.append(sig)
            # Check for consecutive repetition
            if len(history) >= self._trigger:
                recent = list(history)[-self._trigger:]
                if len(set(recent)) == 1:
                    self._pending_hints.add(key)
                    logger.warning(
                        "LoopDetectionMiddleware: detected %d consecutive identical tool calls: %s",
                        self._trigger, sig,
                    )
        return response

    async def before_llm(
        self, ctx: LLMCallContext, messages: list[Message]
    ) -> list[Message]:
        key = self._scope_key(ctx)
        if key is not None and key in self._pending_hints:
            self._pending_hints.discard(key)
            history = self._histories.get(key)
            if history is not None:
                history.clear()
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

    def cleanup_task(self, task_id: str) -> None:
        """Release all loop-detection state retained for ``task_id``."""
        keys = [key for key in self._histories if key[0] == task_id]
        for key in keys:
            self._histories.pop(key, None)
            self._pending_hints.discard(key)
