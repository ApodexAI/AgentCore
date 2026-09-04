"""TodoMiddleware — re-injects task progress after context compaction.

Issue #25 Phase B: prevents long-running tasks from losing their plan
when SummarizationMiddleware compresses the conversation.

Runs AFTER SummarizationMiddleware in the LLM middleware chain.
Detects when context was compacted and injects a compact task progress
reminder from PlanSnapshot + WorkingMemory.
"""

from __future__ import annotations

import logging

from agent_core.components.middleware.llm.base import LLMCallContext, LLMMiddleware
from agent_core.messages import Message, text_of, user_msg

logger = logging.getLogger(__name__)

# Minimum turn count before injecting (avoid noise on early turns)
_MIN_TURN_FOR_INJECTION = 3

# Injection marker so we don't double-inject
_TODO_MARKER = "[Task Progress]"


class TodoMiddleware(LLMMiddleware):
    """LLM middleware: inject task progress after context compaction.

    Detects compacted context (summary message present) and injects a
    compact task progress block derived from PlanSnapshot/WorkingMemory.

    This ensures the LLM always knows:
    - What sub-questions are still open
    - What has been found so far (evidence count, key findings)
    - What the current budget status is
    - What the recommended next action is
    """

    @property
    def name(self) -> str:
        return "todo"

    async def before_llm(
        self,
        ctx: LLMCallContext,
        messages: list[Message],
    ) -> list[Message]:
        """Inject compact task progress when context is compacted or deep."""
        if not self._should_inject(ctx, messages):
            return messages

        progress_block = self._build_progress_block(ctx)
        if not progress_block:
            return messages

        # Inject as a user message before the last user message
        # so the LLM sees it as recent context
        todo_msg = user_msg(progress_block)

        # Find insertion point: before the last non-system message
        insert_idx = len(messages)
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") != "system":
                insert_idx = i
                break

        result = list(messages)
        result.insert(insert_idx, todo_msg)
        logger.debug(
            "TodoMiddleware: injected progress block (%d chars) "
            "at position %d",
            len(progress_block), insert_idx,
        )
        return result

    def _should_inject(
        self,
        ctx: LLMCallContext,
        messages: list[Message],
    ) -> bool:
        """Decide whether to inject task progress."""
        # Don't inject if already present
        for m in messages:
            content = str(text_of(m.get("content")))
            if _TODO_MARKER in content:
                return False

        # Always inject if context was compacted (summary present)
        for m in messages:
            content = str(text_of(m.get("content")))
            if "[Previous conversation summary" in content:
                return True

        # Also inject on high turn counts even without compaction
        turn = ctx.metadata.get("turn", 0)
        return turn >= _MIN_TURN_FOR_INJECTION

    def _build_progress_block(self, ctx: LLMCallContext) -> str:
        """Build compact task progress from PlanSnapshot + WorkingMemory."""
        parts = [_TODO_MARKER]

        # 1. Working Memory summary (findings + evidence count)
        wm_summary = self._get_wm_summary()
        if wm_summary:
            parts.append(wm_summary)

        # 2. PlanSnapshot: open sub-questions + budget
        snapshot_summary = self._get_snapshot_summary(ctx)
        if snapshot_summary:
            parts.append(snapshot_summary)

        # 3. Evaluation guidance (if available in metadata)
        guidance = ctx.metadata.get("continuation_guidance", "")
        if guidance:
            parts.append(f"Suggested focus: {guidance}")

        # Only return if we have substantive content
        if len(parts) <= 1:
            return ""
        return "\n".join(parts)

    def _get_wm_summary(self) -> str:
        """Extract compact summary from WorkingMemory.

        Domain subclasses can override ``one_line_summary`` to surface their
        own progress vocabulary without coupling this middleware to it.
        """
        try:
            from agent_core.components.memory import (
                current_working_memory,
            )

            wm = current_working_memory.get(None)
            if wm is None:
                return ""

            lines: list[str] = [wm.one_line_summary()]
            if wm.key_findings:
                lines.append("Key findings:")
                for f in wm.key_findings[-5:]:
                    lines.append(f"  - {f[:100]}")
            return "\n".join(lines)
        except Exception:
            return ""

    def _get_snapshot_summary(self, ctx: LLMCallContext) -> str:
        """Extract compact summary from execution context."""
        try:
            from agent_core.execution_context import (
                get_current_execution_scope,
            )

            scope = get_current_execution_scope()
            if scope is None:
                return ""

            # Build from scope metadata if available
            meta = scope.metadata or {}
            lines: list[str] = []

            # Turn progress
            turn = ctx.metadata.get("turn", 0)
            max_turns = meta.get("max_turns", 0)
            if max_turns:
                lines.append(f"Turn: {turn}/{max_turns}")

            # Budget
            depth = meta.get("current_depth", 0)
            if depth > 0:
                lines.append(f"Sub-agent depth: {depth}")

            return "\n".join(lines)
        except Exception:
            return ""
