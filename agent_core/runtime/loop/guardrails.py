"""Portable middleware for duplicate-call and repeated-loop guardrails."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import deque
from collections.abc import Mapping
from typing import Any, cast

from agent_core.protocols import ExecutionMiddleware, ToolCallContext

logger = logging.getLogger(__name__)

DEFAULT_DUPLICATE_THRESHOLDS: dict[str, int] = {
    "web_search": 6,
    "web_fetch": 5,
    "bash": 6,
    "file_editor_str_replace": 5,
    "file_editor_view": 5,
    "file_editor_create": 5,
    "read_file": 5,
    "read_text": 5,
    "write_file": 5,
    "grep_search": 5,
    "glob_search": 5,
    "view_image": 4,
    "delegate_subtask": 3,
    "collect_results": 3,
    "abort_task": 3,
}


def _args_fingerprint(tool_name: str, args: Mapping[str, Any]) -> str:
    # ``default=str`` keeps a non-JSON argument (set, dataclass, Path) from
    # turning a guardrail check into an unhandled TypeError mid tool call.
    raw = json.dumps({"t": tool_name, "a": args}, sort_keys=True, default=str)
    # Fingerprinting only; ``usedforsecurity=False`` keeps this importable on
    # FIPS-enabled builds where plain md5 is unavailable.
    return hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()[:12]


class GuardrailsMiddleware(ExecutionMiddleware):
    """Block pathological repetition while allowing hosts to tune tool policy.

    This middleware only *marks* a call via :meth:`ToolCallContext.block`.
    The host dispatcher must honour ``ctx.is_blocked`` after the middleware
    chain and feed ``ctx.block_reason`` back to the model instead of executing
    the tool; otherwise the guardrail is inert.
    """

    def __init__(
        self,
        search_warn_threshold: int = 50,
        max_loop_hints: int = 3,
        *,
        duplicate_thresholds: Mapping[str, int] | None = None,
        default_duplicate_threshold: int = 5,
        search_tool_name: str = "web_search",
        ignored_tools: frozenset[str] = frozenset({"tool_search"}),
    ) -> None:
        self._search_warn_threshold = search_warn_threshold
        self._max_loop_hints = max_loop_hints
        self._duplicate_thresholds = dict(
            duplicate_thresholds or DEFAULT_DUPLICATE_THRESHOLDS,
        )
        self._default_duplicate_threshold = default_duplicate_threshold
        self._search_tool_name = search_tool_name
        self._ignored_tools = ignored_tools
        self._recent_calls: dict[str, deque[str]] = {}
        self._search_counts: dict[str, int] = {}
        self._search_warned: set[str] = set()
        self._loop_hint_counts: dict[str, int] = {}
        self._stats = {
            "duplicate_blocks": 0,
            "search_warnings": 0,
            "loop_escalation_blocks": 0,
            "total_checks": 0,
        }

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def notify_loop_hint(self, task_id: str) -> None:
        count = self._loop_hint_counts.get(task_id, 0) + 1
        self._loop_hint_counts[task_id] = count
        logger.info(
            "Guardrails loop hint #%d for task %s (hard block at %d)",
            count,
            task_id,
            self._max_loop_hints,
        )

    def reset_loop_hints(self, task_id: str) -> None:
        self._loop_hint_counts.pop(task_id, None)

    async def before_tool_call(self, ctx: ToolCallContext) -> ToolCallContext:
        self._stats["total_checks"] += 1
        task_id = ctx.task_id
        tool_name = ctx.tool_name
        if tool_name in self._ignored_tools:
            return ctx

        fingerprint = _args_fingerprint(tool_name, ctx.tool_args)
        recent = self._recent_calls.setdefault(task_id, deque(maxlen=20))
        consecutive = 0
        for previous in reversed(recent):
            if previous != fingerprint:
                break
            consecutive += 1
        threshold = self._duplicate_thresholds.get(
            tool_name,
            self._default_duplicate_threshold,
        )
        if consecutive >= threshold:
            self._stats["duplicate_blocks"] += 1
            ctx.block(
                f"Blocked: {tool_name} called {consecutive + 1} times "
                "consecutively with identical arguments. Try different "
                "parameters or a different approach.",
            )
            return ctx
        recent.append(fingerprint)

        if tool_name == self._search_tool_name:
            count = self._search_counts.get(task_id, 0) + 1
            self._search_counts[task_id] = count
            if count >= self._search_warn_threshold and task_id not in self._search_warned:
                self._search_warned.add(task_id)
                self._stats["search_warnings"] += 1
                logger.warning(
                    "Search count %d reached warning threshold %d for task %s",
                    count,
                    self._search_warn_threshold,
                    task_id,
                )

        hint_count = self._loop_hint_counts.get(task_id, 0)
        if hint_count >= self._max_loop_hints and fingerprint in list(recent)[:-1]:
            self._stats["loop_escalation_blocks"] += 1
            ctx.block(
                "Blocked: repeated tool call pattern detected after "
                f"{hint_count} loop warnings. Try a completely different "
                "approach or conclude with available evidence.",
            )
        return ctx

    def cleanup_task(self, task_id: str) -> None:
        self._recent_calls.pop(task_id, None)
        self._search_counts.pop(task_id, None)
        self._search_warned.discard(task_id)
        self._loop_hint_counts.pop(task_id, None)


DEFAULT_MAX_TOKENS = 500_000


def _as_int(value: object, default: int) -> int:
    """Coerce untyped product config, falling back instead of raising."""
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip().replace("_", "")))
        except ValueError:
            return default
    return default


def check_budget_exhausted(
    working_state: Mapping[str, Any],
    token_usage: Mapping[str, int] | None,
) -> str | None:
    """Return a finalization hint when the allocated token budget is spent."""
    empty: Mapping[str, Any] = {}
    plan_value: object = working_state.get("execution_plan")
    plan: Mapping[str, Any] = (
        cast("Mapping[str, Any]", plan_value)
        if isinstance(plan_value, Mapping)
        else empty
    )
    budget_value: object = plan.get("budget")
    budget: Mapping[str, Any] = (
        cast("Mapping[str, Any]", budget_value)
        if isinstance(budget_value, Mapping)
        else empty
    )
    allocated_value: object = budget.get("allocated")
    allocated: Mapping[str, Any] = (
        cast("Mapping[str, Any]", allocated_value)
        if isinstance(allocated_value, Mapping)
        else budget
    )
    if token_usage:
        maximum = _as_int(allocated.get("max_tokens"), DEFAULT_MAX_TOKENS)
        # Providers and ``UsageMetadata`` emit ``total_tokens``; ``total`` is
        # accepted for host mappings that use the shorter name.
        used = _as_int(
            token_usage.get("total_tokens", token_usage.get("total")),
            0,
        )
        if used >= maximum:
            return (
                f"Token budget exhausted ({used:,}/{maximum:,}). "
                "Provide final analysis with available evidence."
            )
    return None


__all__ = [
    "DEFAULT_DUPLICATE_THRESHOLDS",
    "DEFAULT_MAX_TOKENS",
    "GuardrailsMiddleware",
    "check_budget_exhausted",
]
