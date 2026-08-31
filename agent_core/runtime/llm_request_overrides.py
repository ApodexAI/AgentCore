"""Task-local overrides for one physical LLM request.

The runtime occasionally needs to change generation behaviour for one retry
without mutating a cached/shared client.  ``ContextVar`` keeps that override
isolated across concurrent tasks and automatically restores the client's
normal profile on exit.

Provider adapters opt in to the semantic override they understand.  Today the
OpenAI-compatible adapter maps it onto SGLang/Qwen
``chat_template_kwargs`` and an explicitly configured ``reasoning_effort``.
Unsupported adapters simply keep their normal request shape; the runtime's
retry prompt and output cap remain the portable fallback.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class ThinkingRetryOverride:
    """Semantic thinking controls for a single retry attempt."""

    mode: str = "reduced"
    thinking_budget: int | None = None
    reasoning_effort: str | None = None

    @property
    def disabled(self) -> bool:
        return self.mode == "disabled"


_THINKING_RETRY_OVERRIDE: ContextVar[ThinkingRetryOverride | None] = ContextVar(
    "agent_core_thinking_retry_override",
    default=None,
)


def current_thinking_retry_override() -> ThinkingRetryOverride | None:
    """Return the override active for the current async task, if any."""

    return _THINKING_RETRY_OVERRIDE.get()


@contextmanager
def thinking_retry_override(
    override: ThinkingRetryOverride | None,
) -> Generator[None, None, None]:
    """Apply ``override`` only inside this context and async task."""

    if override is None:
        yield
        return
    token = _THINKING_RETRY_OVERRIDE.set(override)
    try:
        yield
    finally:
        _THINKING_RETRY_OVERRIDE.reset(token)


__all__ = [
    "ThinkingRetryOverride",
    "current_thinking_retry_override",
    "thinking_retry_override",
]
