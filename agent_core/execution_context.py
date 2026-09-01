"""Task-local execution identity shared across LLM and tool calls."""

from __future__ import annotations

import copy
import logging
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any, cast

from agent_core.types import new_prompt_id, new_session_id, new_step_id

logger = logging.getLogger(__name__)


@dataclass
class ExecutionScope:
    """Runtime execution scope for the current phase."""

    task_id: str = ""
    phase_id: str = ""
    role_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])


_CURRENT_SCOPE: ContextVar[ExecutionScope | None] = ContextVar(
    "agent_core_execution_scope", default=None
)
_CURRENT_TOOL_CALL_ID: ContextVar[str] = ContextVar(
    "agent_core_current_tool_call_id", default=""
)
_CURRENT_TOOL_BUDGET: ContextVar[float | None] = ContextVar(
    "agent_core_current_tool_budget", default=None
)
_CHAIN_FALLBACK_ACTIVE: ContextVar[bool] = ContextVar(
    "agent_core_chain_fallback_active", default=False
)


def normalize_execution_context(value: Any) -> dict[str, Any]:
    """Return a fully independent mutable execution-context mapping.

    The copy is deep: callers mutate scope metadata freely (including nested
    dicts and lists), and persisted checkpoint state it was loaded from must
    not change underneath them.
    """
    if not isinstance(value, dict):
        return {}
    mapping = cast("dict[str, Any]", value)
    try:
        return copy.deepcopy(mapping)
    except Exception:
        # Persisted state is expected to be plain data, but an in-memory state
        # dict may carry something uncopyable. Degrading to a shallow copy is
        # better than failing to build the scope at all.
        logger.warning(
            "Execution context could not be deep-copied; falling back to a "
            "shallow copy, so nested values stay shared with the source state.",
            exc_info=True,
        )
        return dict(mapping)


def build_execution_scope(
    *,
    task_id: str,
    phase_id: str,
    role_id: str,
    state: dict[str, Any] | None = None,
) -> ExecutionScope:
    """Build a scope from task identity and optional persisted state."""
    persisted = state.get("execution_context") if state is not None else None
    metadata = normalize_execution_context(persisted)
    # ``role_id`` is authoritative for this scope: a resumed task may carry an
    # ``agent_id`` written during a different phase, and keeping it would
    # attribute this role's work to the previous agent.
    if role_id or "agent_id" not in metadata:
        metadata["agent_id"] = role_id
    return ExecutionScope(
        task_id=task_id,
        phase_id=phase_id,
        role_id=role_id,
        metadata=metadata,
    )


def set_current_execution_scope(scope: ExecutionScope) -> Token[ExecutionScope | None]:
    return _CURRENT_SCOPE.set(scope)


def get_current_execution_scope() -> ExecutionScope | None:
    return _CURRENT_SCOPE.get()


def reset_current_execution_scope(token: Token[ExecutionScope | None]) -> None:
    _CURRENT_SCOPE.reset(token)


def set_current_tool_call_id(tool_call_id: str) -> Token[str]:
    return _CURRENT_TOOL_CALL_ID.set(tool_call_id)


def get_current_tool_call_id() -> str:
    return _CURRENT_TOOL_CALL_ID.get()


def reset_current_tool_call_id(token: Token[str]) -> None:
    _CURRENT_TOOL_CALL_ID.reset(token)


def set_current_tool_budget(seconds: float | None) -> Token[float | None]:
    """Publish the outer loop budget for the active tool task."""
    return _CURRENT_TOOL_BUDGET.set(seconds)


def get_current_tool_budget() -> float | None:
    return _CURRENT_TOOL_BUDGET.get()


def reset_current_tool_budget(token: Token[float | None]) -> None:
    _CURRENT_TOOL_BUDGET.reset(token)


def chain_fallback_active() -> bool:
    """Whether an outer provider-chain runner can rotate this attempt."""
    return _CHAIN_FALLBACK_ACTIVE.get()


@contextmanager
def chain_fallback_scope() -> Generator[None, None, None]:
    """Mark this context as running beneath an outer provider chain."""
    token = _CHAIN_FALLBACK_ACTIVE.set(True)
    try:
        yield
    finally:
        _CHAIN_FALLBACK_ACTIVE.reset(token)


def ensure_trace_metadata(
    metadata: dict[str, Any],
    *,
    default_step_id: str | None = None,
    refresh_prompt_id: bool = False,
) -> dict[str, Any]:
    """Ensure stable session, step, and prompt identifiers exist."""
    metadata.setdefault("session_id", str(new_session_id()))
    metadata.setdefault("step_id", default_step_id or str(new_step_id()))
    if refresh_prompt_id or not metadata.get("prompt_id"):
        metadata["prompt_id"] = str(new_prompt_id())
    return metadata


__all__ = [
    "ExecutionScope",
    "build_execution_scope",
    "chain_fallback_active",
    "chain_fallback_scope",
    "ensure_trace_metadata",
    "get_current_execution_scope",
    "get_current_tool_budget",
    "get_current_tool_call_id",
    "normalize_execution_context",
    "reset_current_execution_scope",
    "reset_current_tool_budget",
    "reset_current_tool_call_id",
    "set_current_execution_scope",
    "set_current_tool_budget",
    "set_current_tool_call_id",
]
