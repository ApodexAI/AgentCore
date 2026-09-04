"""Structural contracts shared by portable runtime components."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agent_core.llm import LLMClient


@runtime_checkable
class EventSink(Protocol):
    """Append-only event destination used by portable runtime components."""

    async def append(
        self,
        task_id: Any = "",
        event_type: Any = None,
        payload: dict[str, Any] | None = None,
        agent_role: str = "system",
    ) -> Any: ...

    def replay(self, task_id: str) -> AsyncIterator[Any]: ...


@runtime_checkable
class EventReader(Protocol):
    """Cursored event reads required by durable inter-agent messaging."""

    async def get_events(
        self,
        task_id: Any,
        event_type: Any = None,
        after_id: int = 0,
        limit: int | None = None,
    ) -> list[Any]: ...

    async def get_events_for_agent(
        self,
        to_agent: str,
        after_id: int = 0,
        limit: int = 50,
        *,
        task_id: str | Any | None = None,
        message_type: str | None = None,
    ) -> list[Any]: ...


@runtime_checkable
class TraceSink(Protocol):
    async def log_llm_call(
        self,
        task_id: str,
        agent_role_id: str,
        action: str,
        input_preview: str,
        output_preview: str,
        duration_ms: int = 0,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any: ...

    async def log_tool_call(
        self,
        task_id: str,
        agent_role_id: str,
        tool_name: str,
        input_data: str,
        output_preview: str,
        duration_ms: int = 0,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any: ...

    async def log_api_error(
        self,
        task_id: str,
        agent_role_id: str,
        error: str,
        **kwargs: Any,
    ) -> Any: ...


@dataclass
class PhaseContext:
    task_id: str
    phase_id: str
    role_id: str = ""
    display_label: str = ""
    state: dict[str, Any] = field(default_factory=dict[str, Any])
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])
    start_time: float = 0.0

    def __post_init__(self) -> None:
        if self.start_time == 0.0:
            self.start_time = time.time()


# Reserved ``ToolCallContext.metadata`` keys. A ``before_tool_call``
# middleware sets ``BLOCKED_KEY`` to veto a call; the host dispatcher MUST
# check it after running the middleware chain and, when set, skip the tool and
# return ``BLOCK_REASON_KEY`` to the model as the tool result. AgentCore does
# not dispatch tools itself, so an unchecked flag means silent non-enforcement.
BLOCKED_KEY = "blocked"
BLOCK_REASON_KEY = "block_reason"


@dataclass
class ToolCallContext:
    task_id: str
    phase_id: str
    role_id: str = ""
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict[str, Any])
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])

    def block(self, reason: str) -> None:
        """Veto this tool call. See ``BLOCKED_KEY`` for the host contract."""
        self.metadata[BLOCKED_KEY] = True
        self.metadata[BLOCK_REASON_KEY] = reason

    @property
    def is_blocked(self) -> bool:
        return bool(self.metadata.get(BLOCKED_KEY))

    @property
    def block_reason(self) -> str:
        return str(self.metadata.get(BLOCK_REASON_KEY) or "")


class ExecutionMiddleware:
    async def before_phase(self, ctx: PhaseContext) -> PhaseContext:
        return ctx

    async def after_phase(
        self, ctx: PhaseContext, result: dict[str, Any]
    ) -> dict[str, Any]:
        return result

    async def before_tool_call(self, ctx: ToolCallContext) -> ToolCallContext:
        return ctx

    async def after_tool_call(self, ctx: ToolCallContext, result: str) -> str:
        return result

    async def on_error(
        self, ctx: PhaseContext, error: Exception
    ) -> Exception | None:
        return error


@runtime_checkable
class PhaseMiddlewareChain(Protocol):
    @property
    def middlewares(self) -> list[ExecutionMiddleware]: ...

    async def run_before_phase(self, ctx: PhaseContext) -> PhaseContext: ...

    async def run_after_phase(
        self, ctx: PhaseContext, result: dict[str, Any]
    ) -> dict[str, Any]: ...

    async def run_on_error(
        self, ctx: PhaseContext, error: Exception
    ) -> Exception | None: ...


@runtime_checkable
class LLMWrapper(Protocol):
    def wrap_llm(self, llm: LLMClient, *, role_id: str) -> LLMClient: ...


@runtime_checkable
class SubAgentProfileRegistry(Protocol):
    def register_sub_agent_profiles(
        self, node_id: str, profiles: Mapping[str, Any]
    ) -> None: ...


@runtime_checkable
class LLMResourceProvider(Protocol):
    """Small resource-manager surface required by generic DAG nodes."""

    def get_llm(self, role_id: str) -> LLMClient: ...


@runtime_checkable
class Skill(Protocol):
    skill_id: str
    name: str
    description: str
    version: str
    tags: list[str]
    allowed_tools: list[str]
    content: str
    root_dir: str
    enabled: bool


@runtime_checkable
class SkillLoader(Protocol):
    def list_skills(self) -> Sequence[Skill]: ...

    def get_skill(self, skill_id: str) -> Skill | None: ...

    def get_enabled_skills(self) -> Sequence[Skill]: ...

    def toggle_skill(self, skill_id: str, enabled: bool) -> bool: ...

    def reload(self) -> None: ...


@runtime_checkable
class CostSink(Protocol):
    """Per-task cost accounting in USD.

    Deliberately synchronous: ``record`` is called from middleware on the LLM
    hot path, and an await there would put an event-loop hop between a
    provider response and the accounting that must not be able to lose it.
    Returns the incremental cost so a caller can log or emit it without a
    second lookup.
    """

    def record(
        self,
        task_id: str,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float: ...


@runtime_checkable
class CostPersister(Protocol):
    """Durable sink for a task's final cost summary.

    Separate from :class:`CostSink` because the two have different lifetimes
    and different failure tolerances: ``CostSink.record`` runs per LLM call and
    may live purely in memory, while ``persist`` runs once at completion and is
    the only path that reaches a database. Keeping it behind a Protocol is what
    lets the token-accounting middleware live here at all -- the schema, the
    column names and the session handling are all host concerns.

    ``summary`` is whatever the host's ``CostSink`` returns from its own
    ``get_summary``; AgentCore forwards it without inspecting it beyond passing
    it through, so a host is free to evolve that shape without a core release.
    """

    async def persist(
        self,
        task_id: str,
        summary: Mapping[str, Any],
        model: str,
    ) -> None: ...


__all__ = [
    "BLOCKED_KEY",
    "BLOCK_REASON_KEY",
    "CostPersister",
    "CostSink",
    "EventReader",
    "EventSink",
    "ExecutionMiddleware",
    "LLMResourceProvider",
    "LLMWrapper",
    "PhaseContext",
    "PhaseMiddlewareChain",
    "Skill",
    "SkillLoader",
    "SubAgentProfileRegistry",
    "ToolCallContext",
    "TraceSink",
]
