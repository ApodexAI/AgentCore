"""Structural contracts shared by portable runtime components."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agent_core.llm import LLMClient


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


@dataclass
class ToolCallContext:
    task_id: str
    phase_id: str
    role_id: str = ""
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict[str, Any])
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])


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


__all__ = [
    "ExecutionMiddleware",
    "LLMResourceProvider",
    "PhaseContext",
    "PhaseMiddlewareChain",
    "Skill",
    "SkillLoader",
    "SubAgentProfileRegistry",
    "ToolCallContext",
    "TraceSink",
]
