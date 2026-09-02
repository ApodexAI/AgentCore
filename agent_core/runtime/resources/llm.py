"""LLM-resolution helpers for ResourceManager."""

from __future__ import annotations

from collections.abc import Callable

from agent_core.llm import LLMClient
from agent_core.models.agent_definition import AgentDefinition
from agent_core.protocols import LLMWrapper
from agent_core.runtime.registries import services as service_registry


def resolve_base_llm_for_role(
    *,
    default_llm: LLMClient,
    role_id: str | None,
    cache: dict[str, LLMClient],
    role_llm_factory: Callable[[AgentDefinition], LLMClient] | None = None,
) -> LLMClient:
    """Resolve the base LLM for a role without middleware wrapping."""
    if role_id is None:
        return default_llm

    if role_id in cache:
        return cache[role_id]

    from agent_core.runtime.registries.agents import AgentRegistry

    try:
        agent_reg = service_registry.get(AgentRegistry)
        defn = agent_reg.get(role_id)
        if defn.model and role_llm_factory is not None:
            role_llm = role_llm_factory(defn)
            cache[role_id] = role_llm
            return role_llm
    except (KeyError, RuntimeError, ImportError):
        pass

    return default_llm


def wrap_llm_with_middleware(
    llm: LLMClient,
    *,
    role_id: str,
) -> LLMClient:
    """Wrap the LLM with an optional LLM wrapper service."""
    try:
        wrapper = service_registry.get_optional(LLMWrapper)
        if wrapper is not None:
            return wrapper.wrap_llm(llm, role_id=role_id)
    except (ImportError, RuntimeError):
        pass
    return llm
