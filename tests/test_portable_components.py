from __future__ import annotations

import pytest

from agent_core.components.agent_bus.shared_pool import SharedArtifactPool
from agent_core.components.middleware.llm.base import (
    LLMCallContext,
    LLMMiddleware,
    LLMMiddlewareChain,
)
from agent_core.components.middleware.llm.proxy import LLMProxy
from agent_core.llm import LLMResponse
from agent_core.models.agent_definition import AgentDefinition
from agent_core.models.pipeline_spec import PipelineSpec
from agent_core.runtime.registries.agents import AgentRegistry
from agent_core.scheduling.pipeline_registry import PipelineRegistry


class _Prefix(LLMMiddleware):
    @property
    def name(self) -> str:
        return "prefix"

    async def before_llm(self, ctx: LLMCallContext, messages):
        return [*messages, {"role": "user", "content": str(ctx.call_index)}]


@pytest.mark.asyncio
async def test_llm_proxy_applies_shared_middleware() -> None:
    class _LLM:
        model = "test"

        async def chat(self, messages, **kwargs):
            return LLMResponse(content=str(messages[-1]["content"]))

    chain = LLMMiddlewareChain()
    chain.add(_Prefix())
    proxy = LLMProxy(_LLM(), chain)
    assert (await proxy.chat([])).content == "1"
    assert (await proxy.chat([])).content == "2"


@pytest.mark.asyncio
async def test_shared_artifact_pool_deduplicates_ids() -> None:
    pool = SharedArtifactPool()
    await pool.add([{"id": "a", "value": 1}, {"id": "a", "value": 2}, {"value": 3}])
    assert pool.get_all() == [{"id": "a", "value": 1}, {"value": 3}]


def test_agent_and_pipeline_registries_round_trip() -> None:
    agents = AgentRegistry()
    definition = AgentDefinition(role_id="researcher", display_name="Researcher")
    agents.register(definition)
    assert agents.get("researcher") is definition

    pipelines = PipelineRegistry()
    spec = PipelineSpec(pipeline_id="portable", name="Portable", entry_point="start")
    pipelines.register(spec)
    assert pipelines.get("portable") is spec
