from __future__ import annotations

import asyncio

import pytest

import agent_core.runtime.loop._call as call_module
from agent_core.errors import (
    AgentCoreError,
    LLMCallExhausted,
    LLMStreamStalled,
)
from agent_core.llm import LLMResponse
from agent_core.messages import user_msg
from agent_core.runtime.llm_request_overrides import (
    ThinkingRetryOverride,
    current_thinking_retry_override,
    thinking_retry_override,
)
from agent_core.runtime.loop._bind import bind_session_id
from agent_core.runtime.loop.llm_client import call_llm


class FakeLLM:
    model = "fake"

    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, messages: object, **kwargs: object) -> LLMResponse:
        self.calls += 1
        return LLMResponse(content="ok")


def test_llm_errors_share_the_agent_core_root_and_timeout_contract() -> None:
    stalled = LLMStreamStalled(stall_s=3, chunks_seen=2, elapsed_s=7)

    assert isinstance(stalled, AgentCoreError)
    assert isinstance(stalled, TimeoutError)
    exhausted = LLMCallExhausted(stalled, "exhausted")
    assert exhausted.last_exc is stalled
    assert exhausted.reason == "exhausted"


def test_bind_session_id_uses_explicit_product_kill_switch() -> None:
    llm = FakeLLM()

    assert bind_session_id(llm, "task-1", sticky_session_enabled=lambda: False) is llm
    bound = bind_session_id(llm, "task-1", sticky_session_enabled=lambda: True)
    assert bound.extra_headers == {"x-upstream-session-id": "task-1"}


@pytest.mark.asyncio
async def test_wall_deadline_hook_refuses_before_provider_call(monkeypatch) -> None:
    monkeypatch.setattr(call_module, "_WALL_DEADLINE_FLOOR_S", 20.0)
    llm = FakeLLM()

    with pytest.raises(LLMCallExhausted) as exc_info:
        await call_llm(
            llm,
            [user_msg("hello")],
            timeout=120,
            max_retries=2,
            turn=1,
            wall_deadline_remaining=lambda: 5.0,
        )

    assert exc_info.value.reason == "wall_deadline"
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_thinking_override_is_task_local_and_restored() -> None:
    async def observe(mode: str) -> str:
        with thinking_retry_override(ThinkingRetryOverride(mode=mode)):
            await asyncio.sleep(0)
            current = current_thinking_retry_override()
            assert current is not None
            return current.mode

    assert await asyncio.gather(observe("reduced"), observe("disabled")) == [
        "reduced",
        "disabled",
    ]
    assert current_thinking_retry_override() is None
