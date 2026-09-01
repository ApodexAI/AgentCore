from __future__ import annotations

from typing import Any

import pytest

from agent_core.llm import LLMResponse
from agent_core.loop_types import LoopConfig, LoopPolicy
from agent_core.runtime.loop.agent_loop import AgentLoopHooks, run_agent_loop
from agent_core.runtime.loop.model_profile import ModelProfile
from agent_core.runtime.loop.tool_exec import ToolExecutionHooks


class SequenceLLM:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = responses
        self.calls: list[list[dict[str, Any]]] = []

    async def chat(self, messages, **_kwargs) -> LLMResponse:
        self.calls.append(messages)
        return self.responses.pop(0)

    def stream(self, messages, **_kwargs):
        raise AssertionError("streaming was not requested")


class EchoTool:
    name = "echo"

    async def ainvoke(self, args: dict[str, Any]) -> Any:
        return args["value"]

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "echo a value",
                "parameters": {"type": "object"},
            },
        }


def _config() -> LoopConfig:
    return LoopConfig(
        max_turns=3,
        loop_policy=LoopPolicy(no_tool_behavior="stop"),
        max_llm_retries=1,
    )


@pytest.mark.asyncio
async def test_agent_loop_executes_tool_then_returns_final_answer() -> None:
    llm = SequenceLLM(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {
                            "name": "echo",
                            "arguments": '{"value":"hello"}',
                        },
                    }
                ],
            ),
            LLMResponse(content="finished"),
        ]
    )

    result = await run_agent_loop(
        system_prompt="system",
        user_message="start",
        llm=llm,
        tools=[EchoTool()],
        config=_config(),
        model_profile=ModelProfile(model_id="test", provider="test"),
    )

    assert result.final_content == "finished"
    assert result.tool_calls_count == 1
    assert any(message.get("role") == "tool" and message.get("content") == "hello"
               for message in result.messages)


@pytest.mark.asyncio
async def test_runtime_hooks_wrap_scope_llm_and_tool_boundaries() -> None:
    events: list[str] = []

    def enter_scope(cfg, phase_id, metadata):
        events.append(f"enter:{cfg.max_turns}:{phase_id}:{metadata['agent_id']}")
        return type("Scope", (), {"metadata": metadata})(), "token"

    hooks = AgentLoopHooks(
        sticky_session_enabled=lambda: events.append("sticky") or True,
        wall_deadline_remaining=lambda: events.append("deadline") or None,
        chain_fallback_active=lambda: False,
        enter_scope=enter_scope,
        exit_scope=lambda token: events.append(f"exit:{token}"),
        tool_execution=ToolExecutionHooks(
            on_call=lambda name: events.append(f"tool:{name}")
        ),
    )
    llm = SequenceLLM([LLMResponse(content="done")])

    result = await run_agent_loop(
        system_prompt="system",
        user_message="start",
        llm=llm,
        tools=[],
        config=LoopConfig(
            max_turns=1,
            task_id="task",
            role_id="role",
            loop_policy=LoopPolicy(no_tool_behavior="stop"),
            max_llm_retries=1,
        ),
        runtime_hooks=hooks,
    )

    assert result.final_content == "done"
    assert events[0] == "sticky"
    assert events[1].startswith("enter:1:")
    assert "deadline" in events
    assert events[-1] == "exit:token"


@pytest.mark.asyncio
async def test_host_can_override_session_binding() -> None:
    bound: list[str] = []
    llm = SequenceLLM([LLMResponse(content="done")])

    result = await run_agent_loop(
        system_prompt="system",
        user_message="start",
        llm=llm,
        tools=[],
        config=LoopConfig(
            max_turns=1,
            task_id="runtime-task",
            llm_session_id="gateway-session",
            loop_policy=LoopPolicy(no_tool_behavior="stop"),
            max_llm_retries=1,
        ),
        runtime_hooks=AgentLoopHooks(
            bind_session=lambda client, session_id: (
                bound.append(session_id) or client
            )
        ),
    )

    assert result.final_content == "done"
    assert bound == ["gateway-session"]
