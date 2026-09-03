from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from agent_core.llm import LLMResponse
from agent_core.loop_types import (
    Intervention,
    LoopConfig,
    LoopPolicy,
    ToolCallIntervention,
)
from agent_core.runtime.loop.agent_loop import AgentLoopHooks, run_agent_loop
from agent_core.runtime.loop.model_profile import HistoryPolicy, ModelProfile
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
async def test_active_scope_receives_llm_call_identity() -> None:
    from agent_core.execution_context import (
        ExecutionScope,
        get_current_execution_scope,
        reset_current_execution_scope,
        set_current_execution_scope,
    )

    seen: dict[str, Any] = {}

    class ScopeInspectingLLM(SequenceLLM):
        async def chat(self, messages, **kwargs) -> LLMResponse:
            scope = get_current_execution_scope()
            assert scope is not None
            seen.update(scope.metadata)
            return await super().chat(messages, **kwargs)

    def enter_scope(cfg, phase_id, metadata):
        scope = ExecutionScope(
            task_id=cfg.task_id,
            phase_id=phase_id,
            role_id=cfg.role_id,
            metadata=metadata,
        )
        return scope, set_current_execution_scope(scope)

    await run_agent_loop(
        system_prompt="system",
        user_message="start",
        llm=ScopeInspectingLLM([LLMResponse(content="done")]),
        tools=[],
        config=LoopConfig(
            max_turns=1,
            task_id="task",
            role_id="role",
            loop_policy=LoopPolicy(no_tool_behavior="stop"),
            max_llm_retries=1,
        ),
        runtime_hooks=AgentLoopHooks(
            enter_scope=enter_scope,
            exit_scope=reset_current_execution_scope,
        ),
    )

    assert str(seen["_llm_call_id"]).startswith("llm_")
    assert str(seen["_llm_attempt_id"]).endswith("_attempt_01")
    assert seen["_llm_attempt_index"] == 1


@pytest.mark.asyncio
async def test_observer_receives_pre_provider_llm_input() -> None:
    captured: list[Any] = []

    class InputObserver:
        async def on_llm_input(self, ctx) -> None:
            captured.append(ctx)

    await run_agent_loop(
        system_prompt="system",
        user_message="start",
        llm=SequenceLLM([LLMResponse(content="done")]),
        tools=[],
        config=LoopConfig(
            max_turns=1,
            task_id="task",
            role_id="role",
            loop_policy=LoopPolicy(no_tool_behavior="stop"),
            max_llm_retries=1,
        ),
        observers=[InputObserver()],
    )

    assert len(captured) == 1
    assert captured[0].messages[0]["content"] == "system"
    assert captured[0].messages[1]["content"] == "start"
    assert str(captured[0].metadata["_llm_call_id"]).startswith("llm_")


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


def _orphan_tool_call_ids(messages: list[dict[str, Any]]) -> set[str]:
    """Ids an assistant message announces that no tool message answers.

    Providers reject these with a hard 400, so any request the loop builds must
    have an empty orphan set.
    """
    announced: set[str] = set()
    answered: set[str] = set()
    for message in messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                if call.get("id"):
                    announced.add(str(call["id"]))
        elif message.get("role") == "tool" and message.get("tool_call_id"):
            answered.add(str(message["tool_call_id"]))
    return announced - answered


def _misplaced_tool_messages(messages: list[dict[str, Any]]) -> list[int]:
    """Indices of tool messages not reachable from the assistant call above.

    A tool message separated from its ``tool_calls`` by a user message is as
    invalid as a missing one, so injection must not split the pair.
    """
    bad: list[int] = []
    for index, message in enumerate(messages):
        if message.get("role") != "tool":
            continue
        cursor = index - 1
        while cursor >= 0 and messages[cursor].get("role") == "tool":
            cursor -= 1
        if cursor < 0 or messages[cursor].get("role") != "assistant":
            bad.append(index)
    return bad


def _native_call(call_id: str, value: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "echo",
            "arguments": json.dumps({"value": value}),
        },
    }


@pytest.mark.asyncio
async def test_skip_tool_execution_still_answers_the_announced_calls() -> None:
    """``skip_tool_execution`` must not strand the assistant's tool_call_id."""

    class SkipExecution:
        # Only critical observers have their Intervention collected;
        # non-critical hooks run detached and their return value is dropped.
        critical = True

        async def on_llm_response(self, ctx: Any) -> Intervention:
            if ctx.tool_calls:
                return Intervention(skip_tool_execution=True)
            return Intervention()

    llm = SequenceLLM(
        [
            LLMResponse(content="", tool_calls=[_native_call("tc1", "hello")]),
            LLMResponse(content="finished"),
        ]
    )

    result = await run_agent_loop(
        system_prompt="system",
        user_message="start",
        llm=llm,
        tools=[EchoTool()],
        config=_config(),
        observers=[SkipExecution()],
        model_profile=ModelProfile(model_id="test", provider="test"),
    )

    assert result.final_content == "finished"
    assert len(llm.calls) == 2
    assert _orphan_tool_call_ids(llm.calls[1]) == set()
    assert _misplaced_tool_messages(llm.calls[1]) == []
    assert _orphan_tool_call_ids(result.messages) == set()


@pytest.mark.asyncio
async def test_continue_to_next_turn_without_pop_answers_the_calls() -> None:
    """A replayed turn keeps the assistant message, so it must answer it too."""

    class ReplayOnce:
        critical = True

        def __init__(self) -> None:
            self.fired = False

        async def on_llm_response(self, ctx: Any) -> Intervention:
            if ctx.tool_calls and not self.fired:
                self.fired = True
                return Intervention(
                    continue_to_next_turn=True,
                    inject_messages=["reconsider that call"],
                )
            return Intervention()

    llm = SequenceLLM(
        [
            LLMResponse(content="", tool_calls=[_native_call("tc1", "hello")]),
            LLMResponse(content="finished"),
        ]
    )

    result = await run_agent_loop(
        system_prompt="system",
        user_message="start",
        llm=llm,
        tools=[EchoTool()],
        config=_config(),
        observers=[ReplayOnce()],
        model_profile=ModelProfile(model_id="test", provider="test"),
    )

    assert result.final_content == "finished"
    assert len(llm.calls) == 2
    assert _orphan_tool_call_ids(llm.calls[1]) == set()
    assert _misplaced_tool_messages(llm.calls[1]) == []


@pytest.mark.asyncio
async def test_skipping_a_text_parsed_call_does_not_reuse_the_synthetic_id() -> None:
    """Text-mode calls carry no provider id, so the engine mints one.

    The skipped and executed halves used to number off different lists, which
    made them collide on the same ``tool_call_id`` once an earlier call was
    short-circuited.
    """

    class SkipFirst:
        def __init__(self) -> None:
            self.seen = 0

        async def on_tool_call(
            self, _ctx: Any, _tool_call: dict[str, Any]
        ) -> ToolCallIntervention:
            self.seen += 1
            if self.seen == 1:
                return ToolCallIntervention(skip_with_result="skipped by policy")
            return ToolCallIntervention()

    text_calls = (
        '<tool_call>{"tool": "echo", "args": {"value": "a"}}</tool_call>'
        '<tool_call>{"tool": "echo", "args": {"value": "b"}}</tool_call>'
    )
    llm = SequenceLLM(
        [LLMResponse(content=text_calls), LLMResponse(content="finished")]
    )

    result = await run_agent_loop(
        system_prompt="system",
        user_message="start",
        llm=llm,
        tools=[EchoTool()],
        config=_config(),
        observers=[SkipFirst()],
        model_profile=ModelProfile(model_id="test", provider="test"),
    )

    assert result.final_content == "finished"
    tool_ids = [
        message.get("tool_call_id")
        for message in result.messages
        if message.get("role") == "tool"
    ]
    assert len(tool_ids) == 2
    assert len(set(tool_ids)) == 2, f"colliding tool_call_ids: {tool_ids}"
    bodies = [
        message.get("content")
        for message in result.messages
        if message.get("role") == "tool"
    ]
    assert "skipped by policy" in bodies
    assert "b" in bodies


@pytest.mark.asyncio
async def test_bind_session_takes_over_and_warns_about_the_sticky_flag(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Two hooks for one concern must not resolve silently."""
    bound: list[str] = []
    sticky_consulted: list[bool] = []
    llm = SequenceLLM([LLMResponse(content="done")])

    with caplog.at_level("WARNING"):
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
                ),
                sticky_session_enabled=lambda: (
                    sticky_consulted.append(True) or True
                ),
            ),
        )

    assert result.final_content == "done"
    assert bound == ["gateway-session"]
    assert sticky_consulted == []
    assert any(
        "bind_session owns session affinity" in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_history_policy_tool_result_cap_is_honoured() -> None:
    """``HistoryPolicy.tool_result_max_chars`` used to be read by nobody."""

    class BigTool:
        name = "echo"

        async def ainvoke(self, args: dict[str, Any]) -> Any:
            return "x" * 500

        def to_openai_schema(self) -> dict[str, Any]:
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": "big",
                    "parameters": {"type": "object"},
                },
            }

    llm = SequenceLLM(
        [
            LLMResponse(content="", tool_calls=[_native_call("tc1", "hello")]),
            LLMResponse(content="finished"),
        ]
    )

    result = await run_agent_loop(
        system_prompt="system",
        user_message="start",
        llm=llm,
        tools=[BigTool()],
        config=_config(),
        history_policy=HistoryPolicy(tool_result_max_chars=100),
        model_profile=ModelProfile(model_id="test", provider="test"),
    )

    assert result.final_content == "finished"
    body = next(
        str(message.get("content") or "")
        for message in result.messages
        if message.get("role") == "tool"
    )
    assert "truncated" in body
    assert len(body) < 400


@pytest.mark.asyncio
async def test_loop_config_cap_overrides_the_policy_default() -> None:
    llm = SequenceLLM(
        [
            LLMResponse(content="", tool_calls=[_native_call("tc1", "hello")]),
            LLMResponse(content="finished"),
        ]
    )

    config = _config()
    config.tool_result_max_chars = 0  # explicit "no cap"

    result = await run_agent_loop(
        system_prompt="system",
        user_message="start",
        llm=llm,
        tools=[EchoTool()],
        config=config,
        history_policy=HistoryPolicy(tool_result_max_chars=2),
        model_profile=ModelProfile(model_id="test", provider="test"),
    )

    body = next(
        str(message.get("content") or "")
        for message in result.messages
        if message.get("role") == "tool"
    )
    assert body == "hello"


@pytest.mark.asyncio
async def test_synthesised_zero_usage_does_not_reset_the_token_estimate() -> None:
    """A gateway that omits usage must not silently disable the overflow guard.

    The zero-filled fallback exists for cost attribution only; letting its
    zeros overwrite the previous turn's real counts made
    ``_handle_context_overflow`` compute its estimate off 0 and never fire.

    Numbers are chosen so the guard's verdict differs *only* because of the
    carried-over counts: floor is 1000 (max_completion_tokens=0, empty summary
    prompt), turn 1 reports 900+50 and appends a tiny tool result, turn 2
    reports no usage and appends a ~1506-token one.

        turn 1            : 950 +    7 + 1000 = 1957  < 3000  (continues)
        turn 2, zeros     :   0 + 1506 + 1000 = 2506  < 3000  (guard misses)
        turn 2, preserved : 950 + 1506 + 1000 = 3456 >= 3000  (guard fires)
    """

    class SizedTool:
        name = "echo"

        async def ainvoke(self, args: dict[str, Any]) -> Any:
            return args["value"]

        def to_openai_schema(self) -> dict[str, Any]:
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": "echo",
                    "parameters": {"type": "object"},
                },
            }

    first = LLMResponse(content="", tool_calls=[_native_call("tc1", "hello")])
    first.usage = {"prompt_tokens": 900, "completion_tokens": 50}
    second = LLMResponse(
        content="", tool_calls=[_native_call("tc2", "x" * 4000)]
    )
    second.usage = None
    llm = SequenceLLM([first, second, LLMResponse(content="finished")])

    config = _config()
    config.context_overflow_guard = True
    config.max_context_length = 3000
    config.max_completion_tokens = 0

    result = await run_agent_loop(
        system_prompt="system",
        user_message="start",
        llm=llm,
        tools=[SizedTool()],
        config=config,
        model_profile=ModelProfile(model_id="test", provider="test"),
    )

    assert result.stopped_by == "context_limit_reached"


@pytest.mark.asyncio
async def test_host_named_fan_in_tool_can_be_interrupted() -> None:
    """A host whose fan-in tool is not named ``collect_reports``.

    Its interrupt waiter used to be ignored no matter that it was supplied.
    """

    class GatherTool:
        name = "gather_subagent_reports"

        async def ainvoke(self, args: dict[str, Any]) -> Any:
            await asyncio.sleep(10)
            return "never"

        def to_openai_schema(self) -> dict[str, Any]:
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": "fan in",
                    "parameters": {"type": "object"},
                },
            }

    class Interrupter:
        async def wait_for_tool_interrupt(
            self, _ctx: Any, _tool_call: dict[str, Any]
        ) -> bool:
            await asyncio.sleep(0)
            return True

    call = {
        "id": "tc1",
        "type": "function",
        "function": {"name": "gather_subagent_reports", "arguments": "{}"},
    }
    llm = SequenceLLM(
        [
            LLMResponse(content="", tool_calls=[call]),
            LLMResponse(content="finished"),
        ]
    )

    result = await run_agent_loop(
        system_prompt="system",
        user_message="start",
        llm=llm,
        tools=[GatherTool()],
        config=_config(),
        observers=[Interrupter()],
        runtime_hooks=AgentLoopHooks(
            tool_execution=ToolExecutionHooks(
                aggregation_tools=frozenset({"gather_subagent_reports"})
            )
        ),
        model_profile=ModelProfile(model_id="test", provider="test"),
    )

    body = next(
        str(message.get("content") or "")
        for message in result.messages
        if message.get("role") == "tool"
    )
    assert body.startswith("[interrupted]")


@pytest.mark.asyncio
async def test_rollback_after_an_injected_message_still_drops_the_turn() -> None:
    """``pop_last_message`` must not be a no-op once a user message is injected.

    ``on_tool_wait_interrupted`` appends a user message *inside* the turn, so at
    end-of-turn rollback the tail is [assistant, tool, user]. Stopping at the
    first non-tool message made the pop a silent no-op, and paired with
    ``continue_to_next_turn`` the same assistant turn replayed until the
    attempt buffer ran out.
    """

    class InterruptThenRollback:
        critical = True

        def __init__(self) -> None:
            self.rolled_back = False

        async def wait_for_tool_interrupt(
            self, _ctx: Any, _tool_call: dict[str, Any]
        ) -> bool:
            await asyncio.sleep(0)
            return not self.rolled_back

        async def on_tool_wait_interrupted(self, _ctx: Any) -> Intervention:
            return Intervention(inject_messages=["a new user message arrived"])

        async def on_turn_end(self, _ctx: Any) -> Intervention:
            if not self.rolled_back:
                self.rolled_back = True
                return Intervention(
                    pop_last_message=True, continue_to_next_turn=True
                )
            return Intervention()

    class FanInTool:
        name = "collect_reports"

        async def ainvoke(self, args: dict[str, Any]) -> Any:
            await asyncio.sleep(10)
            return "never"

        def to_openai_schema(self) -> dict[str, Any]:
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": "fan in",
                    "parameters": {"type": "object"},
                },
            }

    fan_call = {
        "id": "tc1",
        "type": "function",
        "function": {"name": "collect_reports", "arguments": "{}"},
    }
    llm = SequenceLLM(
        [
            LLMResponse(content="", tool_calls=[fan_call]),
            LLMResponse(content="finished"),
        ]
    )

    result = await run_agent_loop(
        system_prompt="system",
        user_message="start",
        llm=llm,
        tools=[FanInTool()],
        config=_config(),
        observers=[InterruptThenRollback()],
        model_profile=ModelProfile(model_id="test", provider="test"),
    )

    assert result.final_content == "finished"
    assert len(llm.calls) == 2
    # The rejected assistant turn and its tool reply are gone...
    assert not any(message.get("tool_calls") for message in llm.calls[1]), (
        llm.calls[1]
    )
    assert not any(
        message.get("role") == "tool" for message in llm.calls[1]
    ), llm.calls[1]
    # ...while the injected notice that replaced it survives.
    assert any(
        "a new user message arrived" in str(message.get("content") or "")
        for message in llm.calls[1]
    )
    assert _orphan_tool_call_ids(llm.calls[1]) == set()
