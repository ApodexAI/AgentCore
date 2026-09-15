from __future__ import annotations

import json
from typing import Any

import pytest

from agent_core.llm import LLMResponse
from agent_core.loop_types import LoopConfig, LoopPolicy
from agent_core.runtime.loop.agent_loop import (
    AgentLoopHooks,
    TurnToolSet,
    run_agent_loop,
)


class _Tool:
    def __init__(self, name: str) -> None:
        self.name = name

    async def ainvoke(self, args: dict[str, Any]) -> Any:
        return f"{self.name}:{args.get('value', '')}"

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.name, "parameters": {"type": "object"}},
        }


class _RecordingLLM:
    """Records the tool schemas each request carried; replays scripted turns."""

    model = "fake"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.seen_tools: list[list[str]] = []

    async def chat(self, messages, **kwargs) -> LLMResponse:
        schemas = kwargs.get("tools") or []
        self.seen_tools.append(sorted(s["function"]["name"] for s in schemas))
        return self._responses.pop(0) if self._responses else LLMResponse(content="done")

    def stream(self, messages, **kwargs):
        raise AssertionError("streaming not requested")


def _call(name: str, value: str = "x") -> LLMResponse:
    return LLMResponse(content="", tool_calls=[{
        "id": f"c_{name}", "type": "function",
        "function": {"name": name, "arguments": json.dumps({"value": value})},
    }])


def _config() -> LoopConfig:
    return LoopConfig(max_turns=6, loop_policy=LoopPolicy(no_tool_behavior="stop"), max_llm_retries=1)


def _tool_results(result) -> list[str]:
    return [m["content"] for m in result.messages if m.get("role") == "tool"]


@pytest.mark.asyncio
async def test_no_hook_keeps_the_tools_frozen() -> None:
    llm = _RecordingLLM([_call("a"), LLMResponse(content="done")])
    result = await run_agent_loop(
        system_prompt="s", user_message="go", llm=llm, tools=[_Tool("a")], config=_config(),
    )
    assert llm.seen_tools[0] == ["a"]
    assert _tool_results(result) == ["a:x"]


@pytest.mark.asyncio
async def test_resolve_can_add_a_tool_the_run_did_not_start_with() -> None:
    """Acceptance: a tool bound only from turn 2 is callable on turn 2."""
    a, b = _Tool("a"), _Tool("b")

    def resolve(cfg, metadata, turn) -> TurnToolSet | None:
        return TurnToolSet(tools=[a, b]) if turn >= 2 else None

    llm = _RecordingLLM([_call("a"), _call("b", "y"), LLMResponse(content="done")])
    result = await run_agent_loop(
        system_prompt="s", user_message="go", llm=llm, tools=[a], config=_config(),
        runtime_hooks=AgentLoopHooks(resolve_turn_tools=resolve),
    )
    assert llm.seen_tools[0] == ["a"], "turn 1: only the starting tool"
    assert llm.seen_tools[1] == ["a", "b"], "turn 2: the added tool is bound and shown"
    # b was bound (not in the initial tools=) yet executed rather than being
    # answered as an unknown tool — the whole point of the seam.
    assert _tool_results(result) == ["a:x", "b:y"]


@pytest.mark.asyncio
async def test_resolve_can_remove_a_tool() -> None:
    a, b = _Tool("a"), _Tool("b")

    def resolve(cfg, metadata, turn) -> TurnToolSet | None:
        return TurnToolSet(tools=[a]) if turn >= 2 else None

    # Turn 2 the model calls b from memory; it is no longer bound.
    llm = _RecordingLLM([_call("a"), _call("b"), LLMResponse(content="done")])
    result = await run_agent_loop(
        system_prompt="s", user_message="go", llm=llm, tools=[a, b], config=_config(),
        runtime_hooks=AgentLoopHooks(resolve_turn_tools=resolve),
    )
    assert llm.seen_tools[1] == ["a"], "b's schema is gone from turn 2"
    tool_msgs = _tool_results(result)
    assert tool_msgs[0] == "a:x"
    assert "not available" in tool_msgs[1] or "not executed" in tool_msgs[1]


@pytest.mark.asyncio
async def test_visible_narrows_both_the_shown_and_the_callable_set() -> None:
    """``visible`` reuses ``_llm_allowed_tools``, which hides AND forbids the
    rest — a tool off this turn, not a deferred one."""
    a, b = _Tool("a"), _Tool("b")

    def resolve(cfg, metadata, turn) -> TurnToolSet | None:
        return TurnToolSet(tools=[a, b], visible=frozenset({"a"}))

    # The model calls b anyway; it is not permitted this turn.
    llm = _RecordingLLM([_call("b", "z"), LLMResponse(content="done")])
    result = await run_agent_loop(
        system_prompt="s", user_message="go", llm=llm, tools=[a, b], config=_config(),
        runtime_hooks=AgentLoopHooks(resolve_turn_tools=resolve),
    )
    assert llm.seen_tools[0] == ["a"], "only the visible schema is sent"
    assert "blocked" in _tool_results(result)[0], "the non-visible tool is refused, not run"


@pytest.mark.asyncio
async def test_resolve_is_called_every_turn_with_turn_number() -> None:
    seen: list[int] = []

    def resolve(cfg, metadata, turn) -> TurnToolSet | None:
        seen.append(turn)
        assert cfg.task_id == "t"
        return None

    llm = _RecordingLLM([_call("a"), _call("a"), LLMResponse(content="done")])
    await run_agent_loop(
        system_prompt="s", user_message="go", llm=llm, tools=[_Tool("a")],
        config=LoopConfig(max_turns=6, task_id="t", loop_policy=LoopPolicy(no_tool_behavior="stop")),
        runtime_hooks=AgentLoopHooks(resolve_turn_tools=resolve),
    )
    assert seen == [1, 2, 3]


@pytest.mark.asyncio
async def test_visible_none_leaves_an_observer_narrowing_untouched() -> None:
    """The engine reads ``_llm_allowed_tools`` before ``on_before_llm`` fires,
    so an observer's narrowing takes effect the *next* turn. A ``visible=None``
    resolve must not clobber that key — turn 2 still shows only what the
    observer asked for on turn 1."""
    a, b = _Tool("a"), _Tool("b")

    class _NarrowEveryTurn:
        critical = False

        async def on_before_llm(self, ctx):
            ctx.metadata["_llm_allowed_tools"] = ["a"]
            return None

    def resolve(cfg, metadata, turn) -> TurnToolSet | None:
        return TurnToolSet(tools=[a, b])  # visible=None: do not touch the channel

    llm = _RecordingLLM([_call("a"), _call("a"), LLMResponse(content="done")])
    await run_agent_loop(
        system_prompt="s", user_message="go", llm=llm, tools=[a, b], config=_config(),
        observers=[_NarrowEveryTurn()],
        runtime_hooks=AgentLoopHooks(resolve_turn_tools=resolve),
    )
    assert llm.seen_tools[0] == ["a", "b"], "turn 1: nothing has narrowed yet"
    assert llm.seen_tools[1] == ["a"], "turn 2: the observer's narrowing survived visible=None"
