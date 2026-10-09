"""A reasoning runaway must be steerable and must never look like a finished run.

Measured on ApodexHarness's 2026-10-08 GDPval batch over claude-opus-5-5 at
``effort=max``: three tasks asked to look up live data offline (S&P 500 closing
prices, a priced equipment list, a hardware bill of materials) spent a turn
entirely in private reasoning. The resample ladder went ``8192 → 4096 → 2048``
with ``next_thinking_mode=disabled`` logged on the last rungs, and every rung was
still pure reasoning. Two independent defects made that a lost task:

* the Anthropic adapter never read the ladder's override, so every rung went out
  with the profile's ``effort=max`` and only a smaller ``max_tokens``;
* ``call_llm`` returned the runaway "for loop-level nudge handling", but the loop
  had no such branch: no text and no tool call reached ``no_tool``, and under
  ``no_tool_behavior="stop"`` the run ended with nothing delivered.
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_core.llm import LLMResponse
from agent_core.loop_types import LoopConfig, LoopPolicy
from agent_core.providers import anthropic as ac
from agent_core.runtime.llm_request_overrides import (
    ThinkingRetryOverride,
    thinking_retry_override,
)
from agent_core.runtime.loop import _call
from agent_core.runtime.loop._runaway import RUNAWAY_LOOP_RECOVERY_GUIDANCE
from agent_core.runtime.loop.agent_loop import run_agent_loop
from agent_core.runtime.loop.model_profile import ModelProfile

# ── provider: the ladder's effort reaches the request ─────────────────────


def _request(client: ac.AnthropicClient) -> dict[str, Any]:
    return client._build_kwargs(
        [{"role": "user", "content": "hi"}], tools=None, temperature=None,
        max_tokens=2048, extra_headers=None, timeout=None,
    )


def _opus(effort: str = "max") -> ac.AnthropicClient:
    return ac.AnthropicClient(
        "claude-opus-5-5", api_key="x", max_tokens=128000,
        thinking={"type": "adaptive", "display": "summarized"}, effort=effort,
    )


def test_profile_effort_is_sent_without_an_override() -> None:
    assert _request(_opus())["extra_body"] == {"output_config": {"effort": "max"}}


@pytest.mark.parametrize("mode", ["expanded", "reduced", "disabled"])
def test_retry_override_replaces_effort_and_never_touches_thinking(mode: str) -> None:
    """``disabled`` is expressed through effort: opus-5-5 rejects
    ``thinking={"type": "disabled"}`` with a 400."""
    client = _opus()
    with thinking_retry_override(ThinkingRetryOverride(mode=mode, reasoning_effort="low")):
        request = _request(client)

    assert request["extra_body"] == {"output_config": {"effort": "low"}}
    assert request["thinking"] == {"type": "adaptive", "display": "summarized"}


def test_override_without_effort_keeps_the_profile_effort() -> None:
    with thinking_retry_override(ThinkingRetryOverride(mode="reduced", thinking_budget=512)):
        request = _request(_opus())
    assert request["extra_body"] == {"output_config": {"effort": "max"}}


def test_client_without_effort_does_not_gain_one() -> None:
    """Like the OpenAI adapter: replace an effort the profile opted into, never add one."""
    client = ac.AnthropicClient("claude-x", api_key="x")
    with thinking_retry_override(ThinkingRetryOverride(mode="disabled", reasoning_effort="low")):
        assert "extra_body" not in _request(client)


# ── loop: a runaway turn is recovered, and a persistent one is named ──────


def _runaway() -> LLMResponse:
    return LLMResponse(
        content="", reasoning_content="x" * 5_000, finish_reason="length",
        usage={"prompt_tokens": 10, "completion_tokens": 2_048},
    )


def _tool_call(n: int) -> LLMResponse:
    return LLMResponse(
        content="", finish_reason="tool_calls",
        tool_calls=[{"type": "function", "id": f"c{n}",
                     "function": {"name": "bash", "arguments": json.dumps({"command": "ls"})}}],
        usage={"prompt_tokens": 10, "completion_tokens": 20},
    )


def _final() -> LLMResponse:
    return LLMResponse(content="done", finish_reason="stop",
                       usage={"prompt_tokens": 10, "completion_tokens": 2})


class _Scripted:
    model = "fake"

    def __init__(self, *responses: LLMResponse) -> None:
        self._pending = list(responses)
        self.requests: list[list[Any]] = []

    async def chat(self, messages: list[Any], **_kwargs: Any) -> LLMResponse:
        self.requests.append(list(messages))
        return self._pending.pop(0)


@pytest.fixture
def no_call_ladder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hand every runaway straight back to the loop, as an exhausted ladder does."""
    monkeypatch.setattr(_call, "_RUNAWAY_MAX_RETRIES", 0)
    monkeypatch.setattr(_call, "_RUNAWAY_BACKOFF_S", 0.0)


def _tool() -> MagicMock:
    tool = MagicMock()
    tool.name = "bash"
    tool.ainvoke = AsyncMock(return_value="ok")
    return tool


async def _run(llm: _Scripted, tool: MagicMock, **cfg: Any) -> Any:
    return await run_agent_loop(
        system_prompt="system", user_message="start", llm=llm, tools=[tool],
        config=LoopConfig(
            max_turns=8, max_llm_retries=2, retry_wait_fixed=0,
            stream_llm_tokens=False,
            loop_policy=LoopPolicy(no_tool_behavior="stop"),
            **cfg,
        ),
        model_profile=ModelProfile(model_id="fake", provider="p", protocol="openai"),
    )


@pytest.mark.asyncio
@pytest.mark.usefixtures("no_call_ladder")
async def test_runaway_turn_is_recovered_instead_of_ending_as_no_tool() -> None:
    llm, tool = _Scripted(_runaway(), _tool_call(1), _final()), _tool()

    result = await _run(llm, tool)

    tool.ainvoke.assert_awaited_once()
    assert result.final_content == "done"
    assert result.stopped_by == "no_tool"  # the real, chosen finish
    assert any(m.get("content") == RUNAWAY_LOOP_RECOVERY_GUIDANCE for m in llm.requests[1])


@pytest.mark.asyncio
@pytest.mark.usefixtures("no_call_ladder")
async def test_persistent_runaway_stops_with_its_own_reason() -> None:
    llm, tool = _Scripted(_runaway(), _runaway()), _tool()

    result = await _run(llm, tool)

    assert len(llm.requests) == 2  # one turn + one loop-level recovery
    tool.ainvoke.assert_not_awaited()
    assert result.stopped_by == "reasoning_runaway"


@pytest.mark.asyncio
@pytest.mark.usefixtures("no_call_ladder")
async def test_recovery_allowance_resets_after_progress() -> None:
    llm = _Scripted(_runaway(), _tool_call(1), _runaway(), _tool_call(2), _final())
    tool = _tool()

    result = await _run(llm, tool)

    assert tool.ainvoke.await_count == 2
    assert result.final_content == "done"


@pytest.mark.asyncio
@pytest.mark.usefixtures("no_call_ladder")
async def test_zero_allowance_still_names_the_failure() -> None:
    llm, tool = _Scripted(_runaway()), _tool()

    result = await _run(llm, tool, runaway_max_loop_recoveries=0)

    assert len(llm.requests) == 1
    assert result.stopped_by == "reasoning_runaway"
