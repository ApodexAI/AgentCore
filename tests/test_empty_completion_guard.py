"""A response with nothing in it is an upstream failure, not an answer.

A reply with no tool call is shaped exactly like "the model chose to stop
talking", and under ``no_tool_behavior="stop"`` that ends the run. So a blank
reply — no text, no tool call, no usage — silently discards every turn of work
already done and records ``stopped_by="no_tool"``, which reads as a clean
finish. Measured on ApodexHarness's 2026-10-06 GDPval batch over llm-hub: 10 of
19 streamed trials died this way, six inside five minutes, one on turn 1 at 37s.

``is_empty_completion`` already named "an empty stream" and already routed it to
a same-key resample then a chain advance; the raise was what was missing. Two
lines of defence are tested here:

  1. ``_stream_llm_response`` raises ``LLMEmptyCompletion`` on a wholly empty
     stream, so it never becomes an ``LLMResponse`` at all.
  2. ``run_agent_loop`` resamples a blank reply that reaches it anyway (a
     non-streamed call, or one a product wrapper rebuilt) instead of taking the
     no-tool exit.
"""
from __future__ import annotations

from typing import Any

import pytest

from agent_core.errors import LLMEmptyCompletion
from agent_core.llm import LLMResponse, StreamDelta
from agent_core.loop_types import LoopConfig, LoopPolicy
from agent_core.runtime.loop._streaming import _stream_llm_response
from agent_core.runtime.loop.agent_loop import run_agent_loop
from agent_core.runtime.loop.model_profile import ModelProfile
from agent_core.runtime.retriable import is_empty_completion, is_retriable_with_fallback


class _Stream:
    def __init__(self, deltas: list[StreamDelta]) -> None:
        self._deltas = deltas

    async def stream(self, messages, timeout=None):
        for d in self._deltas:
            yield d


async def _noop(*_a: Any, **_k: Any) -> None:
    return None


async def _run_stream(deltas: list[StreamDelta]):
    return await _stream_llm_response(
        _Stream(deltas), messages=[], timeout=30, on_delta=_noop,
    )


# ── line 1: the stream never yields a blank response ──


@pytest.mark.asyncio
async def test_a_stream_that_yields_nothing_raises() -> None:
    with pytest.raises(LLMEmptyCompletion) as caught:
        await _run_stream([])
    assert caught.value.chunks_seen == 0


@pytest.mark.asyncio
async def test_the_raised_error_is_classified_by_existing_rules() -> None:
    """No registry edit should be needed: the wording matches the patterns
    ``is_empty_completion`` has always carried, and the recovery it selects
    (resample on the same key, then advance the chain) is the documented one."""
    err = LLMEmptyCompletion(chunks_seen=0, elapsed_s=1.0)
    assert is_empty_completion(err)
    assert is_retriable_with_fallback(err)


@pytest.mark.asyncio
@pytest.mark.parametrize("deltas,why", [
    ([StreamDelta(content="hi")], "visible text"),
    ([StreamDelta(reasoning_content="thinking")], "reasoning only"),
    ([StreamDelta(tool_call_deltas=[{"index": 0, "id": "c", "name": "bash"}])], "a usable tool call"),
    ([StreamDelta(usage={"prompt_tokens": 12, "completion_tokens": 0})], "usage but no text"),
])
async def test_anything_at_all_is_not_empty(deltas, why) -> None:
    """The test is "nothing whatsoever", not "no visible text".

    A real 0-token reply still reports usage, and a reasoning-only stream is the
    runaway guard's business — both must keep flowing to the handlers that
    already reason about them. Only the wholly blank response raises.
    """
    response = await _run_stream(deltas)
    assert isinstance(response, LLMResponse), why


# ── line 2: the loop resamples a blank reply instead of stopping ──


class _BlankThenAnswer:
    """Returns N wholly blank replies, then a real one."""

    def __init__(self, blanks: int) -> None:
        self.blanks = blanks
        self.calls = 0

    async def chat(self, messages, **_kwargs) -> LLMResponse:
        self.calls += 1
        if self.calls <= self.blanks:
            return LLMResponse(content="")
        return LLMResponse(content="finished")

    def stream(self, *_a: Any, **_k: Any):
        raise AssertionError("streaming was not requested")


def _cfg(**kw: Any) -> LoopConfig:
    return LoopConfig(
        max_turns=4, max_llm_retries=1,
        loop_policy=LoopPolicy(no_tool_behavior="stop"), **kw,
    )


async def _loop(llm: Any, **cfg: Any):
    return await run_agent_loop(
        system_prompt="s", user_message="u", llm=llm, tools=[],
        config=_cfg(**cfg),
        model_profile=ModelProfile(model_id="m", provider="p"),
    )


@pytest.mark.asyncio
async def test_a_blank_reply_is_resampled_not_treated_as_an_answer() -> None:
    llm = _BlankThenAnswer(blanks=1)
    result = await _loop(llm)
    assert result.final_content == "finished"
    assert llm.calls == 2, "the blank reply should have cost a resample, not the run"


@pytest.mark.asyncio
async def test_the_resample_budget_is_bounded_and_says_why_it_stopped() -> None:
    """A provider returning blank forever is down, not slow. The run ends, but
    under its own stop reason — ``no_tool`` would read as a clean finish."""
    llm = _BlankThenAnswer(blanks=99)
    result = await _loop(llm, empty_completion_max_retries=2)
    assert result.stopped_by == "empty_completion"
    assert llm.calls == 3, "two resamples then stop"


@pytest.mark.asyncio
async def test_a_blank_turn_does_not_spend_the_turn_budget() -> None:
    """The resample must not eat a logical turn: on the landing turn there is no
    later turn to recover in."""
    llm = _BlankThenAnswer(blanks=2)
    result = await _loop(llm)
    assert result.final_content == "finished"
    assert result.turns_used == 1
