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

from copy import deepcopy
from types import SimpleNamespace
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
        max_turns=4, max_llm_retries=1, retry_wait_fixed=0,
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


class _SequenceLLM:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[Any] = []

    async def chat(self, messages, **_kwargs) -> LLMResponse:
        self.requests.append(deepcopy(messages))
        return self.responses.pop(0)


class _EchoTool:
    name = "echo"

    async def ainvoke(self, args: dict[str, Any]) -> str:
        return str(args["value"])

    def to_openai_schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {
            "name": self.name, "parameters": {"type": "object"},
        }}


def _tool_response(index: int) -> LLMResponse:
    return LLMResponse(tool_calls=[{
        "id": f"call_{index}", "type": "function",
        "function": {"name": "echo", "arguments": '{"value":"ok"}'},
    }])


@pytest.mark.asyncio
async def test_each_empty_episode_gets_a_fresh_retry_budget() -> None:
    llm = _SequenceLLM([
        LLMResponse(), LLMResponse(), _tool_response(1),
        LLMResponse(), LLMResponse(), _tool_response(2),
        LLMResponse(), LLMResponse(), LLMResponse(content="finished"),
    ])
    result = await run_agent_loop(
        system_prompt="s", user_message="u", llm=llm, tools=[_EchoTool()],
        config=_cfg(), model_profile=ModelProfile(model_id="m", provider="p"),
    )
    assert result.final_content == "finished"
    assert result.stopped_by == "no_tool"
    assert result.turns_used == 3
    assert result.tool_calls_count == 2
    assert len(llm.requests) == 9


@pytest.mark.asyncio
@pytest.mark.parametrize("blank", ["", " \n\t", [], [{"type": "text", "text": " "}]])
async def test_resamples_preserve_history_and_skip_response_observers(blank) -> None:
    class Observer:
        def __init__(self) -> None:
            self.responses: list[Any] = []

        async def on_llm_response(self, ctx):
            self.responses.append(ctx.ai_text)

    observer = Observer()
    llm = _SequenceLLM([
        _tool_response(1), LLMResponse(content=blank), LLMResponse(content="finished"),
    ])
    result = await run_agent_loop(
        system_prompt="s", user_message="u", llm=llm, tools=[_EchoTool()],
        observers=[observer], config=_cfg(),
        model_profile=ModelProfile(model_id="m", provider="p"),
    )
    assert llm.requests[1] == llm.requests[2]
    assert llm.requests[2][-1]["role"] == "tool"
    assert observer.responses == ["", "finished"]
    assert [m for m in result.messages if m["role"] == "assistant"] == [
        {"role": "assistant", "content": "", "tool_calls": _tool_response(1).tool_calls},
        {"role": "assistant", "content": "finished"},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("response,thinking_format", [
    (LLMResponse(reasoning_content="real reasoning"), "reasoning_content"),
    (LLMResponse(content="<think>real reasoning</think>"), "tag"),
    (LLMResponse(content=[{"type": "thinking", "thinking": "real reasoning", "signature": "sig"}]), "content_block"),
    (LLMResponse(content=[{"type": "redacted_thinking", "data": "opaque"}]), "content_block"),
    (LLMResponse(content=[{"type": "reasoning", "encrypted_content": "opaque"}]), "content_block"),
    (LLMResponse(usage={"prompt_tokens": 12, "completion_tokens": 0}), "none"),
    (LLMResponse(usage={"prompt_tokens": 0, "completion_tokens": 0}), "none"),
    (LLMResponse(finish_reason="refusal", response_metadata={"stop_details": {"type": "refusal"}}), "none"),
])
async def test_nonempty_raw_responses_are_not_resampled(response, thinking_format) -> None:
    llm = _SequenceLLM([response])
    result = await run_agent_loop(
        system_prompt="s", user_message="u", llm=llm, tools=[], config=_cfg(),
        model_profile=ModelProfile(model_id="m", provider="p", thinking_format=thinking_format),
    )
    assert len(llm.requests) == 1
    assert result.stopped_by == ("refusal" if response.finish_reason == "refusal" else "no_tool")
    assert result.turns_used == 1


@pytest.mark.asyncio
async def test_no_retry_budget_does_not_commit_the_blank_response() -> None:
    llm = _SequenceLLM([LLMResponse(content=" ")])
    result = await _loop(llm, empty_completion_max_retries=0)
    assert len(llm.requests) == 1
    assert result.stopped_by == "empty_completion"
    assert all(m["role"] != "assistant" for m in result.messages)


@pytest.mark.asyncio
async def test_blank_resamples_on_the_only_available_turn() -> None:
    llm = _SequenceLLM([LLMResponse(), LLMResponse(), LLMResponse(content="finished")])
    result = await run_agent_loop(
        system_prompt="s", user_message="u", llm=llm, tools=[],
        config=LoopConfig(max_turns=1, max_llm_retries=1, retry_wait_fixed=0),
    )
    assert result.final_content == "finished"
    assert result.turns_used == 1


@pytest.mark.parametrize("response", [
    SimpleNamespace(content="", tool_calls=[], additional_kwargs={"reasoning_content": "thinking"}),
    SimpleNamespace(content="", tool_calls=[], usage_metadata={"input_tokens": 12}),
    SimpleNamespace(content="", tool_calls=[], response_metadata={"token_usage": {"prompt_tokens": 12}}),
    SimpleNamespace(content=["legacy text"], tool_calls=[]),
    SimpleNamespace(content=[{"type": "text", "content": "legacy text"}], tool_calls=[]),
])
def test_wrapper_payloads_are_preserved(response) -> None:
    from agent_core.runtime.loop.llm_client import is_wholly_empty_response

    assert not is_wholly_empty_response(response)


@pytest.mark.asyncio
@pytest.mark.parametrize("delta", [
    StreamDelta(reasoning_blocks=[{"type": "redacted_thinking", "data": "opaque"}]),
    StreamDelta(usage={"prompt_tokens": 0, "completion_tokens": 0}),
    StreamDelta(finish_reason="refusal", stop_details={"type": "refusal"}),
])
async def test_stream_guard_preserves_opaque_usage_and_refusal_signals(delta) -> None:
    assert isinstance(await _run_stream([delta]), LLMResponse)
