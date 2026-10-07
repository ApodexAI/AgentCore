"""An Anthropic stream that ends before ``message_stop`` is a dropped connection.

Measured on ApodexHarness's 2026-10-07 GDPval batch over llm-hub: 10 of 15
finished trials ended on a turn shaped exactly like the fixture below --
``message_start`` usage, a thinking block, then nothing: no text, no tool call,
no ``message_delta``. Several unrelated in-flight streams were cut in the same
second, three times. The SDK raised nothing (no SSE ``error`` event, no broken
chunked body), the assembled response carried usage so the empty-completion
guard let it through, and ``no_tool`` ended each run with its work discarded.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_core.errors import LLMTruncatedStream
from agent_core.loop_types import LoopConfig, LoopPolicy
from agent_core.messages import user_msg
from agent_core.providers import anthropic as ac
from agent_core.providers._stream_activity import STREAM_TERMINATOR_ENV
from agent_core.retry_policy import legacy_retryable
from agent_core.runtime.loop.agent_loop import run_agent_loop
from agent_core.runtime.loop.model_profile import ModelProfile
from agent_core.runtime.retriable import (
    classify_error,
    is_empty_completion,
    is_retriable_with_fallback,
    is_transient_network,
    is_truncated_stream,
)


def _message_start() -> SimpleNamespace:
    return SimpleNamespace(type="message_start", message=SimpleNamespace(
        model="claude-x",
        usage=SimpleNamespace(input_tokens=158, cache_read_input_tokens=9326),
    ))


def _thinking(idx: int = 0) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(type="content_block_start", index=idx,
                        content_block=SimpleNamespace(type="thinking")),
        SimpleNamespace(type="content_block_delta", index=idx, delta=SimpleNamespace(
            type="thinking_delta", thinking="I need to write the STEP file next.")),
        SimpleNamespace(type="content_block_delta", index=idx, delta=SimpleNamespace(
            type="signature_delta", signature="sig")),
        SimpleNamespace(type="content_block_stop", index=idx),
    ]


def _truncated() -> list[SimpleNamespace]:
    """The measured shape: usage and thinking, then the body just ends."""
    return [_message_start(), *_thinking()]


def _tool_call_turn() -> list[SimpleNamespace]:
    return [
        _message_start(),
        *_thinking(),
        SimpleNamespace(type="content_block_start", index=1, content_block=SimpleNamespace(
            type="tool_use", id="toolu_1", name="bash", input={})),
        SimpleNamespace(type="content_block_delta", index=1, delta=SimpleNamespace(
            type="input_json_delta", partial_json='{"command": "ls"}')),
        SimpleNamespace(type="content_block_stop", index=1),
        SimpleNamespace(type="message_delta", delta=SimpleNamespace(stop_reason="tool_use"),
                        usage=SimpleNamespace(output_tokens=40)),
        SimpleNamespace(type="message_stop"),
    ]


def _final_text_turn() -> list[SimpleNamespace]:
    return [
        _message_start(),
        SimpleNamespace(type="content_block_start", index=0,
                        content_block=SimpleNamespace(type="text")),
        SimpleNamespace(type="content_block_delta", index=0,
                        delta=SimpleNamespace(type="text_delta", text="done")),
        SimpleNamespace(type="content_block_stop", index=0),
        SimpleNamespace(type="message_delta", delta=SimpleNamespace(stop_reason="end_turn"),
                        usage=SimpleNamespace(output_tokens=2)),
        SimpleNamespace(type="message_stop"),
    ]


def _client(*streams: list[SimpleNamespace]) -> ac.AnthropicClient:
    pending = list(streams)

    async def create(**_kwargs: Any) -> Any:
        events = pending.pop(0)

        async def gen() -> Any:
            for event in events:
                yield event

        return gen()

    c = ac.AnthropicClient("claude-x", api_key="x")
    c._client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(side_effect=create)))
    return c


@pytest.mark.asyncio
async def test_stream_without_message_stop_raises_with_its_end_record() -> None:
    c = _client(_truncated())

    with pytest.raises(LLMTruncatedStream) as info:
        _ = [d async for d in c.stream([user_msg("go")])]

    err = info.value
    assert err.last_event == "content_block_stop"
    assert err.events_seen == 5
    assert err.saw_message_delta is False
    assert err.block_types == ["thinking"]
    assert "ended without message_stop" in str(err)


@pytest.mark.asyncio
async def test_message_delta_alone_is_still_truncated() -> None:
    """``message_stop`` is the terminator; a stop_reason without it was cut too."""
    events = [*_tool_call_turn()[:-1]]  # drop only message_stop
    c = _client(events)

    with pytest.raises(LLMTruncatedStream) as info:
        _ = [d async for d in c.stream([user_msg("go")])]

    assert info.value.saw_message_delta is True
    assert info.value.last_event == "message_delta"


@pytest.mark.asyncio
async def test_complete_stream_is_untouched() -> None:
    deltas = [d async for d in _client(_tool_call_turn()).stream([user_msg("go")])]
    assert deltas[-1].stop_reason == "tool_use"


def test_classified_as_truncated_stream() -> None:
    """Resample on the same key; an active chain advances, like empty completion."""
    err = LLMTruncatedStream(last_event="content_block_stop", events_seen=5,
                             saw_message_delta=False, block_types=["thinking"],
                             elapsed_s=776.0)
    assert is_truncated_stream(err)
    assert classify_error(err) == "truncated_stream"
    assert is_retriable_with_fallback(err)
    assert not is_transient_network(err)
    assert not is_empty_completion(err)
    assert legacy_retryable(err)


@pytest.mark.asyncio
async def test_escape_hatch_accepts_the_partial_turn(monkeypatch, caplog) -> None:
    monkeypatch.setenv(STREAM_TERMINATOR_ENV, "0")
    with caplog.at_level("WARNING"):
        deltas = [d async for d in _client(_truncated()).stream([user_msg("go")])]
    assert deltas[-1].usage
    assert "ended without message_stop" in caplog.text


@pytest.mark.asyncio
async def test_loop_resamples_a_truncated_turn_instead_of_ending_the_run() -> None:
    llm = _client(_truncated(), _tool_call_turn(), _final_text_turn())
    tool = MagicMock()
    tool.name = "bash"
    tool.ainvoke = AsyncMock(return_value="ok")

    result = await run_agent_loop(
        system_prompt="system", user_message="start", llm=llm, tools=[tool],
        config=LoopConfig(
            max_turns=5, max_llm_retries=3, retry_wait_fixed=0,
            stream_llm_tokens=True,
            loop_policy=LoopPolicy(no_tool_behavior="stop"),
        ),
        model_profile=ModelProfile(model_id="claude-x", provider="p", protocol="anthropic"),
    )

    assert llm._client.messages.create.await_count == 3
    tool.ainvoke.assert_awaited_once()
    assert result.stopped_by == "no_tool"
    assert result.final_content == "done"
