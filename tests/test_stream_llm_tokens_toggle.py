"""Who decides whether a turn streams, and what a protocol gate may not hide.

Streaming used to be chosen entirely by the run: the reasoning-only watchdog or
an observer declaring ``wants_llm_delta`` asked for deltas, and three native
protocols were then suppressed unconditionally. That suppression was silent, so
a profile configuring the watchdog on one of them got no watchdog and no
warning, and a host whose GATEWAY requires streaming had no way to say so --
llm-hub's "Claude Code" channel abandons a non-streaming request after ~76s of
waiting for response headers, which for a slow model is most real turns.
"""
from __future__ import annotations

from typing import Any

import pytest

from agent_core.llm import LLMResponse, StreamDelta
from agent_core.loop_types import LoopConfig, LoopPolicy
from agent_core.runtime.loop.agent_loop import (
    UNVERIFIED_STREAM_PROTOCOLS,
    run_agent_loop,
)
from agent_core.runtime.loop.model_profile import ModelProfile


class RecordingLLM:
    """Answers either way and records which surface the loop reached for."""

    def __init__(self) -> None:
        self.chat_calls = 0
        self.stream_calls = 0

    async def chat(self, messages, **_kwargs) -> LLMResponse:
        self.chat_calls += 1
        return LLMResponse(content="finished")

    async def stream(self, messages, **_kwargs):
        self.stream_calls += 1
        yield StreamDelta(content="finished")
        yield StreamDelta(content="", finish_reason="stop")


class DeltaObserver:
    wants_llm_delta = True

    def __init__(self) -> None:
        self.deltas: list[str] = []

    async def on_llm_delta(self, ctx: Any) -> None:
        # ``LLMDeltaContext``, not the provider's raw chunk.
        self.deltas.append(getattr(ctx, "delta", "") or "")


def _config(**kwargs: Any) -> LoopConfig:
    return LoopConfig(
        max_turns=2, max_llm_retries=1,
        loop_policy=LoopPolicy(no_tool_behavior="stop"), **kwargs,
    )


async def _run(llm: RecordingLLM, *, protocol: str, observers: list[Any] | None = None,
               **cfg: Any) -> None:
    await run_agent_loop(
        system_prompt="system", user_message="start", llm=llm, tools=[],
        config=_config(**cfg),
        model_profile=ModelProfile(model_id="m", provider="p", protocol=protocol),
        observers=observers or [],
    )


@pytest.mark.asyncio
async def test_anthropic_streams_when_an_observer_wants_deltas() -> None:
    """The regression this file exists for.

    ``anthropic`` was suppressed one day before the provider substrate taught
    ``AnthropicClient.stream`` to rebuild the verbatim block list (signature
    deltas, redacted thinking) that makes a streamed turn replay like its
    non-streaming twin. It stayed suppressed for a month.
    """
    llm, observer = RecordingLLM(), DeltaObserver()
    await _run(llm, protocol="anthropic", observers=[observer])
    assert (llm.stream_calls, llm.chat_calls) == (1, 0)
    assert "finished" in "".join(observer.deltas)


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", sorted(UNVERIFIED_STREAM_PROTOCOLS))
async def test_unverified_protocols_stay_non_streaming_but_say_so(
    protocol, caplog,
) -> None:
    """Silence was the bug, not the suppression: these two have no test proving
    their streamed replay, so the default holds — and now announces itself."""
    llm = RecordingLLM()
    with caplog.at_level("WARNING"):
        await _run(llm, protocol=protocol, observers=[DeltaObserver()])
    assert (llm.stream_calls, llm.chat_calls) == (0, 1)
    assert any("watchdog is inert" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["anthropic", "bedrock", "chat_completions"])
async def test_explicit_true_streams_on_any_protocol(protocol) -> None:
    """The transport, not the observers, can be the reason to stream: a host
    that states it takes responsibility for its protocol's replay fidelity."""
    llm = RecordingLLM()
    await _run(llm, protocol=protocol, stream_llm_tokens=True)
    assert (llm.stream_calls, llm.chat_calls) == (1, 0)


@pytest.mark.asyncio
async def test_explicit_false_outranks_a_delta_hungry_observer() -> None:
    llm, observer = RecordingLLM(), DeltaObserver()
    await _run(llm, protocol="chat_completions", observers=[observer],
               stream_llm_tokens=False)
    assert (llm.stream_calls, llm.chat_calls) == (0, 1)
    assert observer.deltas == []


@pytest.mark.asyncio
async def test_nothing_asking_keeps_the_cheaper_non_streaming_call() -> None:
    llm = RecordingLLM()
    await _run(llm, protocol="chat_completions")
    assert (llm.stream_calls, llm.chat_calls) == (0, 1)


@pytest.mark.asyncio
async def test_reasoning_only_watchdog_also_selects_streaming() -> None:
    """The watchdog reads the stream, so configuring it is itself a request."""
    llm = RecordingLLM()
    await _run(llm, protocol="anthropic", reasoning_only_timeout_s=120)
    assert (llm.stream_calls, llm.chat_calls) == (1, 0)


def test_the_verified_set_is_stated_not_guessed() -> None:
    """``anthropic`` leaving this set is the behaviour change; pin it so a
    future edit has to be deliberate."""
    assert frozenset({"responses", "bedrock"}) == UNVERIFIED_STREAM_PROTOCOLS
