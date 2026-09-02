from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from agent_core.llm import LLMResponse, StreamDelta
from agent_core.messages import Message, user_msg
from agent_core.providers.fallback import CooldownFallbackLLM, legacy_retryable


class ScriptedLLM:
    def __init__(self, model: str, script: list[LLMResponse | Exception]) -> None:
        self.model = model
        self.script = script
        self.calls = 0
        self.kwargs: list[dict[str, object]] = []

    async def chat(self, _messages: list[Message], **kwargs: object) -> LLMResponse:
        self.kwargs.append(kwargs)
        item = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item

    async def stream(
        self,
        messages: list[Message],
        **kwargs: object,
    ) -> AsyncIterator[StreamDelta]:
        response = await self.chat(messages, **kwargs)
        yield StreamDelta(content=str(response.content))


@pytest.mark.asyncio
async def test_cooldown_fallback_retries_then_skips_primary() -> None:
    primary = ScriptedLLM("primary", [TimeoutError("timeout")])
    fallback = ScriptedLLM("fallback", [LLMResponse(content="ok")])
    sleeps: list[float] = []

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    clock = iter([10.0, 10.0, 11.0]).__next__
    llm = CooldownFallbackLLM(
        primary,
        fallback,
        max_retries=1,
        cooldown_seconds=60,
        clock=clock,
        sleep=sleep,
        jitter=lambda: 0.0,
    )
    assert (await llm.chat([user_msg("one")])).content == "ok"
    assert (await llm.chat([user_msg("two")])).content == "ok"
    assert primary.calls == 1
    assert fallback.calls == 2
    assert sleeps == [0.5]


@pytest.mark.asyncio
async def test_cooldown_fallback_preserves_kwargs_and_original_error() -> None:
    original = TimeoutError("primary failed")
    primary = ScriptedLLM("primary", [original])
    fallback = ScriptedLLM("fallback", [RuntimeError("fallback failed")])
    llm = CooldownFallbackLLM(
        primary,
        fallback,
        max_retries=1,
        cooldown_seconds=0,
        sleep=lambda _delay: _completed(),
        jitter=lambda: 0.0,
    )
    tools = [{"type": "function", "function": {"name": "search"}}]
    with pytest.raises(TimeoutError, match="primary failed"):
        await llm.chat([user_msg("one")], tools=tools)
    assert primary.kwargs[0]["tools"] == tools
    assert fallback.kwargs[0]["tools"] == tools


async def _completed() -> None:
    return None


def test_legacy_retryable_contract() -> None:
    assert legacy_retryable(TimeoutError("timed out"))
    assert legacy_retryable(AttributeError("model_dump"))
    assert not legacy_retryable(ValueError("invalid json"))
