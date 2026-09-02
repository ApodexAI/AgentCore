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
    # max_retries=1 means a single attempt: no backoff sleep, since no retry
    # follows it. See test_no_backoff_sleep_after_the_final_attempt.
    assert sleeps == []


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


@pytest.mark.asyncio
async def test_no_backoff_sleep_after_the_final_attempt() -> None:
    """The last attempt must not sleep — no retry follows it."""
    primary = ScriptedLLM("primary", [TimeoutError("timeout")])
    fallback = ScriptedLLM("fallback", [LLMResponse(content="ok")])
    sleeps: list[float] = []
    events: list[str] = []

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    async def hook(name: str, _payload: dict[str, object]) -> None:
        events.append(name)

    llm = CooldownFallbackLLM(
        primary,
        fallback,
        max_retries=1,
        cooldown_seconds=60,
        clock=lambda: 0.0,
        sleep=sleep,
        jitter=lambda: 0.0,
        event_hook=hook,
    )
    assert (await llm.chat([user_msg("one")])).content == "ok"
    assert sleeps == []
    assert "retry" not in events

    # Two attempts sleep exactly once, between them.
    primary2 = ScriptedLLM(
        "primary",
        [TimeoutError("timeout"), TimeoutError("timeout")],
    )
    sleeps.clear()
    llm2 = CooldownFallbackLLM(
        primary2,
        ScriptedLLM("fallback", [LLMResponse(content="ok")]),
        max_retries=2,
        clock=lambda: 0.0,
        sleep=sleep,
        jitter=lambda: 0.0,
    )
    assert (await llm2.chat([user_msg("two")])).content == "ok"
    assert sleeps == [0.5]
    assert primary2.calls == 2


class PartialStream:
    model = "primary"

    def __init__(self) -> None:
        self.starts = 0

    async def stream(
        self,
        _messages: list[Message],
        **_kwargs: object,
    ) -> AsyncIterator[StreamDelta]:
        self.starts += 1
        yield StreamDelta(content="part1 ")
        yield StreamDelta(content="part2 ")
        raise TimeoutError("timeout mid-stream")


class OkStream:
    model = "fallback"

    async def stream(
        self,
        _messages: list[Message],
        **_kwargs: object,
    ) -> AsyncIterator[StreamDelta]:
        yield StreamDelta(content="FALLBACK")


@pytest.mark.asyncio
async def test_stream_does_not_replay_already_yielded_deltas() -> None:
    primary = PartialStream()
    llm = CooldownFallbackLLM(
        primary,
        OkStream(),
        max_retries=3,
        clock=lambda: 0.0,
        sleep=lambda _d: _completed(),
        jitter=lambda: 0.0,
    )
    seen = [delta.content async for delta in llm.stream([user_msg("x")])]
    assert seen == ["part1 ", "part2 ", "FALLBACK"]
    assert primary.starts == 1


@pytest.mark.asyncio
async def test_stream_replay_is_opt_in() -> None:
    primary = PartialStream()
    llm = CooldownFallbackLLM(
        primary,
        OkStream(),
        max_retries=2,
        clock=lambda: 0.0,
        sleep=lambda _d: _completed(),
        jitter=lambda: 0.0,
        replay_partial_stream=True,
    )
    seen = [delta.content async for delta in llm.stream([user_msg("x")])]
    assert seen == ["part1 ", "part2 ", "part1 ", "part2 ", "FALLBACK"]
    assert primary.starts == 2


@pytest.mark.asyncio
async def test_every_event_names_the_leg_it_is_about() -> None:
    """``leg`` is always present, and ``retry`` belongs to the primary.

    A tracing adapter labels each record by ``leg``; leaving it off any event
    forced the adapter to guess, which mislabelled primary retries as fallback
    traffic.
    """
    primary = ScriptedLLM(
        "primary-model",
        [TimeoutError("timeout"), TimeoutError("timeout")],
    )
    fallback = ScriptedLLM("fallback-model", [LLMResponse(content="ok")])
    events: list[tuple[str, dict[str, object]]] = []

    async def hook(name: str, payload: dict[str, object]) -> None:
        events.append((name, payload))

    llm = CooldownFallbackLLM(
        primary,
        fallback,
        max_retries=2,
        cooldown_seconds=60,
        clock=lambda: 0.0,
        sleep=lambda _d: _completed(),
        jitter=lambda: 0.0,
        event_hook=hook,
    )
    assert (await llm.chat([user_msg("x")])).content == "ok"

    assert all("leg" in payload for _name, payload in events), events
    assert [(name, payload["leg"]) for name, payload in events] == [
        ("request", "primary"),
        ("error", "primary"),
        ("retry", "primary"),
        ("request", "primary"),
        ("error", "primary"),
        ("degrade", "fallback"),
        ("request", "fallback"),
    ]
    degrade = next(payload for name, payload in events if name == "degrade")
    assert degrade["degrade_from"] == "primary-model"
    assert degrade["degrade_to"] == "fallback-model"


@pytest.mark.asyncio
async def test_fallback_leg_call_is_traceable_in_cooldown() -> None:
    """The cooldown shortcut still announces the fallback request it makes."""
    fallback = ScriptedLLM("fallback-model", [LLMResponse(content="ok")])
    events: list[tuple[str, dict[str, object]]] = []

    async def hook(name: str, payload: dict[str, object]) -> None:
        events.append((name, payload))

    llm = CooldownFallbackLLM(
        ScriptedLLM("primary-model", [TimeoutError("timeout")]),
        fallback,
        max_retries=1,
        cooldown_seconds=60,
        clock=lambda: 0.0,
        sleep=lambda _d: _completed(),
        jitter=lambda: 0.0,
        event_hook=hook,
    )
    await llm.chat([user_msg("one")])  # enters cooldown
    events.clear()
    await llm.chat([user_msg("two")])  # served from cooldown

    assert [(name, payload.get("mode")) for name, payload in events] == [
        ("degrade", None),
        ("request", "cooldown"),
    ]
    assert fallback.calls == 2


@pytest.mark.asyncio
async def test_stream_events_carry_leg_labels_and_degrade_pair() -> None:
    """Stream-path events reach the same contract as the ``chat`` path."""
    events: list[tuple[str, dict[str, object]]] = []

    async def hook(name: str, payload: dict[str, object]) -> None:
        events.append((name, payload))

    llm = CooldownFallbackLLM(
        PartialStream(),
        OkStream(),
        max_retries=3,
        cooldown_seconds=60,
        clock=lambda: 0.0,
        sleep=lambda _d: _completed(),
        jitter=lambda: 0.0,
        event_hook=hook,
    )
    seen = [delta.content async for delta in llm.stream([user_msg("x")])]
    assert seen == ["part1 ", "part2 ", "FALLBACK"]

    assert [(name, payload["leg"]) for name, payload in events] == [
        ("request", "primary"),
        ("error", "primary"),
        ("abandon_stream_retry", "primary"),
        ("degrade", "fallback"),
        ("request", "fallback"),
    ]
    assert all(payload.get("streaming") for _name, payload in events), events
    for name in ("abandon_stream_retry", "degrade"):
        payload = next(p for n, p in events if n == name)
        assert payload["degrade_from"] == "primary"
        assert payload["degrade_to"] == "fallback"
        assert payload["reason"]
