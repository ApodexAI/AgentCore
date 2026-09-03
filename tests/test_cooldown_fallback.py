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
    # Same keys as the transition degrade, so a host adapter reads one shape.
    degrade = next(payload for name, payload in events if name == "degrade")
    assert degrade["reason"] == "primary_cooldown"
    assert degrade["degrade_from"] == "primary-model"
    assert degrade["degrade_to"] == "fallback-model"
    assert degrade["cooldown_seconds"] == 60
    assert degrade["cooldown_remaining_s"] == 60.0
    assert degrade["error"] == ""


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

    # ``abandon_stream_retry`` is retry-shaped: it says why the primary was not
    # retried. The ``degrade`` that follows is the one carrying the model pair,
    # so a host mapping both to degrade records would write two per degradation.
    # Hence ``abandon_reason``: ``reason`` is degrade's, and nothing else.
    abandon = next(p for n, p in events if n == "abandon_stream_retry")
    assert abandon["abandon_reason"] == "primary_stream_partial"
    assert "reason" not in abandon
    assert "degrade_from" not in abandon
    degrade = next(p for n, p in events if n == "degrade")
    assert degrade["degrade_from"] == "primary"
    assert degrade["degrade_to"] == "fallback"


@pytest.mark.asyncio
async def test_degrades_are_visible_in_logs_without_any_hook(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Logging is the always-wired channel: no ``event_hook``, still visible.

    An operator has to be able to see that traffic moved to another model for
    the cooldown window, and a telemetry sink may not be registered at all.
    """
    llm = CooldownFallbackLLM(
        # The message has to match the retry policy for a retry to happen.
        ScriptedLLM("primary-model", [TimeoutError("timed out")]),
        ScriptedLLM("fallback-model", [LLMResponse(content="ok")]),
        max_retries=2,
        cooldown_seconds=30,
        clock=lambda: 0.0,
        sleep=lambda _d: _completed(),
        jitter=lambda: 0.0,
    )
    with caplog.at_level("DEBUG", logger="agent_core.providers.fallback"):
        await llm.chat([user_msg("x")])

    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("retrying in 0.5s" in line and "TimeoutError" in line for line in warnings)
    assert any(
        "degrading primary-model -> fallback-model (primary_exhausted)" in line
        and "cooldown 30s" in line
        for line in warnings
    )
    # The primary's own error stays at debug: the lines above carry the signal.
    assert [r.levelname for r in caplog.records].count("ERROR") == 0
    assert any("primary leg failed: timed out" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_failing_fallback_leg_logs_an_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = CooldownFallbackLLM(
        ScriptedLLM("primary-model", [TimeoutError("boom")]),
        ScriptedLLM("fallback-model", [RuntimeError("fallback down")]),
        max_retries=1,
        clock=lambda: 0.0,
        sleep=lambda _d: _completed(),
        jitter=lambda: 0.0,
    )
    with (
        caplog.at_level("ERROR", logger="agent_core.providers.fallback"),
        pytest.raises(TimeoutError),
    ):
        await llm.chat([user_msg("x")])

    assert any(
        "fallback leg also failed: fallback down" in r.getMessage()
        for r in caplog.records
    )


def test_model_name_follows_a_swapped_primary() -> None:
    """``model_name`` reads through; ``model`` cannot (see its comment)."""
    primary = ScriptedLLM("first", [])
    llm = CooldownFallbackLLM(primary, ScriptedLLM("fallback", []))
    assert llm.model == "first"
    assert llm.model_name == "fallback(first)"

    primary.model = "rotated"
    assert llm.model_name == "fallback(rotated)"
    assert llm.model == "first"  # the settable protocol slot stays a snapshot


def test_model_label_is_always_a_string() -> None:
    """A duck-typed client may hold a non-str model; telemetry needs a str."""
    class EnumishModel:
        def __str__(self) -> str:
            return "enum-model"

    class OddClient:
        model = EnumishModel()

    llm = CooldownFallbackLLM(OddClient(), ScriptedLLM("fallback", []))
    assert llm.model == "enum-model"
    assert isinstance(llm.model, str)


@pytest.mark.asyncio
async def test_every_degrade_carries_the_documented_keys() -> None:
    """One payload shape across all four degrade sites, ``chat`` and ``stream``.

    A host adapter reading ``payload["error"]`` off a degrade record must not
    hit a ``KeyError`` because the degradation happened to be a streaming one.
    """
    events: list[tuple[str, dict[str, object]]] = []

    async def hook(name: str, payload: dict[str, object]) -> None:
        events.append((name, payload))

    def make(primary: object, fallback: object) -> CooldownFallbackLLM:
        return CooldownFallbackLLM(
            primary,
            fallback,
            max_retries=1,
            cooldown_seconds=60,
            clock=lambda: 0.0,
            sleep=lambda _d: _completed(),
            jitter=lambda: 0.0,
            event_hook=hook,
        )

    chat = make(
        ScriptedLLM("primary-model", [TimeoutError("timed out")]),
        ScriptedLLM("fallback-model", [LLMResponse(content="ok")]),
    )
    await chat.chat([user_msg("one")])  # primary_exhausted
    await chat.chat([user_msg("two")])  # primary_cooldown

    stream = make(PartialStream(), OkStream())
    async for _delta in stream.stream([user_msg("x")]):  # primary_stream_*
        pass
    async for _delta in stream.stream([user_msg("y")]):  # ..._stream_cooldown
        pass

    degrades = [payload for name, payload in events if name == "degrade"]
    assert [d["reason"] for d in degrades] == [
        "primary_exhausted",
        "primary_cooldown",
        "primary_stream_exhausted",
        "primary_stream_cooldown",
    ]
    for degrade in degrades:
        assert degrade["leg"] == "fallback"
        assert degrade["degrade_from"] and degrade["degrade_to"]
        assert degrade["cooldown_seconds"] == 60
        assert isinstance(degrade["error"], str)
    assert degrades[0]["error"] == "timed out"
    assert degrades[1]["error"] == ""


@pytest.mark.asyncio
async def test_cooldown_degrade_logs_at_debug_not_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The transition is a warning; the window repeating itself is not.

    At 20 req/s a 60 s cooldown would otherwise emit ~1200 WARNING lines and
    bury the one line that marks the actual move to the other model.
    """
    llm = CooldownFallbackLLM(
        ScriptedLLM("primary-model", [TimeoutError("timed out")]),
        ScriptedLLM("fallback-model", [LLMResponse(content="ok")]),
        max_retries=1,
        cooldown_seconds=60,
        clock=iter([0.0, 0.0, 1.0]).__next__,
        sleep=lambda _d: _completed(),
        jitter=lambda: 0.0,
    )
    with caplog.at_level("DEBUG", logger="agent_core.providers.fallback"):
        await llm.chat([user_msg("one")])  # the transition
        caplog.clear()
        await llm.chat([user_msg("two")])  # served from the cooldown window

    assert [r.levelname for r in caplog.records] == ["DEBUG"]
    # ..and it reports what is left of the window, not the configured constant.
    assert "59s remaining of 60" in caplog.records[0].getMessage()


@pytest.mark.asyncio
async def test_failing_fallback_leg_in_cooldown_is_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Cooldown-window traffic is most fallback traffic; its failures count.

    Without this the whole cooldown window can fail silently: no ``error``
    event, no error log, and a ``request`` record that never terminates.
    """
    events: list[tuple[str, dict[str, object]]] = []

    async def hook(name: str, payload: dict[str, object]) -> None:
        events.append((name, payload))

    llm = CooldownFallbackLLM(
        ScriptedLLM("primary-model", [TimeoutError("timed out")]),
        ScriptedLLM(
            "fallback-model",
            [LLMResponse(content="ok"), RuntimeError("fallback down")],
        ),
        max_retries=1,
        cooldown_seconds=60,
        clock=lambda: 0.0,
        sleep=lambda _d: _completed(),
        jitter=lambda: 0.0,
        event_hook=hook,
    )
    await llm.chat([user_msg("one")])  # enters cooldown
    events.clear()
    with (
        caplog.at_level("ERROR", logger="agent_core.providers.fallback"),
        pytest.raises(RuntimeError, match="fallback down"),
    ):
        await llm.chat([user_msg("two")])

    assert [(name, payload["leg"]) for name, payload in events] == [
        ("degrade", "fallback"),
        ("request", "fallback"),
        ("error", "fallback"),
    ]
    assert any(
        "fallback leg also failed: fallback down" in r.getMessage()
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_failing_fallback_stream_in_cooldown_is_reported() -> None:
    events: list[tuple[str, dict[str, object]]] = []

    async def hook(name: str, payload: dict[str, object]) -> None:
        events.append((name, payload))

    class DownStream:
        model = "fallback-model"

        async def stream(
            self,
            _messages: list[Message],
            **_kwargs: object,
        ) -> AsyncIterator[StreamDelta]:
            raise RuntimeError("fallback down")
            yield StreamDelta(content="")  # pragma: no cover

    llm = CooldownFallbackLLM(
        ScriptedLLM("primary-model", [TimeoutError("timed out")]),
        DownStream(),
        max_retries=1,
        cooldown_seconds=60,
        clock=lambda: 0.0,
        sleep=lambda _d: _completed(),
        jitter=lambda: 0.0,
        event_hook=hook,
    )
    llm._cooldown_until = 60.0  # already in the window
    with pytest.raises(RuntimeError, match="fallback down"):
        async for _delta in llm.stream([user_msg("x")]):
            pass

    assert [(name, payload["leg"]) for name, payload in events] == [
        ("degrade", "fallback"),
        ("request", "fallback"),
        ("error", "fallback"),
    ]
