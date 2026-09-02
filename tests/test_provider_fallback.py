"""``LLMFallbackChain`` unit tests.

Cover the failover contract: primary success short-circuits, matching
trigger falls through to the next entry, non-matching trigger
propagates, exhausted chain re-raises the last error, and successful
calls stamp ``fallback_used`` / ``model_actually_used`` onto the
response metadata.
"""
from __future__ import annotations

from typing import ClassVar

import pytest

from agent_core.llm import LLMResponse, StreamDelta
from agent_core.providers.fallback import (
    FallbackEntry,
    FallbackTrigger,
    LLMFallbackChain,
    with_provider_stamp,
)


class _ScriptedLLM:
    """Native :class:`LLMClient` mock returning a configured answer or raising."""

    def __init__(
        self,
        *,
        name: str = "scripted",
        answer: str = "ok",
        raise_exc: BaseException | None = None,
    ) -> None:
        self.model = name
        self.model_name = name
        self.answer = answer
        self.raise_exc = raise_exc

    async def chat(
        self,
        messages,
        *,
        tools=None,
        temperature=None,
        max_tokens=None,
        extra_headers=None,
        timeout=None,
    ) -> LLMResponse:
        if self.raise_exc is not None:
            raise self.raise_exc
        return LLMResponse(content=self.answer, model=self.model)

    async def stream(
        self,
        messages,
        *,
        tools=None,
        temperature=None,
        max_tokens=None,
        extra_headers=None,
        timeout=None,
    ):
        if self.raise_exc is not None:
            raise self.raise_exc
        from agent_core.llm import StreamDelta

        yield StreamDelta(content=self.answer)


class _Status5xxError(Exception):
    """Mock provider 5xx error carrying a ``status_code`` attr."""

    def __init__(self, status: int, msg: str = "server error") -> None:
        super().__init__(msg)
        self.status_code = status


@pytest.mark.asyncio
async def test_primary_success_short_circuits() -> None:
    primary = _ScriptedLLM(name="primary", answer="from-primary")
    secondary = _ScriptedLLM(name="secondary", answer="from-secondary")
    chain = LLMFallbackChain.from_models([primary, secondary])

    result = await chain.chat([])

    assert result.content == "from-primary"
    assert result.response_metadata["fallback_used"] == 0
    assert result.response_metadata["model_actually_used"] == "primary"


@pytest.mark.asyncio
async def test_provider_actually_used_stamped_on_success() -> None:
    """When ``FallbackEntry.provider`` is set, the served entry's provider
    is stamped on ``response_metadata.provider_actually_used`` so
    downstream billing can attribute the call per vendor."""
    primary = _ScriptedLLM(name="primary", answer="ok-primary")
    secondary = _ScriptedLLM(name="secondary", answer="ok-secondary")
    chain = LLMFallbackChain(
        entries=[
            FallbackEntry(model=primary, provider="openai"),
            FallbackEntry(model=secondary, provider="anthropic"),
        ]
    )
    result = await chain.chat([])
    rmd = result.response_metadata
    assert rmd["provider_actually_used"] == "openai"


@pytest.mark.asyncio
async def test_stream_stamps_provider_on_deltas() -> None:
    """Streaming has no ``response_metadata`` channel, so the chain stamps
    the served leg's vendor onto each ``StreamDelta.provider`` — the
    assembler folds it into the response's metadata downstream. Without
    this, streamed calls (the main agent loop) billed against an empty
    vendor."""
    primary = _ScriptedLLM(name="primary", answer="streamed")
    chain = LLMFallbackChain(
        entries=[FallbackEntry(model=primary, provider="apodex")],
    )
    providers = [delta.provider async for delta in chain.stream([])]
    assert providers and all(p == "apodex" for p in providers)


@pytest.mark.asyncio
async def test_provider_actually_used_reflects_fallback_vendor() -> None:
    """Cross-provider fallback: primary openai fails, anthropic backup
    serves — ``provider_actually_used`` reflects the vendor that actually
    answered, not the primary's vendor."""
    primary = _ScriptedLLM(
        name="primary", raise_exc=TimeoutError("slow"),
    )
    secondary = _ScriptedLLM(name="secondary", answer="from-anthropic")
    chain = LLMFallbackChain(
        entries=[
            FallbackEntry(
                model=primary, triggers=("timeout",), provider="openai",
            ),
            FallbackEntry(
                model=secondary, triggers=(), provider="anthropic",
            ),
        ]
    )
    result = await chain.chat([])
    rmd = result.response_metadata
    assert rmd["fallback_used"] == 1
    assert rmd["provider_actually_used"] == "anthropic"


@pytest.mark.asyncio
async def test_timeout_triggers_fallback() -> None:
    primary = _ScriptedLLM(name="primary", raise_exc=TimeoutError("slow"))
    secondary = _ScriptedLLM(name="secondary", answer="from-secondary")

    chain = LLMFallbackChain(
        entries=[
            FallbackEntry(model=primary, triggers=("timeout",)),
            FallbackEntry(model=secondary, triggers=()),
        ]
    )
    result = await chain.chat([])

    assert result.content == "from-secondary"
    assert result.response_metadata["fallback_used"] == 1
    assert result.response_metadata["model_actually_used"] == "secondary"


@pytest.mark.asyncio
async def test_5xx_status_triggers_fallback() -> None:
    primary = _ScriptedLLM(
        name="primary",
        raise_exc=_Status5xxError(503, "service unavailable"),
    )
    secondary = _ScriptedLLM(name="secondary", answer="recovered")

    chain = LLMFallbackChain(
        entries=[
            FallbackEntry(model=primary, triggers=("5xx",)),
            FallbackEntry(model=secondary, triggers=()),
        ]
    )
    result = await chain.chat([])

    assert result.content == "recovered"
    assert result.response_metadata["fallback_used"] == 1


@pytest.mark.asyncio
async def test_rate_limit_triggers_fallback_via_message_match() -> None:
    primary = _ScriptedLLM(
        name="primary",
        raise_exc=RuntimeError("Quota exceeded for model gpt-4o"),
    )
    secondary = _ScriptedLLM(name="secondary", answer="ok")

    chain = LLMFallbackChain(
        entries=[
            FallbackEntry(model=primary, triggers=("rate_limit",)),
            FallbackEntry(model=secondary, triggers=()),
        ]
    )
    result = await chain.chat([])
    assert result.content == "ok"


@pytest.mark.asyncio
async def test_non_matching_trigger_propagates() -> None:
    """A timeout-only trigger must NOT swallow an unrelated ValueError."""
    primary = _ScriptedLLM(name="primary", raise_exc=ValueError("bad request"))
    secondary = _ScriptedLLM(name="secondary", answer="never reached")

    chain = LLMFallbackChain(
        entries=[
            FallbackEntry(model=primary, triggers=("timeout",)),
            FallbackEntry(model=secondary, triggers=()),
        ]
    )
    with pytest.raises(ValueError, match="bad request"):
        await chain.chat([])


@pytest.mark.asyncio
async def test_bad_request_does_not_trigger_availability_fallback() -> None:
    """A deterministic 400 payload defect should be fixed, not rerouted."""
    primary = _ScriptedLLM(
        name="primary",
        raise_exc=_Status5xxError(400, "zero-length empty document"),
    )
    secondary = _ScriptedLLM(name="secondary", answer="must not be used")
    chain = LLMFallbackChain(
        entries=[
            FallbackEntry(
                model=primary,
                triggers=("timeout", "rate_limit", "5xx"),
            ),
            FallbackEntry(model=secondary, triggers=()),
        ]
    )

    with pytest.raises(_Status5xxError, match="zero-length"):
        await chain.chat([])


@pytest.mark.asyncio
async def test_exhausted_chain_reraises_last_error() -> None:
    err1 = TimeoutError("primary slow")
    err2 = _Status5xxError(502, "secondary down")
    primary = _ScriptedLLM(name="primary", raise_exc=err1)
    secondary = _ScriptedLLM(name="secondary", raise_exc=err2)

    chain = LLMFallbackChain(
        entries=[
            FallbackEntry(model=primary, triggers=("any_error",)),
            FallbackEntry(model=secondary, triggers=("any_error",)),
        ]
    )
    with pytest.raises(_Status5xxError):
        await chain.chat([])


@pytest.mark.asyncio
async def test_any_error_trigger_catches_arbitrary_exception() -> None:
    primary = _ScriptedLLM(name="primary", raise_exc=KeyError("oops"))
    secondary = _ScriptedLLM(name="secondary", answer="ok")

    chain = LLMFallbackChain.from_models([primary, secondary])
    result = await chain.chat([])
    assert result.content == "ok"


def test_empty_entries_list_rejected() -> None:
    with pytest.raises(ValueError, match="at least one entry"):
        LLMFallbackChain(entries=[])


def test_default_triggers_apply_to_unconfigured_entries() -> None:
    primary = _ScriptedLLM(name="primary", answer="ok")
    chain = LLMFallbackChain(
        entries=[FallbackEntry(model=primary)],
        default_triggers=("timeout", "5xx"),
    )
    # ``__post_init__`` fills in the default for an entry that left triggers
    # UNSET (``None``) — the only shape that means "not configured".
    assert chain.entries[0].triggers == ("timeout", "5xx")


def test_explicit_empty_triggers_survive_normalisation() -> None:
    """``triggers=()`` is a hard barrier, not a missing value.

    ``__post_init__`` must not substitute ``default_triggers`` for it, or an
    entry declared as "never fall through" silently falls through on any error.
    """
    primary = _ScriptedLLM(name="primary", answer="ok")
    chain = LLMFallbackChain(
        entries=[FallbackEntry(model=primary, triggers=()), FallbackEntry(model=primary)],
        default_triggers=("any_error",),
    )
    assert chain.entries[0].triggers == ()
    assert chain.entries[1].triggers == ("any_error",)
    assert not chain.entries[0].matches(Exception("boom"))


async def test_mid_chain_barrier_stops_failover() -> None:
    """A middle entry with ``triggers=()`` must end the chain, not be skipped."""
    tier1 = _ScriptedLLM(name="tier1", raise_exc=RuntimeError("tier1 down"))
    tier2 = _ScriptedLLM(name="tier2", raise_exc=RuntimeError("tier2 down"))
    tier3 = _ScriptedLLM(name="tier3", answer="should never be reached")
    chain = LLMFallbackChain(
        entries=[
            FallbackEntry(model=tier1),
            FallbackEntry(model=tier2, triggers=()),
            FallbackEntry(model=tier3),
        ],
        default_triggers=("any_error",),
    )
    with pytest.raises(RuntimeError, match="tier2 down"):
        await chain.chat([{"role": "user", "content": "hi"}])


def test_standalone_entry_defaults_to_any_error() -> None:
    """An unset entry used outside a chain keeps the permissive default."""
    primary = _ScriptedLLM(name="primary", answer="ok")
    assert FallbackEntry(model=primary).matches(Exception("boom"))


def test_model_name_lists_all_models() -> None:
    primary = _ScriptedLLM(name="primary", answer="ok")
    secondary = _ScriptedLLM(name="secondary", answer="ok")
    chain = LLMFallbackChain.from_models([primary, secondary])
    assert chain.model_name == "llm_fallback_chain[primary,secondary]"


def test_fallback_trigger_typing() -> None:
    """All built-in trigger keywords must be accepted by ``_trigger_matches``."""
    from agent_core.providers.fallback import _trigger_matches

    triggers: list[FallbackTrigger] = ["timeout", "rate_limit", "5xx", "any_error"]
    for trig in triggers:
        # ``any_error`` is the only one that matches a generic Exception.
        assert _trigger_matches(trig, Exception("x")) == (trig == "any_error")


class _KwargsRecordingLLM:
    """Native :class:`LLMClient` mock recording the kwargs reaching ``chat``.

    Lets us assert that ``tools=[...]`` survives the trip through
    ``LLMFallbackChain`` (the chain forwards ``tools`` to each entry's
    ``chat``). Replaces the former langchain ``BaseChatModel`` +
    ``Runnable.bind`` mock — the native chain has no ``bind_tools`` /
    ``RunnableBinding`` indirection; ``tools`` is a first-class ``chat`` kwarg.
    """

    last_kwargs: ClassVar[dict] = {}

    def __init__(self, name: str = "recorder") -> None:
        self.model = name
        self.model_name = name

    async def chat(
        self, messages, *, tools=None, temperature=None,
        max_tokens=None, extra_headers=None, timeout=None,
    ) -> LLMResponse:
        type(self).last_kwargs = {
            k: v for k, v in {
                "tools": tools, "temperature": temperature,
                "max_tokens": max_tokens, "extra_headers": extra_headers,
                "timeout": timeout,
            }.items() if v is not None
        }
        return LLMResponse(content="ok", model=self.model)

    async def stream(self, messages, *, tools=None, **kwargs):
        type(self).last_kwargs = {"tools": tools} if tools is not None else {}
        yield StreamDelta(content="ok")


@pytest.mark.asyncio
async def test_tools_kwarg_reaches_underlying_model() -> None:
    """Regression: ``tools=`` must survive the fallback wrapper.

    Native ``LLMFallbackChain.chat`` forwards ``tools`` to each entry's
    ``chat``; ``with_provider_stamp`` builds a transparent 1-entry chain,
    so a tool-call request through it must still carry the tools. (Replaces
    the old langchain ``bind_tools`` / ``RunnableBinding`` kwargs-merge
    regression — that indirection no longer exists post-de-langchain;
    ``tools`` is a first-class ``chat`` kwarg.)
    """
    inner = _KwargsRecordingLLM()
    chain = with_provider_stamp(inner, "aliyun")
    fake_tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "parameters": {"type": "object"},
            },
        }
    ]

    _KwargsRecordingLLM.last_kwargs = {}
    await chain.chat([{"content": "hello", "role": "user"}], tools=fake_tools)

    assert "tools" in _KwargsRecordingLLM.last_kwargs, (
        f"tools= dropped through LLMFallbackChain; "
        f"saw kwargs={list(_KwargsRecordingLLM.last_kwargs.keys())}"
    )
    assert _KwargsRecordingLLM.last_kwargs["tools"] == fake_tools
