"""``AnthropicPromptCacheAdapter`` unit tests.

The adapter wraps a Claude-family :class:`~agent_core.llm.LLMClient`
to inject ``cache_control: ephemeral`` on the system prompt before
delegating to the inner client's :meth:`chat` / :meth:`stream`. The
trickiest part of that wiring is :meth:`bind_tools`: it must keep the
cache adapter in front of a *tool-bound* inner so the reporter loop's
tool-augmented call still goes out with both the cache-marked system
prompt and the bound ``tools=[submit_report, …]`` definitions. If the
adapter dropped the bound tools, the model could never call
``submit_report`` and the agent could never terminate.

Native substrate note: the old LangChain implementation wrapped a
``BaseChatModel`` and had to defeat ``RunnableBinding.__getattr__``
silently dropping bound kwargs on the ``_generate`` / ``_agenerate``
delegation path. The native ``LLMClient`` carries no such binding —
``bind_tools(llm, tools)`` threads ``tools`` straight into
:meth:`LLMClient.chat` — so the contract here is simply: bound tools
must reach the inner client's ``chat``.
"""
from __future__ import annotations

import asyncio
from typing import Any

from agent_core.llm import LLMResponse
from agent_core.messages import system_msg, user_msg
from agent_core.providers.prompt_cache import (
    AnthropicPromptCacheAdapter,
    maybe_wrap_for_prompt_cache,
)


class _RecordingLLM:
    """A minimal native ``LLMClient`` that records every ``chat`` /
    ``stream`` call's messages + kwargs."""

    model: str = "recorder"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> LLMResponse:
        self.calls.append({
            "messages": list(messages),
            "kwargs": {
                "tools": tools,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "extra_headers": extra_headers,
                "timeout": timeout,
            },
        })
        return LLMResponse(content="ok")

    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        self.calls.append({
            "messages": list(messages),
            "kwargs": {
                "tools": tools,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "extra_headers": extra_headers,
                "timeout": timeout,
            },
        })
        if False:  # pragma: no cover - make this an async generator
            yield None


def test_cache_control_injected_on_system_message() -> None:
    """Sanity: system prompt gets ``cache_control: ephemeral`` added."""
    inner = _RecordingLLM()
    adapter = AnthropicPromptCacheAdapter(inner=inner)

    messages = [system_msg("be helpful"), user_msg("hi")]
    asyncio.run(adapter.chat(messages))

    assert len(inner.calls) == 1
    sys_msg = inner.calls[0]["messages"][0]
    assert sys_msg.get("role") == "system"
    assert isinstance(sys_msg["content"], list)
    assert sys_msg["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_bound_tool_kwargs_survive_after_bind_tools() -> None:
    """Regression: ``bind_tools`` must keep the cache adapter in front of a
    tool-bound inner so the reporter loop's ``tools=[submit_report, …]``
    still reaches the upstream Claude client. Without this the request goes
    out with no tools and the agent can never terminate.
    """
    inner = _RecordingLLM()
    adapter = AnthropicPromptCacheAdapter(inner=inner)

    tool_spec = [{
        "type": "function",
        "function": {
            "name": "submit_report",
            "description": "x",
            "parameters": {"type": "object", "properties": {}},
        },
    }]
    bound = adapter.bind_tools(tool_spec)

    # Self-check: bind_tools keeps the cache adapter on the outside.
    assert isinstance(bound, AnthropicPromptCacheAdapter)

    asyncio.run(bound.chat([
        system_msg("be helpful"),
        user_msg("hi"),
    ]))

    assert len(inner.calls) == 1
    kwargs = inner.calls[0]["kwargs"]
    assert kwargs.get("tools") == tool_spec, (
        "bound tools must be forwarded to the underlying chat model; "
        "without this the reporter loop sends requests with no tools "
        "and the model can never call submit_report."
    )
    # The cache adapter still injects cache_control on top of the tools.
    sys_msg = inner.calls[0]["messages"][0]
    assert isinstance(sys_msg["content"], list)
    assert sys_msg["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_caller_tools_override_bound_tools() -> None:
    """When the caller passes ``tools`` directly it wins over the bound
    default — matches the native ``chat(..., tools=...)`` precedence so
    callers that override at the call site behave the same with or without
    the cache adapter in the chain.
    """
    inner = _RecordingLLM()
    adapter = AnthropicPromptCacheAdapter(inner=inner)
    bound = adapter.bind_tools([{
        "type": "function",
        "function": {
            "name": "x",
            "description": "y",
            "parameters": {"type": "object", "properties": {}},
        },
    }])

    override = [{
        "type": "function",
        "function": {"name": "z", "description": "w", "parameters": {}},
    }]
    asyncio.run(bound.chat([user_msg("hi")], tools=override))

    assert inner.calls[0]["kwargs"]["tools"] == override


def test_maybe_wrap_short_circuits_for_non_claude() -> None:
    inner = _RecordingLLM()
    result = maybe_wrap_for_prompt_cache(inner, provider="openai", model="gpt-4o")
    assert result is inner


def test_maybe_wrap_wraps_claude_family() -> None:
    inner = _RecordingLLM()
    result = maybe_wrap_for_prompt_cache(
        inner, provider="openrouter", model="anthropic/claude-sonnet-4-5",
    )
    assert isinstance(result, AnthropicPromptCacheAdapter)
    assert result.inner is inner
