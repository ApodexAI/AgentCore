"""LLM client contracts — provider-agnostic chat completion interface."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from agent_core.messages import Message, ToolCall


@dataclass
class LLMResponse:
    """One non-streaming completion result."""

    content: Any = ""  # str | list[dict]
    tool_calls: list[ToolCall] = field(default_factory=list[ToolCall])
    reasoning_content: str = ""
    finish_reason: str = ""
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict[str, int])
    response_metadata: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass
class StreamDelta:
    """Incremental update during a streaming completion."""

    content: str = ""
    reasoning_content: str = ""
    tool_call_deltas: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    # Terminal metadata. Providers send these late in the stream — usage on a
    # separate ``choices=[]`` chunk (OpenAI ``include_usage``), finish_reason on
    # the last content chunk. Carried here so the stream assembler can put them
    # on the final ``LLMResponse`` (else streaming usage/billing reads 0 and
    # ``finish_reason="length"`` is invisible to truncation/rollback observers).
    usage: dict[str, int] = field(default_factory=dict[str, int])
    finish_reason: str = ""
    model: str = ""
    # Vendor label of the leg serving this stream, stamped by a product's
    # provider-chain wrapper (constant once the chain commits to an entry —
    # failover only fires before the first yield). The stream
    # assembler folds it into ``LLMResponse.response_metadata`` so per-call
    # billing attribution works for streamed calls too — without this the
    # streaming path had no channel for the provider and every billing
    # consumer read an empty vendor, which split one model's usage across
    # a ``provider=""`` bucket and a named bucket.
    provider: str = ""
    # VERBATIM provider content blocks for the assembled turn, carried only by
    # providers whose replay contract needs the structured block list rather
    # than flattened text — currently Anthropic extended thinking, where each
    # ``thinking`` block's cryptographic ``signature`` (and any opaque
    # ``redacted_thinking`` payload) MUST be echoed back unmodified on the next
    # turn or the model cannot continue the signed reasoning state.
    #
    # ``content``/``reasoning_content`` above are the flattened text channels;
    # they cannot represent a signature, so without this field the streaming
    # path silently produced reasoning that ``thinking_format="content_block"``
    # could not replay. Providers that need no block list leave it empty and
    # the stream assembler keeps using the flattened string.
    reasoning_blocks: list[dict[str, Any]] = field(
        default_factory=list[dict[str, Any]],
    )


@runtime_checkable
class LLMClient(Protocol):
    """Minimal async chat completion client."""

    # Stays a settable attribute: LLMClient is not purely structural — concrete
    # clients such as OpenAIClient subclass it and assign ``self.model`` in
    # __init__, so a read-only property here would break them at runtime.
    model: str

    async def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> LLMResponse:
        """Send one non-streaming completion request. ``tools`` is a list
        of OpenAI function-schema dicts; ``None`` runs without tools."""
        ...

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[StreamDelta]:
        """Stream a completion as a sequence of incremental ``StreamDelta``s.

        Terminal metadata is carried by late deltas; the consuming runtime is
        responsible for assembling those deltas into its final response.
        """
        ...


__all__ = ["LLMClient", "LLMResponse", "StreamDelta"]
