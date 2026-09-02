"""Anthropic LLMClient — wraps :class:`anthropic.AsyncAnthropic`.

Translates between OpenAI Chat Completions message format (used everywhere
else in the runtime) and Anthropic's block-based messages API.

Replaces ``langchain_anthropic.ChatAnthropic``. Anthropic gotchas handled here:

- ``cache_read_input_tokens`` / ``cache_creation_input_tokens`` map to the
  shared cache-read/cache-write usage fields, including Anthropic's optional
  nested 1-hour cache-creation count.
- Thinking-block ``signature``: the *response* parser preserves it on the
  structured ``thinking`` block AND the *outbound* assistant re-send
  (:func:`_to_anthropic_msg`) echoes signed ``thinking`` /
  ``redacted_thinking`` blocks back verbatim, so multi-turn Claude *extended
  thinking* with tool use continues the signed reasoning state. This only
  engages when ``thinking=`` is requested (opt-in via the client ctor / a
  profile ``protocol: anthropic``); with thinking off the branch is inert
  and the assistant re-send is text + tool_use only, as before.
"""

from __future__ import annotations

# pyright: basic, reportPrivateImportUsage=false
import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from agent_core.llm import LLMClient, LLMResponse, StreamDelta
from agent_core.messages import Message, ToolCall, text_of

logger = logging.getLogger(__name__)


class AnthropicClient(LLMClient):
    """Non-streaming-first Anthropic adapter."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = 4096,
        timeout: float | None = 300.0,
        thinking: dict[str, Any] | None = None,
        effort: str = "",
        bedrock: bool = False,
        default_headers: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self.default_timeout = timeout
        # Extended thinking: when set (e.g. ``{"type": "adaptive", "display":
        # "summarized"}``) the request carries ``thinking=`` so responses return
        # thinking + signature blocks; the response parser keeps them verbatim
        # (content_block) for faithful multi-turn replay. ``temperature`` is
        # dropped when thinking is on (Anthropic 400s on the combo). ``effort``
        # (low|medium|high|xhigh|max) → ``output_config.effort`` via extra_body.
        self._thinking = thinking or None
        self._effort = (effort or "").strip()
        # Transport: ``bedrock`` swaps AsyncAnthropic (``/v1/messages`` +
        # ``x-api-key``) for the AWS Bedrock runtime (``/model/{id}/invoke`` +
        # ``anthropic_version`` body stamp) authenticated with a Bedrock API Key
        # (``Authorization: Bearer``) instead of IAM SigV4. Everything downstream
        # (_build_kwargs / _to_llm_response / thinking replay) is transport-
        # agnostic and reused unchanged.
        if bedrock:
            self._client = _build_bedrock_client(
                api_key,
                base_url,
                timeout,
                default_headers,
            )
        else:
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(
                api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0,
                default_headers=default_headers,
            )

    def _build_kwargs(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        max_tokens: int | None,
        extra_headers: dict[str, str] | None,
        timeout: float | None,
    ) -> dict[str, Any]:
        """Shared request-shape builder for :meth:`chat` and :meth:`stream`."""
        system, msgs = _split_system(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            # ``_to_anthropic_msg`` returns None for a message with nothing
            # sendable (a contentless assistant turn); those are dropped.
            "messages": [
                converted
                for converted in (_to_anthropic_msg(m) for m in msgs)
                if converted is not None
            ],
            "max_tokens": max_tokens or self.default_max_tokens or 4096,
        }
        if system:
            kwargs["system"] = system
        if self._thinking:
            # Anthropic rejects ``temperature`` together with thinking, so it is
            # OMITTED here regardless of the configured default. ``effort`` rides
            # on ``extra_body.output_config`` so any value (incl. ``xhigh``)
            # reaches ``messages.create`` without the SDK's stricter validation.
            kwargs["thinking"] = self._thinking
            if self._effort:
                kwargs["extra_body"] = {"output_config": {"effort": self._effort}}
        else:
            eff_temp = temperature if temperature is not None else self.default_temperature
            if eff_temp is not None:
                kwargs["temperature"] = eff_temp
        if tools:
            kwargs["tools"] = [_to_anthropic_tool(t) for t in tools]
        if extra_headers:
            kwargs["extra_headers"] = extra_headers
        if timeout is not None:
            kwargs["timeout"] = timeout
        elif self.default_timeout is not None:
            kwargs["timeout"] = self.default_timeout
        _add_prompt_cache(kwargs)
        return kwargs

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
        kwargs = self._build_kwargs(
            messages, tools=tools, temperature=temperature,
            max_tokens=max_tokens, extra_headers=extra_headers, timeout=timeout,
        )
        raw = await self._client.messages.create(**kwargs)
        return _to_llm_response(raw)

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[StreamDelta]:
        # Real token-by-token streaming over Anthropic's raw event stream.
        # Each event maps to the same ``StreamDelta`` shape the kernel
        # assembler consumes for OpenAI (content / reasoning_content /
        # tool_call_deltas), and the terminal delta carries usage/finish/model
        # just like the OpenAI ``include_usage`` chunk.
        kwargs = self._build_kwargs(
            messages, tools=tools, temperature=temperature,
            max_tokens=max_tokens, extra_headers=extra_headers, timeout=timeout,
        )
        kwargs["stream"] = True
        input_tokens: int | None = None
        output_tokens: int | None = None
        cache_read: int | None = None
        cache_write: int | None = None
        reasoning_tokens: int | None = None
        model = ""
        stop_reason = ""
        # Verbatim block list, in the provider's own emission order, rebuilt
        # from the event stream so the streamed turn replays exactly like the
        # non-streaming one (``_to_llm_response``). Keyed by the stream's block
        # ``index`` while open; ``_ordered_blocks`` flattens it at the end.
        #
        # This exists for extended thinking: a ``thinking`` block's
        # ``signature`` arrives as its own ``signature_delta`` event AFTER the
        # ``thinking_delta`` text, and ``redacted_thinking`` blocks carry their
        # opaque payload on ``content_block_start`` with no deltas at all.
        # Neither can be expressed in the flattened ``reasoning_content``
        # string, so a streamed thinking turn used to yield reasoning that
        # ``thinking_format="content_block"`` could not replay.
        blocks: dict[int, dict[str, Any]] = {}
        stream = await self._client.messages.create(**kwargs)
        async for event in stream:
            etype = getattr(event, "type", "")
            if etype == "message_start":
                msg = getattr(event, "message", None)
                model = getattr(msg, "model", "") or model
                u = getattr(msg, "usage", None)
                if u is not None:
                    input_tokens = getattr(u, "input_tokens", input_tokens)
                    cr = getattr(u, "cache_read_input_tokens", None)
                    if cr is not None:
                        cache_read = cr
                    cw = _anthropic_cache_write_tokens(u)
                    if cw is not None:
                        cache_write = cw
            elif etype == "content_block_start":
                cb = getattr(event, "content_block", None)
                idx = getattr(event, "index", 0)
                cbtype = getattr(cb, "type", "")
                if cbtype == "tool_use":
                    # Open a tool-call slot: id + name set once; arguments
                    # arrive as ``input_json_delta`` partial-JSON fragments.
                    # Deliberately NOT recorded in ``blocks``: tool calls ride
                    # the ``tool_call_deltas`` channel and are re-emitted from
                    # ``Message.tool_calls`` on replay, exactly as the
                    # non-streaming ``_to_llm_response`` does.
                    yield StreamDelta(tool_call_deltas=[{
                        "index": idx,
                        "id": getattr(cb, "id", "") or "",
                        "name": getattr(cb, "name", "") or "",
                        "arguments": "",
                    }])
                elif cbtype == "text":
                    blocks[idx] = {
                        "type": "text",
                        "text": getattr(cb, "text", "") or "",
                    }
                elif cbtype == "thinking":
                    blocks[idx] = {
                        "type": "thinking",
                        "thinking": getattr(cb, "thinking", "") or "",
                        "signature": getattr(cb, "signature", "") or "",
                    }
                elif cbtype == "redacted_thinking":
                    # Whole payload lands on the start event — no deltas follow.
                    blocks[idx] = {
                        "type": "redacted_thinking",
                        "data": getattr(cb, "data", "") or "",
                    }
            elif etype == "content_block_delta":
                d = getattr(event, "delta", None)
                dtype = getattr(d, "type", "")
                idx = getattr(event, "index", 0)
                if dtype == "text_delta":
                    chunk = getattr(d, "text", "") or ""
                    blk = blocks.get(idx)
                    if blk is not None and blk.get("type") == "text":
                        blk["text"] += chunk
                    yield StreamDelta(content=chunk)
                elif dtype == "thinking_delta":
                    chunk = getattr(d, "thinking", "") or ""
                    blk = blocks.get(idx)
                    if blk is not None and blk.get("type") == "thinking":
                        blk["thinking"] += chunk
                    yield StreamDelta(reasoning_content=chunk)
                elif dtype == "signature_delta":
                    # The signature is a single opaque token, not an
                    # incremental text stream, but it is appended rather than
                    # assigned so a provider that ever chunks it still
                    # reassembles correctly.
                    blk = blocks.get(idx)
                    if blk is not None and blk.get("type") == "thinking":
                        blk["signature"] += getattr(d, "signature", "") or ""
                elif dtype == "input_json_delta":
                    yield StreamDelta(tool_call_deltas=[{
                        "index": idx,
                        "id": None,
                        "name": None,
                        "arguments": getattr(d, "partial_json", "") or "",
                    }])
            elif etype == "message_delta":
                d = getattr(event, "delta", None)
                stop_reason = getattr(d, "stop_reason", "") or stop_reason
                u = getattr(event, "usage", None)
                if u is not None:
                    ot = getattr(u, "output_tokens", None)
                    if ot is not None:
                        output_tokens = ot
                    # ``output_tokens_details.thinking_tokens`` when the payload
                    # carries it; without this the streamed path reported no
                    # ``reasoning_tokens`` at all while the non-streaming path
                    # did, splitting one model's thinking spend across two
                    # differently-shaped usage dicts.
                    rt = _anthropic_reasoning_tokens(u)
                    if rt is not None:
                        reasoning_tokens = rt
        # Terminal delta: fold the accumulated usage/finish/model onto the
        # assembled ``LLMResponse`` (mirrors OpenAI's empty-choices chunk).
        # ``reasoning_blocks`` is sent ONLY for a thinking turn — for a plain
        # text turn the flattened ``content`` string is the faithful shape and
        # the assembler should keep using it.
        ordered = _ordered_blocks(blocks)
        yield StreamDelta(
            usage=_anthropic_usage_dict(
                input_tokens,
                output_tokens,
                cache_read,
                cache_write,
                reasoning_tokens,
            ),
            finish_reason=stop_reason,
            model=model,
            reasoning_blocks=ordered if _has_thinking(ordered) else [],
        )


# ── Bedrock transport ────────────────────────────────────────────────────


def _bedrock_region_from_url(base_url: str | None) -> str:
    """Best-effort region from a bedrock-runtime base_url.

    ``https://bedrock-runtime.us-east-1.amazonaws.com`` → ``us-east-1``;
    defaults to ``us-east-1`` when it can't be parsed (the region only labels
    the SDK client — the endpoint is ``base_url`` verbatim)."""
    host = (base_url or "").split("//", 1)[-1].split("/", 1)[0]
    parts = host.split(".")
    if len(parts) >= 3 and parts[0].startswith("bedrock-runtime"):
        return parts[1]
    return "us-east-1"


def _build_bedrock_client(
    api_key: str | None,
    base_url: str | None,
    timeout: float | None,
    default_headers: dict[str, str] | None = None,
):
    """AsyncAnthropicBedrock that authenticates with a Bedrock API Key
    (``Authorization: Bearer``) instead of IAM SigV4.

    Mirrors the proven reporter pattern (``report_llm._build_bedrock_raw``):
    the stock ``AsyncAnthropicBedrock._prepare_request`` SigV4-signs via boto3
    (needs AWS creds); we override it to inject the Bearer header. Everything
    else the Bedrock client gives for free is what we want — the
    ``/v1/messages`` → ``/model/{id}/invoke`` URL rewrite and the
    ``anthropic_version: bedrock-2023-05-31`` body stamp.

    Because the override REPLACES SigV4 outright, a missing key cannot fall back
    to the AWS credential chain the stock method would have consulted — it would
    send a bare ``Authorization: Bearer`` and get a 401 that looks like a bad
    key rather than a bypassed auth path. So the key is resolved from
    ``api_key`` then ``AWS_BEARER_TOKEN_BEDROCK`` (the env var the Bedrock API-key
    flow documents), and a still-empty value raises instead of building a client
    that cannot authenticate."""
    import httpx
    from anthropic import AsyncAnthropicBedrock

    bearer = api_key or os.getenv("AWS_BEARER_TOKEN_BEDROCK", "")
    if not bearer:
        raise ValueError(
            "Bedrock transport needs a Bedrock API key: pass api_key= or set "
            "AWS_BEARER_TOKEN_BEDROCK. This client overrides SigV4 request "
            "signing with Bearer auth, so ambient AWS credentials are NOT used.",
        )

    class _BearerBedrock(AsyncAnthropicBedrock):
        async def _prepare_request(self, request: httpx.Request) -> None:
            request.headers["Authorization"] = f"Bearer {bearer}"

    return _BearerBedrock(
        aws_region=_bedrock_region_from_url(base_url),
        base_url=base_url or None,
        timeout=timeout,
        max_retries=0,
        default_headers=default_headers,
    )


# ── Conversion helpers ───────────────────────────────────────────────────


def _split_system(messages: list[Message]) -> tuple[str, list[Message]]:
    """Pull out the (single) leading system message; Anthropic takes it
    as a top-level kwarg, not as a message."""
    if messages and messages[0].get("role") == "system":
        return text_of(messages[0].get("content", "")), messages[1:]
    return "", list(messages)


def _add_prompt_cache(kwargs: dict[str, Any]) -> None:
    """Set Anthropic prompt-cache breakpoints on ``kwargs`` in place.

    Anthropic caching is opt-in per content block (unlike OpenAI's automatic
    caching), so without breakpoints the full growing prompt is re-billed every
    turn (``cached_tokens=0``). Place two ``ephemeral`` breakpoints — the system
    prefix (static across the run) and the last message's final block (a rolling
    breakpoint that caches the growing conversation prefix). Anthropic allows up
    to 4 and serves the longest matching cached prefix, so these two cover the
    static head and the moving tail. This lives inside ``AnthropicClient`` so it
    only ever touches Anthropic requests. Disable with ``ANTHROPIC_PROMPT_CACHE=0``.
    """
    if os.getenv("ANTHROPIC_PROMPT_CACHE", "1") == "0":
        return
    # System prefix (a plain string) -> one cache-controlled text block.
    system = kwargs.get("system")
    if isinstance(system, str) and system:
        kwargs["system"] = [{
            "type": "text",
            "text": system,
            "cache_control": {"type": "ephemeral"},
        }]
    # Rolling tail: mark the last message's final content block.
    msgs = kwargs.get("messages")
    if not msgs:
        return
    last = msgs[-1]
    content = last.get("content")
    if isinstance(content, str):
        if content:
            last["content"] = [{
                "type": "text",
                "text": content,
                "cache_control": {"type": "ephemeral"},
            }]
    elif isinstance(content, list) and content and isinstance(content[-1], dict):
        content[-1] = {**content[-1], "cache_control": {"type": "ephemeral"}}


def _to_anthropic_msg(m: Message) -> dict[str, Any] | None:
    role = m.get("role")
    if role == "tool":
        return {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id", ""),
                "content": text_of(m.get("content", "")),
            }],
        }
    if role == "assistant":
        blocks: list[dict[str, Any]] = []
        raw = m.get("content")
        if isinstance(raw, list):
            # Extended-thinking continuation: history kept the VERBATIM block
            # list (via model_profile.to_history for content_block). Re-send the
            # signed ``thinking`` / ``redacted_thinking`` blocks UNMODIFIED so
            # Anthropic can validate the signature server-side and continue the
            # signed reasoning state, then append the visible text + tool_use.
            for block in raw:
                if not isinstance(block, dict):
                    continue
                bt = block.get("type")
                if bt == "thinking":
                    tb: dict[str, Any] = {
                        "type": "thinking",
                        "thinking": block.get("thinking", "") or "",
                    }
                    sig = block.get("signature")
                    if sig:
                        tb["signature"] = sig
                    blocks.append(tb)
                elif bt == "redacted_thinking":
                    blocks.append({
                        "type": "redacted_thinking",
                        "data": block.get("data", "") or "",
                    })
                elif bt == "text":
                    txt = block.get("text", "") or ""
                    if txt:
                        blocks.append({"type": "text", "text": txt})
        else:
            body = text_of(raw or "")
            if body:
                blocks.append({"type": "text", "text": body})
        for tc in m.get("tool_calls", []) or []:
            block = _to_anthropic_tool_use(tc)
            if block is not None:
                blocks.append(block)
        if not blocks:
            # Nothing to say and nothing to call. An empty ``text`` block is
            # NOT a usable placeholder — Anthropic rejects zero-length text
            # ("text content blocks must be non-empty"), and once such a turn
            # lands in durable history EVERY later request replaying it fails
            # in conversion, before a request is even sent. Whitespace-only is
            # no safer: as the final message it trips the trailing-whitespace
            # check instead. Reachable in practice from
            # ``finish_reason="length"`` with empty content, which
            # ``_to_llm_response`` maps to ``content=""``.
            #
            # Dropping the turn is the faithful shape — there was no assistant
            # output to replay — and it is safe because the Messages API
            # combines consecutive same-role messages rather than requiring
            # strict alternation. The caller filters the ``None``.
            logger.debug(
                "dropping contentless assistant message from Anthropic request",
            )
            return None
        return {"role": "assistant", "content": blocks}
    return {"role": "user", "content": text_of(m.get("content", ""))}


def _to_anthropic_tool_use(tc: Any) -> dict[str, Any] | None:
    """One OpenAI ``tool_calls`` entry → an Anthropic ``tool_use`` block.

    Returns ``None`` for a call this conversion cannot express, rather than
    raising. Everything here runs while REPLAYING durable history, so an
    exception is not a one-request failure: the malformed call is already
    recorded, and every subsequent turn in the session re-converts it and dies
    the same way. A partial or truncated tool call must degrade, not wedge the
    session.

    - Missing ``id`` or function ``name`` → dropped. Anthropic requires both,
      and a call with no id can have no matching ``tool_result`` to orphan.
    - Unparseable or non-object ``arguments`` → ``{}``. Anthropic's ``input``
      must be a JSON object, and on replay the arguments are historical detail
      (the tool already ran); the block's *identity* is what the following
      ``tool_result`` needs to validate against. The OpenAI path passes the raw
      string through, so parity here means surviving the same input.
    """
    if not isinstance(tc, dict):
        logger.warning("skipping non-dict tool_call in Anthropic conversion")
        return None
    fn = tc.get("function")
    fn = fn if isinstance(fn, dict) else {}
    call_id = tc.get("id") or ""
    name = fn.get("name") or ""
    if not call_id or not name:
        logger.warning(
            "skipping malformed tool_call in Anthropic conversion "
            "(id=%r, name=%r)", call_id, name,
        )
        return None
    raw_args = fn.get("arguments") or "{}"
    try:
        parsed = json.loads(raw_args)
    except (TypeError, ValueError):
        logger.warning(
            "tool_call %s (%s) has unparseable arguments; replaying with an "
            "empty input object", call_id, name,
        )
        parsed = {}
    if not isinstance(parsed, dict):
        logger.warning(
            "tool_call %s (%s) arguments parsed to %s, not an object; "
            "replaying with an empty input object",
            call_id, name, type(parsed).__name__,
        )
        parsed = {}
    return {"type": "tool_use", "id": call_id, "name": name, "input": parsed}


def _to_anthropic_tool(t: dict[str, Any]) -> dict[str, Any]:
    """OpenAI ``{type:function, function:{name,description,parameters}}`` →
    Anthropic ``{name, description, input_schema}``."""
    fn = t.get("function") or t
    return {
        "name": fn.get("name", ""),
        "description": fn.get("description", ""),
        "input_schema": fn.get("parameters", {}),
    }


def _anthropic_usage_dict(
    input_tokens: int | None,
    output_tokens: int | None,
    cache_read: int | None,
    cache_write: int | None,
    reasoning: int | None = None,
) -> dict[str, int]:
    """Normalise Anthropic token counts into the wire-shape usage dict shared
    by the non-streaming ``_to_llm_response`` and the streaming assembler.
    Cache reads and writes are kept separate for billing and also summed into
    the backward-compatible ``cached_tokens`` field. ``reasoning``
    (extended-thinking tokens, part of ``output_tokens``) is surfaced
    separately when the payload reported it — ``None`` means "not reported" and
    omits the key, while ``0`` is recorded as a real zero."""
    out: dict[str, int] = {}
    if input_tokens is not None:
        out["prompt_tokens"] = int(input_tokens)
    if output_tokens is not None:
        out["completion_tokens"] = int(output_tokens)
    if cache_read is not None or cache_write is not None:
        read = int(cache_read or 0)
        write = int(cache_write or 0)
        out["cache_read_tokens"] = read
        out["cache_write_tokens"] = write
        out["cached_tokens"] = read + write
        out["cache_creation_tokens"] = write
    if reasoning is not None:
        out["reasoning_tokens"] = int(reasoning)
    if out.get("prompt_tokens") or out.get("completion_tokens"):
        out["total_tokens"] = (
            out.get("prompt_tokens", 0) + out.get("completion_tokens", 0)
        )
    return out


def _anthropic_cache_write_tokens(usage: Any) -> int | None:
    """Return Anthropic cache-creation tokens, including the 1-hour extension."""
    if usage is None:
        return None
    raw = getattr(usage, "cache_creation_input_tokens", None)
    if raw is None and isinstance(usage, dict):
        raw = usage.get("cache_creation_input_tokens")
    nested = getattr(usage, "cache_creation", None)
    if nested is None and isinstance(usage, dict):
        nested = usage.get("cache_creation")
    extension = getattr(nested, "ephemeral_1h_input_tokens", None)
    if extension is None and isinstance(nested, dict):
        extension = nested.get("ephemeral_1h_input_tokens")
    if raw is None and extension is None:
        return None
    return max(0, int(raw or 0)) + max(0, int(extension or 0))


def _anthropic_reasoning_tokens(usage: Any) -> int | None:
    """Best-effort extended-thinking token count off an Anthropic usage object.

    Newer usage payloads may expose ``output_tokens_details.thinking_tokens``;
    absent that the count is folded into ``output_tokens`` and unrecoverable.

    Returns ``None`` for "the payload didn't say", distinct from ``0`` for "the
    payload said zero" — the ``reasoning_tokens`` key is omitted only in the
    former case. Collapsing the two would make a model that genuinely spent no
    thinking tokens indistinguishable from a gateway that reports nothing, which
    is the same ambiguity ``openai_chat._usage_dict`` and
    ``_responses_usage_dict`` avoid with their own ``is not None`` checks."""
    if usage is None:
        return None
    otd = getattr(usage, "output_tokens_details", None)
    if otd is None:
        return None
    val = getattr(otd, "thinking_tokens", None)
    if val is None and isinstance(otd, dict):
        val = otd.get("thinking_tokens")
    return None if val is None else int(val)


def _ordered_blocks(blocks: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten the streamed block accumulator back into emission order.

    Anthropic numbers content blocks with a monotonically increasing ``index``
    per message, so sorting by key restores the order the provider produced —
    which is the order signed thinking must be replayed in."""
    return [blocks[i] for i in sorted(blocks)]


def _has_thinking(blocks: list[dict[str, Any]]) -> bool:
    """Whether a block list carries reasoning that needs verbatim replay.

    Mirrors ``_to_llm_response``'s ``thinking_parts or has_redacted`` gate: the
    structured block list is only worth carrying when there is a signature or
    an opaque redacted payload to preserve."""
    return any(
        b.get("type") in ("thinking", "redacted_thinking") for b in blocks
    )


def _to_llm_response(raw: Any) -> LLMResponse:
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    has_redacted = False
    blocks_out: list[dict[str, Any]] = []
    tool_calls: list[ToolCall] = []
    for block in (getattr(raw, "content", None) or []):
        btype = getattr(block, "type", None)
        if btype == "text":
            text = getattr(block, "text", "") or ""
            text_parts.append(text)
            blocks_out.append({"type": "text", "text": text})
        elif btype == "thinking":
            thinking = getattr(block, "thinking", "") or ""
            thinking_parts.append(thinking)
            blocks_out.append({
                "type": "thinking",
                "thinking": thinking,
                # ``signature`` is the cryptographic token Anthropic returns
                # with each thinking block; resending it on the next turn
                # lets the model continue from the same reasoning state.
                "signature": getattr(block, "signature", "") or "",
            })
        elif btype == "redacted_thinking":
            # Encrypted thinking Anthropic chose not to surface. It carries no
            # readable text but MUST be preserved verbatim (raw_content_blocks)
            # and replayed unmodified — the outbound ``_to_anthropic_msg`` echoes
            # it, and dropping it here would break signature/replay continuity.
            has_redacted = True
            blocks_out.append({
                "type": "redacted_thinking",
                "data": getattr(block, "data", "") or "",
            })
        elif btype == "tool_use":
            tool_calls.append({
                "id": getattr(block, "id", ""),
                "type": "function",
                "function": {
                    "name": getattr(block, "name", ""),
                    "arguments": json.dumps(getattr(block, "input", {}) or {}, ensure_ascii=False),
                },
            })

    # When thinking is present (readable or redacted), keep the structured block
    # list so the ``content_block`` parser picks out reasoning vs visible text
    # AND the verbatim signed/redacted blocks survive for replay. Otherwise
    # flatten to a string for the simpler downstream path.
    if thinking_parts or has_redacted:
        content: Any = blocks_out
    else:
        content = "\n".join(text_parts)

    usage = getattr(raw, "usage", None)
    usage_dict = _anthropic_usage_dict(
        getattr(usage, "input_tokens", None) if usage else None,
        getattr(usage, "output_tokens", None) if usage else None,
        getattr(usage, "cache_read_input_tokens", None) if usage else None,
        _anthropic_cache_write_tokens(usage),
        _anthropic_reasoning_tokens(usage),
    )

    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        reasoning_content="\n".join(thinking_parts),
        finish_reason=getattr(raw, "stop_reason", "") or "",
        model=getattr(raw, "model", "") or "",
        usage=usage_dict,
        response_metadata={"id": getattr(raw, "id", "")},
    )


__all__ = ["AnthropicClient"]
