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
import contextlib
import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from agent_core.llm import LLMClient, LLMResponse, StreamDelta
from agent_core.messages import Message, ToolCall, text_of
from agent_core.model_capabilities import ModelCapabilities, resolve_model_capabilities
from agent_core.providers._stream_activity import StreamActivity, stream_events_with_activity
from agent_core.providers.finish_reason import normalize_finish_reason

logger = logging.getLogger(__name__)


class AnthropicClient(LLMClient):
    """Non-streaming-first Anthropic adapter.

    Constructor, chat and stream accept but omit ``temperature`` for compatibility:
    SDK v1 removed sampling parameters and newer models reject non-default values with 400.
    """

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
        capabilities: ModelCapabilities | None = None,
        prompt_cache_ttl: str | None = "",
    ) -> None:
        self.model = model
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self.default_timeout = timeout
        # Extended thinking: when set (e.g. ``{"type": "adaptive", "display":
        # "summarized"}``) the request carries ``thinking=`` so responses return
        # thinking + signature blocks; the response parser keeps them verbatim
        # (content_block) for faithful multi-turn replay. ``effort``
        # (low|medium|high|xhigh|max) → ``output_config.effort`` via extra_body.
        self._thinking = thinking or None
        self._effort = (effort or "").strip()
        # Rejected here rather than per call: a bad value is a config error and
        # should fail at construction, not on the first request of a long run.
        self._prompt_cache_ttl = normalize_prompt_cache_ttl(prompt_cache_ttl)
        self.capabilities = capabilities or resolve_model_capabilities(
            model, protocol="bedrock" if bedrock else "anthropic",
        )
        self.capabilities.validate_request(
            model=model, thinking=self._thinking, effort=self._effort,
            max_tokens=max_tokens,
        )
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
        output_limit = max_tokens or self.default_max_tokens or 4096
        # Thinking and effort are fixed and validated at construction; only
        # the per-call output limit can change here.
        self.capabilities.validate_request(
            model=self.model, thinking=None, max_tokens=output_limit,
        )
        system, msgs = _split_system(messages)
        # ``_to_anthropic_msg`` returns None for a message with nothing
        # sendable (a contentless assistant turn); those are dropped.
        pairs = [
            (converted, bool(m.get("transient")))
            for m in msgs
            if (converted := _to_anthropic_msg(m)) is not None
        ]
        pairs = _merge_tool_results(pairs)
        transient_tail = 0
        for _, is_transient in reversed(pairs):
            if not is_transient:
                break
            transient_tail += 1
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [converted for converted, _ in pairs],
            "max_tokens": output_limit,
        }
        if system:
            kwargs["system"] = system
        if self._thinking:
            kwargs["thinking"] = self._thinking
        # Current Claude models think adaptively even when thinking is omitted.
        # Keep effort independent of that optional display/configuration field.
        if self._effort:
            kwargs["extra_body"] = {"output_config": {"effort": self._effort}}
        if tools:
            kwargs["tools"] = [_to_anthropic_tool(t) for t in tools]
        if extra_headers:
            kwargs["extra_headers"] = extra_headers
        if timeout is not None:
            kwargs["timeout"] = timeout
        elif self.default_timeout is not None:
            kwargs["timeout"] = self.default_timeout
        _add_prompt_cache(
            kwargs, transient_tail=transient_tail, ttl=self._prompt_cache_ttl,
        )
        # After the cache breakpoint is placed, so it stays on the last
        # persistent block rather than moving onto the per-call text.
        kwargs["messages"] = _fold_transient_tail(kwargs["messages"], transient_tail)
        return kwargs

    async def _create_message(self, kwargs: dict[str, Any]) -> tuple[Any, bool]:
        """Retry once when the API rejects a historical thinking signature.

        Fable 5.1 / Opus 5.5 bind thinking to its conversation prefix. Runtime
        compaction, tool filtering, or a changed system prompt can invalidate
        that prefix. Remove all thinking for this request only, leaving durable
        history intact; unrelated 400s and a second rejection propagate.

        Matching is on "signature" + "thinking" rather than one exact wording:
        the phrasing is not a documented contract, and omitting historical
        thinking is always a valid request, so any signature rejection is
        recoverable the same way. A narrower match fails silently on rewording.
        """
        from anthropic import BadRequestError

        try:
            return await self._client.messages.create(**kwargs), False
        except BadRequestError as exc:
            error = str(exc).lower()
            if not ("signature" in error and "thinking" in error):
                raise
            messages, stripped = _without_thinking_blocks(kwargs["messages"])
            if not stripped:
                raise
            logger.warning(
                "Anthropic thinking signature no longer matches the conversation; "
                "retrying once without historical thinking blocks",
            )
            raw = await self._client.messages.create(**{**kwargs, "messages": messages})
            return raw, True

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
        raw, reset = await self._create_message(kwargs)
        response = _to_llm_response(raw)
        if reset:
            response.response_metadata["thinking_history_reset"] = True
        return response

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
        stop_details: dict[str, Any] = {}
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
        stream, reset = await self._create_message(kwargs)
        activity = StreamActivity()
        usage_estimated = False
        async with contextlib.aclosing(stream_events_with_activity(stream, activity)) as events:
            async for event in events:
                if event is None:
                    # Transport activity resets the loop's stall timer without
                    # fabricating text, reasoning, tool calls or observer deltas.
                    yield StreamDelta(transport_activity=True)
                    continue
                etype = getattr(event, "type", "")
                if etype == "message_start":
                    msg = getattr(event, "message", None)
                    model = getattr(msg, "model", "") or model
                    u = getattr(msg, "usage", None)
                    if u is not None:
                        usage_estimated = usage_estimated or bool(getattr(u, "estimated", False))
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
                        # Also keep its position among signed thinking blocks.
                        # Reordering tool calls changes the prefix of later thinking.
                        blocks[idx] = {
                            "type": "tool_use", "id": getattr(cb, "id", "") or "",
                            "name": getattr(cb, "name", "") or "",
                            "input": getattr(cb, "input", {}) or {},
                        }
                        activity.mark_output()
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
                        activity.mark_output()
                        yield StreamDelta(content=chunk)
                    elif dtype == "thinking_delta":
                        chunk = getattr(d, "thinking", "") or ""
                        blk = blocks.get(idx)
                        if blk is not None and blk.get("type") == "thinking":
                            blk["thinking"] += chunk
                        activity.mark_output()
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
                        blk = blocks.get(idx)
                        if blk is not None and blk.get("type") == "tool_use":
                            blk["_partial_json"] = (
                                blk.get("_partial_json", "")
                                + (getattr(d, "partial_json", "") or "")
                            )
                        activity.mark_output()
                        yield StreamDelta(tool_call_deltas=[{
                            "index": idx,
                            "id": None,
                            "name": None,
                            "arguments": getattr(d, "partial_json", "") or "",
                        }])
                elif etype == "message_delta":
                    d = getattr(event, "delta", None)
                    stop_reason = getattr(d, "stop_reason", "") or stop_reason
                    # A classifier refusal ends the stream here, with the detail on
                    # the same delta that carries the stop reason. Captured so the
                    # streamed turn reports it exactly like the non-streaming one.
                    details = _anthropic_stop_details(d)
                    if details is not None:
                        stop_details = details
                    u = getattr(event, "usage", None)
                    if u is not None:
                        usage_estimated = usage_estimated or bool(getattr(u, "estimated", False))
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
        for block in ordered:
            partial = block.pop("_partial_json", None)
            if partial is not None:
                try:
                    block["input"] = json.loads(partial)
                except (ValueError, TypeError):
                    # The tool-call channel retains truncated arguments for
                    # the runtime's repair/replay logic.
                    block["input"] = {}
        activity.mark_output()
        yield StreamDelta(
            usage_source="estimated" if usage_estimated else "provider",
            usage=_anthropic_usage_dict(
                input_tokens,
                output_tokens,
                cache_read,
                cache_write,
                reasoning_tokens,
            ),
            # ``max_tokens`` must reach the runaway/truncation checks as
            # ``length``; other stop reasons pass through untouched.
            finish_reason=normalize_finish_reason(stop_reason),
            model=model,
            reasoning_blocks=ordered if _has_thinking(ordered) else [],
            stop_details=stop_details,
            stop_reason=stop_reason,
            thinking_history_reset=reset,
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


def _without_thinking_blocks(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], bool]:
    out: list[dict[str, Any]] = []
    stripped = False
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            out.append(message)
            continue
        kept = [b for b in content if b.get("type") not in ("thinking", "redacted_thinking")]
        if len(kept) == len(content):
            out.append(message)
            continue
        stripped = True
        if kept:
            out.append({**message, "content": kept})
    return out, stripped


def _merge_tool_results(
    pairs: list[tuple[dict[str, Any], bool]],
) -> list[tuple[dict[str, Any], bool]]:
    """Fold consecutive tool-result-only user messages into one.

    Each OpenAI ``tool`` message converts to its own user message, so a turn
    with parallel calls produces several in a row. Anthropic merges them, but
    the documented shape is ONE user message carrying every ``tool_result``,
    and translating gateways (llm-hub in front of a non-Claude model) reject
    the split form: "An assistant message with 'tool_calls' must be followed
    by tool messages responding to each 'tool_call_id'".
    """
    def only_results(msg: dict[str, Any]) -> bool:
        content = msg.get("content")
        return (
            msg.get("role") == "user"
            and isinstance(content, list)
            and bool(content)
            and all(isinstance(b, dict) and b.get("type") == "tool_result" for b in content)
        )

    out: list[tuple[dict[str, Any], bool]] = []
    for msg, transient in pairs:
        if out and not transient and not out[-1][1] and only_results(msg) and only_results(
            out[-1][0]
        ):
            prev = out[-1][0]
            out[-1] = ({**prev, "content": [*prev["content"], *msg["content"]]}, False)
        else:
            out.append((msg, transient))
    return out


def _fold_transient_tail(msgs: list[dict[str, Any]], transient_tail: int) -> list[dict[str, Any]]:
    """Append trailing per-call user text to the user message before it.

    The runtime addendum (``system_addendum_per_call_role="user"``) arrives as
    one more user message after the tool results. Anthropic accepts the pair,
    but translating gateways do not reliably: llm-hub in front of deepseek-flash
    answered six of 195 requests in one Forge run (2026-10-01) with
    "An assistant message with 'tool_calls' must be followed by tool messages
    responding to each 'tool_call_id'", and the loop then gave up on the
    episode's history. One user message holding the results and then the text
    is the shape every translator maps cleanly.
    """
    if transient_tail <= 0 or len(msgs) <= transient_tail:
        return msgs
    head, tail = msgs[:-transient_tail], msgs[-transient_tail:]
    target = head[-1]
    if target.get("role") != "user" or any(m.get("role") != "user" for m in tail):
        return msgs
    blocks = target.get("content")
    blocks = [{"type": "text", "text": blocks}] if isinstance(blocks, str) else list(blocks or [])
    for m in tail:
        extra = m.get("content")
        if isinstance(extra, str):
            if extra:
                blocks.append({"type": "text", "text": extra})
        elif isinstance(extra, list):
            blocks.extend(extra)
    return [*head[:-1], {**target, "content": blocks}]


def _split_system(messages: list[Message]) -> tuple[str, list[Message]]:
    """Pull out the (single) leading system message; Anthropic takes it
    as a top-level kwarg, not as a message."""
    if messages and messages[0].get("role") == "system":
        return text_of(messages[0].get("content", "")), messages[1:]
    return "", list(messages)


#: Cache lifetimes Anthropic accepts on ``cache_control``. The empty string is
#: this adapter's "not configured", which omits the field and takes the API
#: default of five minutes.
PROMPT_CACHE_TTLS = frozenset({"5m", "1h"})

#: Deployment-wide default when no client was configured with one, so an
#: operator can turn the longer lifetime on for a whole run without a config
#: path reaching every construction site. A client's own value wins.
PROMPT_CACHE_TTL_ENV = "ANTHROPIC_PROMPT_CACHE_TTL"


def normalize_prompt_cache_ttl(value: object) -> str:
    """Normalize a configured cache TTL; ``""`` means "not configured".

    Unknown values raise rather than falling back to the default, matching
    :func:`agent_core.model_capabilities.normalize_thinking_mode`: a typo that
    silently reverts to five minutes is the failure this whole knob exists to
    fix, and it would only surface as a cache-hit-rate regression nobody is
    watching.
    """
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("prompt cache ttl must be a string or null")
    ttl = value.strip().lower()
    if not ttl:
        return ""
    if ttl not in PROMPT_CACHE_TTLS:
        raise ValueError(
            f"unsupported prompt cache ttl {value!r}; use {sorted(PROMPT_CACHE_TTLS)}",
        )
    return ttl


def _cache_control(ttl: str) -> dict[str, str]:
    """The ``cache_control`` value for a breakpoint, with TTL when configured."""
    resolved = ttl or normalize_prompt_cache_ttl(os.getenv(PROMPT_CACHE_TTL_ENV))
    if not resolved or resolved == "5m":
        # Omitted rather than sent explicitly: "5m" is the API default, and not
        # sending the field keeps the request shape of every existing consumer
        # byte-identical.
        return {"type": "ephemeral"}
    return {"type": "ephemeral", "ttl": resolved}


def _add_prompt_cache(
    kwargs: dict[str, Any], *, transient_tail: int = 0, ttl: str = "",
) -> None:
    """Set Anthropic prompt-cache breakpoints on ``kwargs`` in place.

    Anthropic caching is opt-in per content block (unlike OpenAI's automatic
    caching), so without breakpoints the full growing prompt is re-billed every
    turn (``cached_tokens=0``). Place two ``ephemeral`` breakpoints — the system
    prefix (static across the run) and the last message's final block (a rolling
    breakpoint that caches the growing conversation prefix). Anthropic allows up
    to 4 and serves the longest matching cached prefix, so these two cover the
    static head and the moving tail. This lives inside ``AnthropicClient`` so it
    only ever touches Anthropic requests. Disable with ``ANTHROPIC_PROMPT_CACHE=0``.

    ``transient_tail`` counts trailing messages that exist only in this request
    (``Message.transient``, e.g. the per-call runtime addendum). The rolling
    breakpoint skips them: the next request replaces them with the real turn,
    so a cached prefix ending on one never matches again and every turn would
    re-write the whole conversation at the cache-write rate while reading only
    the static head.

    ``ttl`` ("5m" default, or "1h") applies to BOTH breakpoints. A cached entry
    expires that long after its last use, so the lifetime that matters is not
    how long a run takes but how long one TURN takes: a gap longer than the TTL
    loses the whole prefix and re-writes it. On a slow model that is a routine
    event rather than an edge case — in ApodexHarness's 2026-10-05 GDPval batch,
    12.9% of claude-opus-5-5 turn gaps exceeded five minutes (p90 355s) and
    those calls missed the cache 35.7% of the time against 9.5% for the rest,
    re-writing a median 81k tokens each. The trade is the write rate: 2x base
    input for an hour against 1.25x for five minutes, so this pays off exactly
    when turns are slow enough to straddle the shorter window and costs extra
    when they are not. Writes land in ``cache_creation.ephemeral_1h_input_tokens``,
    which :func:`_anthropic_cache_write_tokens` already counts.
    """
    if os.getenv("ANTHROPIC_PROMPT_CACHE", "1") == "0":
        return
    cache_control = _cache_control(ttl)
    # System prefix (a plain string) -> one cache-controlled text block.
    system = kwargs.get("system")
    if isinstance(system, str) and system:
        kwargs["system"] = [{
            "type": "text",
            "text": system,
            "cache_control": cache_control,
        }]
    # Rolling tail: mark the last persistent message's final content block.
    msgs = kwargs.get("messages")
    if not msgs or transient_tail >= len(msgs):
        return
    last = msgs[-1 - transient_tail]
    content = last.get("content")
    if isinstance(content, str):
        if content:
            last["content"] = [{
                "type": "text",
                "text": content,
                "cache_control": cache_control,
            }]
    elif isinstance(content, list) and content and isinstance(content[-1], dict):
        content[-1] = {**content[-1], "cache_control": cache_control}


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
        calls = [
            converted for tc in (m.get("tool_calls") or [])
            if (converted := _to_anthropic_tool_use(tc)) is not None
        ]
        remaining = {call["id"]: call for call in calls}
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
                elif bt == "tool_use":
                    # The canonical tool-call channel remains authoritative
                    # after runtime repair/filtering; native blocks supply order.
                    call = remaining.pop(block.get("id"), None)
                    if call is not None:
                        blocks.append(call)
        else:
            body = text_of(raw or "")
            if body:
                blocks.append({"type": "text", "text": body})
        blocks.extend(remaining.values())
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


def _anthropic_stop_details(raw: Any) -> dict[str, Any] | None:
    """Structured refusal detail off a response, or ``None`` when absent.

    Anthropic populates ``stop_details`` ONLY when ``stop_reason ==
    "refusal"`` — a safety classifier declined the request. That arrives as a
    normal HTTP 200 with empty ``content``, so without this the turn is
    indistinguishable from "the model chose to say nothing": the loop sees no
    text and no tool call, takes its no-tool exit, and the run ends looking
    clean. The 2026-10-05 gdpval triage had to rule a refusal in or out by
    hand for exactly this reason.

    ``category`` is an open set (``cyber``, ``bio``, ``reasoning_extraction``,
    ``frontier_llm``, ``general_harms``, ``None``, …) and models keep adding
    to it, so every field is passed through as-is rather than validated
    against a list this module would have to chase. Returns ``None`` when the
    payload carries nothing, which keeps the key out of ``response_metadata``
    for the overwhelmingly common non-refusal turn.
    """
    details = getattr(raw, "stop_details", None)
    if details is None and isinstance(raw, dict):
        details = raw.get("stop_details")
    if details is None:
        return None
    if isinstance(details, dict):
        out = {k: v for k, v in details.items() if v is not None}
        return out or None
    model_dump = getattr(details, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json", exclude_none=True)
        if isinstance(dumped, dict):
            return dumped or None
    # SDK model object: read the documented fields off it.
    out = {}
    for field in ("type", "category", "explanation"):
        value = getattr(details, field, None)
        if value is not None:
            out[field] = value
    return out or None


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
            blocks_out.append({
                "type": "tool_use", "id": getattr(block, "id", ""),
                "name": getattr(block, "name", ""),
                "input": getattr(block, "input", {}) or {},
            })
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

    if getattr(usage, "estimated", False):
        usage_dict["estimated"] = True

    # ``stop_reason`` is normalised for ``finish_reason`` (``max_tokens`` →
    # ``length``), which is what the loop's truncation checks need. The RAW
    # value is kept alongside it: ``refusal`` survives normalisation today,
    # but a consumer asking "did the provider decline?" should not have to
    # know which markers this function rewrites.
    metadata: dict[str, Any] = {"id": getattr(raw, "id", "")}
    stop_reason_raw = str(getattr(raw, "stop_reason", "") or "")
    if stop_reason_raw:
        metadata["stop_reason"] = stop_reason_raw
    stop_details = _anthropic_stop_details(raw)
    if stop_details is not None:
        metadata["stop_details"] = stop_details

    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        reasoning_content="\n".join(thinking_parts),
        finish_reason=normalize_finish_reason(getattr(raw, "stop_reason", "")),
        model=getattr(raw, "model", "") or "",
        usage=usage_dict,
        usage_source="estimated" if usage_dict.get("estimated") else "provider" if usage_dict else "",
        response_metadata=metadata,
    )


__all__ = ["AnthropicClient"]
