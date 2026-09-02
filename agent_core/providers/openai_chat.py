"""OpenAI-compatible LLMClient — wraps :class:`openai.AsyncOpenAI`.

Works against the OpenAI API and any OpenAI-Chat-Completions-compatible
endpoint (vLLM, SGLang, OpenRouter, etc.). For Anthropic,
use ``infra.anthropic_client.AnthropicClient`` (separate file, separate SDK).

Replaces ``langchain_openai.ChatOpenAI``. The wire-shape choices below
(``stream:false`` explicit, ``with_raw_response.create().parse()``, per-call
``timeout`` only when explicitly passed) are byte-aligned with LangChain's
ChatOpenAI request — see migration gotchas #3/#4. Do not "simplify" them.
"""

from __future__ import annotations

# pyright: basic, reportPrivateImportUsage=false
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

from openai import AsyncOpenAI, BadRequestError

from agent_core.llm import LLMClient, LLMResponse, StreamDelta
from agent_core.messages import Message, ToolCall, for_wire
from agent_core.runtime.llm_request_overrides import (
    current_thinking_retry_override,
)

type SessionQueryResolver = Callable[[dict[str, str] | None], dict[str, str]]
type SessionScopeResolver = Callable[[], str]


def _no_session_query(_headers: dict[str, str] | None) -> dict[str, str]:
    return {}


_session_query_resolver: SessionQueryResolver = _no_session_query
_session_scope_resolver: SessionScopeResolver | None = None


def configure_session_query_resolver(resolver: SessionQueryResolver | None) -> None:
    """Configure the host affinity adapter used by subsequently built clients."""
    global _session_query_resolver
    _session_query_resolver = resolver or _no_session_query


def configure_session_scope_resolver(resolver: SessionScopeResolver | None) -> None:
    """Configure the host task scope used to reject stale cached affinity."""
    global _session_scope_resolver
    _session_scope_resolver = resolver

logger = logging.getLogger(__name__)

# Stands in for a missing/empty tool_call function name. Mirrors the platform
# llm-worker's sanitize (packages/llm-worker/internal/worker/sanitize.go): the
# EAS "LLM intelligent routing" gateway 400s deterministically ("can only
# concatenate str (not \"NoneType\") to str") on an assistant history message
# whose tool_call carries an empty function.name, while the same bytes sent
# straight to the backend return 200. Models do emit such tool_calls now and
# then and the loop replays history verbatim, so we repair on the way out.
_PLACEHOLDER_TOOL_NAME = "unknown_tool"


def _sanitize_tool_call_names(messages: list[Message]) -> list[Message]:
    """Rewrite empty tool_call function names before the wire (copy-on-write).

    Rewrite — never drop: the paired ``role:"tool"`` result points at this
    call's ``tool_call_id``, and an orphaned tool result trips a *different*
    chat-template failure upstream. A meaningless name costs the model a
    little context; a broken message chain costs the whole request — and once
    the malformed call is in history, every later turn of the task fails.

    Conservative and non-mutating, in the style of the workflow-level
    normalizers: anything that doesn't parse as the expected shape is left
    alone, and the input list itself is returned when nothing needed fixing.
    """
    out: list[Message] | None = None
    fixed = 0
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        calls = msg.get("tool_calls")
        if not isinstance(calls, list) or not calls:
            continue
        new_calls: list[Any] | None = None
        for j, tc in enumerate(calls):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function")
            if not isinstance(fn, dict):
                continue
            name = fn.get("name")
            if isinstance(name, str) and name:
                continue
            if new_calls is None:
                new_calls = list(calls)
            new_calls[j] = {**tc, "function": {**fn, "name": _PLACEHOLDER_TOOL_NAME}}
            fixed += 1
        if new_calls is not None:
            if out is None:
                out = list(messages)
            out[i] = {**msg, "tool_calls": new_calls}
    if out is not None:
        logger.warning(
            "repaired %d tool_call(s) with an empty function.name before send "
            "— the upstream LLM router 400s on these",
            fixed,
        )
        return out
    return messages


def _unsupported_reasoning_effort(exc: Exception) -> bool:
    """Whether ``exc`` is a gateway rejecting the requested reasoning effort."""
    message = str(exc).lower()
    return "reasoning_effort" in message and (
        "unsupported" in message or "not support" in message
    )


def _demoted_reasoning_effort(model: str | None, current: Any) -> str | None:
    """Return the next wire effort to try, or ``None`` when there is none.

    The gpt-5 family advertises ``none`` but a given gateway may still reject
    it, and ``low`` is the next enum those models do accept (``minimal`` is
    not in gpt-5.1/5.5's set). Every other model/effort pair stays
    single-attempt. ``workflows/heavy_mode/reporter_v2/report_llm.py`` runs
    the same policy inside the heavy reporter's own adapter, scoped there to
    the model that reporter ships with.
    """
    value = str(current or "").strip().lower()
    normalized = (model or "").strip().lower().rsplit("/", 1)[-1]
    if normalized.startswith("gpt-5") and value == "none":
        return "low"
    return None


class OpenAIClient(LLMClient):
    """Thin async wrapper. Honors the standard OpenAI knobs (``temperature``,
    ``max_completion_tokens``, ``tools``, ``stream``); per-call ``extra_headers``
    are forwarded for proxies needing custom auth or session affinity."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        timeout: float | None = None,
        default_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        session_query_resolver: SessionQueryResolver | None = None,
        session_scope_resolver: SessionScopeResolver | None = None,
        sdk_default_query: bool = True,
    ) -> None:
        self.model = model
        self.default_temperature = temperature
        self.default_max_tokens = max_completion_tokens
        self.default_timeout = timeout
        self.extra_body = extra_body or {}
        self._session_query_resolver = session_query_resolver or _session_query_resolver
        self._session_scope_resolver = (
            session_scope_resolver or _session_scope_resolver
        )
        self._default_session_scope = (
            self._session_scope_resolver()
            if self._session_scope_resolver is not None
            else ""
        )
        self._default_session_query = self._session_query_resolver(default_headers)
        # Some OpenAI-compatible gateways reject ``stream_options`` with a 400.
        # We probe optimistically and flip this off on first rejection so the
        # stream still runs (streaming usage then reads 0 for that gateway).
        self._stream_options_supported = True
        # Per-call timeout is enforced by the agent loop's
        # ``asyncio.wait_for`` wrapper, not by the SDK.
        #
        # EAS UCH affinity hashes the URL *query parameter*, not the header
        # (see ``mirror_session_query``) — mirror a construction-time session
        # header into ``default_query`` so it rides every request's URL.
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url or None,
            timeout=timeout,
            default_headers=default_headers,
            # A task-derived query cannot live in the SDK defaults: cached
            # clients may outlive the task that constructed them. Static
            # affinity (no construction scope) remains safe to install here.
            default_query=(
                self._default_session_query or None
                if sdk_default_query and not self._default_session_scope
                else None
            ),
            max_retries=0,           # retries are owned by the runtime loop
        )

    def _effective_extra_body(self) -> dict[str, Any]:
        """Return the profile body plus this task's one-attempt override.

        Only mutate thinking knobs the profile explicitly configured.  This
        keeps strict OpenAI endpoints from receiving SGLang-only fields while
        letting Qwen/Apodex retries lower or disable thinking without changing
        the cached client shared by concurrent tasks.
        """
        body = dict(self.extra_body)
        override = current_thinking_retry_override()
        if override is None:
            return body

        configured_template = body.get("chat_template_kwargs")
        if isinstance(configured_template, dict):
            template = dict(configured_template)
            if override.disabled:
                template["enable_thinking"] = False
                # ``preserve_thinking`` controls how prior/interleaved
                # reasoning is rendered, not whether this generation may
                # think.  Keep the profile value while hard-disabling only
                # the current generation.
                template.pop("thinking_budget", None)
            elif template.get("enable_thinking") is True:
                if override.thinking_budget is not None:
                    template["thinking_budget"] = int(
                        override.thinking_budget,
                    )
            body["chat_template_kwargs"] = template

        # Some OpenAI-compatible reasoning proxies use a top-level request
        # body field instead of SGLang chat-template kwargs.  Override it only
        # when the profile already opted into that dialect.
        if "reasoning_effort" in body and override.reasoning_effort:
            body["reasoning_effort"] = override.reasoning_effort

        return body

    def _session_query(self, extra_headers: dict[str, str] | None) -> dict[str, str]:
        """Resolve per-call affinity, falling back to construction-time affinity."""
        per_call = self._session_query_resolver(extra_headers)
        if per_call:
            return per_call
        fallback = self._default_session_query
        if (
            fallback
            and self._default_session_scope
            and self._session_scope_resolver is not None
            and self._session_scope_resolver() != self._default_session_scope
        ):
            logger.debug(
                "Dropping stale session-affinity query built for scope %r",
                self._default_session_scope,
            )
            return {}
        return fallback

    def _demote_reasoning_effort(self, exc: Exception) -> bool:
        """Sticky one-way downgrade after a rejected ``reasoning_effort``.

        Same self-healing shape as ``_stream_options_supported``: some
        OpenAI-compatible gateways advertise an effort enum they then 400 on,
        and without this the caller's only signal is a hard request failure.
        Flipping ``self.extra_body`` means the whole client pays the probe once
        instead of on every call. Returns whether the caller should retry.
        """
        if not _unsupported_reasoning_effort(exc):
            return False
        current = self.extra_body.get("reasoning_effort")
        demoted = _demoted_reasoning_effort(self.model, current)
        if demoted is None:
            return False
        self.extra_body = {**self.extra_body, "reasoning_effort": demoted}
        logger.warning(
            "Gateway rejected reasoning_effort=%s for %s; falling back to %s "
            "for this client. %s",
            current, self.model, demoted, exc,
        )
        return True

    # ── Non-streaming ────────────────────────────────────────────────────

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
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": for_wire(_sanitize_tool_call_names(list(messages))),
            "stream": False,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["parallel_tool_calls"] = True
        eff_temp = temperature if temperature is not None else self.default_temperature
        if eff_temp is not None:
            kwargs["temperature"] = eff_temp
        eff_mt = max_tokens if max_tokens is not None else self.default_max_tokens
        if eff_mt is not None:
            kwargs["max_completion_tokens"] = eff_mt
        if extra_headers:
            kwargs["extra_headers"] = extra_headers
            # EAS UCH affinity keys on the URL query parameter (the header
            # alone never pins an upstream replica) — mirror the per-call
            # session id into ``extra_query``; per-call wins over any
            # construction-time ``default_query``.
            session_query = self._session_query(extra_headers)
            if session_query:
                kwargs["extra_query"] = session_query
        extra_body = self._effective_extra_body()
        if extra_body:
            kwargs["extra_body"] = extra_body
        # Per-call ``timeout`` triggers ``x-stainless-read-timeout``
        # header; only set when the caller explicitly passed one.
        if timeout is not None:
            kwargs["timeout"] = timeout

        raw_response = await self._create(kwargs)
        raw = raw_response.parse()
        return _to_llm_response(raw)

    async def _create(self, kwargs: dict[str, Any]) -> Any:
        """Open a non-streaming completion, retrying once past a rejected effort."""
        create = self._client.chat.completions.with_raw_response.create
        try:
            return await create(**kwargs)
        except Exception as exc:
            if not self._demote_reasoning_effort(exc):
                raise
        return await create(**{**kwargs, "extra_body": self._effective_extra_body()})

    # ── Streaming ────────────────────────────────────────────────────────

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
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": for_wire(_sanitize_tool_call_names(list(messages))),
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["parallel_tool_calls"] = True
        eff_temp = temperature if temperature is not None else self.default_temperature
        if eff_temp is not None:
            kwargs["temperature"] = eff_temp
        eff_mt = max_tokens if max_tokens is not None else self.default_max_tokens
        if eff_mt is not None:
            kwargs["max_completion_tokens"] = eff_mt
        if extra_headers:
            kwargs["extra_headers"] = extra_headers
            # EAS UCH affinity keys on the URL query parameter (the header
            # alone never pins an upstream replica) — mirror the per-call
            # session id into ``extra_query``; per-call wins over any
            # construction-time ``default_query``.
            session_query = self._session_query(extra_headers)
            if session_query:
                kwargs["extra_query"] = session_query
        extra_body = self._effective_extra_body()
        if extra_body:
            kwargs["extra_body"] = extra_body
        if timeout is not None:
            kwargs["timeout"] = timeout
        elif self.default_timeout is not None:
            kwargs["timeout"] = self.default_timeout

        async for chunk in await self._open_stream(kwargs):
            chunk_usage = _usage_dict(getattr(chunk, "usage", None))
            chunk_model = getattr(chunk, "model", "") or ""
            if not chunk.choices:
                # Terminal ``include_usage`` chunk: empty ``choices`` but carries
                # the final token usage. Forward it (was previously dropped, so
                # streaming usage/billing read 0).
                if chunk_usage or chunk_model:
                    yield StreamDelta(usage=chunk_usage, model=chunk_model)
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            yield StreamDelta(
                content=getattr(delta, "content", None) or "",
                reasoning_content=_reasoning_text(delta),
                tool_call_deltas=_tool_call_deltas(getattr(delta, "tool_calls", None)),
                finish_reason=getattr(choice, "finish_reason", None) or "",
                model=chunk_model,
                usage=chunk_usage,
            )

    async def _open_stream(self, kwargs: dict[str, Any]) -> Any:
        """Open the stream, retrying once past a rejected ``reasoning_effort``."""
        try:
            return await self._open_stream_once(kwargs)
        except Exception as exc:
            if not self._demote_reasoning_effort(exc):
                raise
        return await self._open_stream_once(
            {**kwargs, "extra_body": self._effective_extra_body()},
        )

    async def _open_stream_once(self, kwargs: dict[str, Any]) -> Any:
        """Open the streaming completion, requesting ``stream_options.include_usage``
        so the terminal usage chunk arrives. Some OpenAI-compatible gateways
        reject ``stream_options`` with a 400 — on first rejection we disable it
        for this client and retry once without it (the stream still runs;
        streaming usage just reads 0 for that gateway, as it did before usage
        forwarding existed)."""
        if self._stream_options_supported:
            try:
                return await self._client.chat.completions.create(
                    stream_options={"include_usage": True}, **kwargs,
                )
            except BadRequestError as exc:
                if "stream_options" not in str(exc).lower():
                    raise
                self._stream_options_supported = False
                logger.warning(
                    "Gateway rejected stream_options.include_usage; disabling "
                    "it for this client (streaming token usage will read 0). %s",
                    exc,
                )
        return await self._client.chat.completions.create(**kwargs)


# ── Adapters ─────────────────────────────────────────────────────────────


def _reasoning_text(obj: Any) -> str:
    """Pull a model's thinking-channel text off a streamed ``delta`` or a
    completed ``message``.

    OpenAI-compatible endpoints disagree on the field name: SGLang / DeepSeek
    use ``reasoning_content``, while OpenRouter-style proxies (incl.
    ``api.miromind.site``) surface it as ``reasoning``. We accept either so the
    thinking channel survives whichever gateway is in front of the model.
    """
    return (
        getattr(obj, "reasoning_content", None)
        or getattr(obj, "reasoning", None)
        or ""
    )


def _usage_dict(usage: Any) -> dict[str, int]:
    """Normalise an OpenAI ``usage`` object into the wire-shape token dict.

    Shared by the non-streaming ``_to_llm_response`` and the streaming
    assembler (the terminal ``include_usage`` chunk carries the same shape).
    """
    out: dict[str, int] = {}
    if not usage:
        return out
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        v = getattr(usage, k, None)
        if v is not None:
            out[k] = int(v)
    # Prompt-caching surfaced under ``usage.prompt_tokens_details.cached_tokens``
    # on modern OpenAI / OpenRouter responses.
    ptd = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(ptd, "cached_tokens", None) if ptd is not None else None
    if cached is not None:
        out["cached_tokens"] = int(cached)
    # Thinking tokens, under ``usage.completion_tokens_details.reasoning_tokens``.
    # Billed inside ``completion_tokens``, so the totals were always right, but
    # ``extract_usage`` reads a FLAT ``reasoning_tokens`` key that nothing filled
    # here — every reasoning model behind this client (deepseek / qwen / apodex,
    # i.e. the whole chat-completions path) reported 0 and cost attribution could
    # not separate thinking from output. The sibling adapters already do this
    # (``anthropic_client._anthropic_reasoning_tokens``,
    # ``openai_responses_client`` off ``output_tokens_details``); this one was the
    # gap. Absent details (some gateways send ``null``) leave the key unset and
    # ``extract_usage`` zero-fills it.
    ctd = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(ctd, "reasoning_tokens", None) if ctd is not None else None
    if reasoning is not None:
        out["reasoning_tokens"] = int(reasoning)
    return out


def _to_llm_response(raw: Any) -> LLMResponse:
    """Convert an OpenAI ``ChatCompletion`` into an :class:`LLMResponse`."""
    choices = getattr(raw, "choices", None)
    if not choices:
        raise ValueError(
            "OpenAI-compatible response has no choices "
            f"(response_id={getattr(raw, 'id', '')!r}, "
            f"model={getattr(raw, 'model', '')!r})",
        )
    choice = choices[0]
    msg = choice.message
    tool_calls: list[ToolCall] = []
    for tc in (getattr(msg, "tool_calls", None) or []):
        tool_calls.append({
            "type": "function",
            "id": tc.id,
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments or "{}",
            },
        })
    usage_dict = _usage_dict(getattr(raw, "usage", None))
    content = getattr(msg, "content", "") or ""
    # Mirror the streaming path (llm_client._stream_llm_response): a Qwen
    # ``</think>\n\n`` separator remnant is either the whole of ``content``
    # (whitespace-only → drop) or leads the real answer (``\n\nAnswer…`` →
    # lstrip). It carries no user-visible meaning and doubles the separator
    # when ``thinking_in_history`` reconstructs the turn.
    if isinstance(content, str) and content:
        content = content.lstrip() if content.strip() else ""
    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        reasoning_content=_reasoning_text(msg),
        finish_reason=getattr(choice, "finish_reason", "") or "",
        model=getattr(raw, "model", "") or "",
        usage=usage_dict,
        response_metadata={"id": getattr(raw, "id", "")},
    )


def _tool_call_deltas(raw: Any) -> list[dict[str, Any]]:
    if not raw:
        return []
    out: list[dict[str, Any]] = []
    for tc in raw:
        out.append({
            "index": getattr(tc, "index", 0),
            "id": getattr(tc, "id", None),
            "name": getattr(getattr(tc, "function", None), "name", None),
            "arguments": getattr(getattr(tc, "function", None), "arguments", None),
        })
    return out


__all__ = [
    "OpenAIClient",
    "SessionQueryResolver",
    "SessionScopeResolver",
    "configure_session_query_resolver",
    "configure_session_scope_resolver",
]
