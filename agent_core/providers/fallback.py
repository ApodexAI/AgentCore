"""``LLMFallbackChain`` — provider-failover :class:`LLMClient` wrapper.

Master design §5.9. Wraps an ordered list of LLM clients; each call tries
the primary first, and on configured failure modes falls through to the
next entry. Implements the native :class:`agent_core.llm.LLMClient`
interface so existing callers (kernel loop, ``ResourceManager.get_llm()``)
get failover transparently — they keep calling ``chat`` / ``stream`` as
before.

Triggers
--------
Each ``FallbackEntry`` declares which exception classes / signatures
trigger the fall-through. Built-in triggers:

- ``timeout``     — ``asyncio.TimeoutError`` or any exception whose name
                    or message contains "timeout".
- ``rate_limit``  — HTTP 429, or message contains "rate limit" / "quota"
                    / "too many requests".
- ``5xx``         — exception with a ``status_code`` attribute in
                    [500, 599], or message starting with "5".
- ``any_error``   — match anything (use this on the last entry to
                    guarantee at least one fallback).

A trigger only applies to the entry that *failed*; the next entry
inherits its own trigger list. The chain stops on the first successful
generation, or re-raises the last exception when no entry's trigger
matched.

Layering
--------
This module imports nothing from ``core/runtime/`` or ``components/``
— it is pure infra. The kernel loop's ``llm_client`` accepts any
``LLMClient``; passing an ``LLMFallbackChain`` instance just works.

Telemetry
---------
On a successful call, the chosen response's ``response_metadata`` is
stamped with two extra fields:

- ``fallback_used`` — index of the entry that succeeded (0 means the
                      primary; >0 means a fallback fired).
- ``model_actually_used`` — best-effort model id of the entry that
                            succeeded (``model_name`` / ``model``
                            attribute, or class name).

The proxy / tracing middleware in ``core/runtime/middleware/llm/``
reads these from ``response_metadata`` and forwards them into the
``llm_call_finished`` event so observability can flag failover runs
without a separate IPC channel.
"""

from __future__ import annotations

# pyright: basic, reportPrivateImportUsage=false
import asyncio
import logging
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from agent_core.llm import LLMResponse, StreamDelta
from agent_core.messages import Message

logger = logging.getLogger(__name__)

__all__ = [
    "LEGACY_RETRYABLE_KEYWORDS",
    "CooldownFallbackLLM",
    "FallbackEntry",
    "FallbackTrigger",
    "LLMFallbackChain",
    "legacy_retryable",
    "with_provider_stamp",
]

type FallbackEventHook = Callable[[str, dict[str, Any]], Awaitable[None]]

# The single retryable-keyword table. ``components.middleware.llm.base``
# re-binds its own ``_RETRYABLE_KEYWORDS`` to this rather than keeping a second
# copy — two tables for one policy drift.
LEGACY_RETRYABLE_KEYWORDS = frozenset({
    "timeout", "timed out", "429", "500", "502", "503", "504", "529",
    "overloaded", "rate limit", "rate_limit", "server error",
    "connection reset", "connection error", "econnreset", "gateway timeout",
    "model_dump", "model_not_found",
})


def legacy_retryable(error: Exception) -> bool:
    """Match the historical two-model fallback wrapper's retry policy."""
    return isinstance(error, AttributeError) or any(
        keyword in str(error).lower() for keyword in LEGACY_RETRYABLE_KEYWORDS
    )


async def _noop_event(_name: str, _payload: dict[str, Any]) -> None:
    return None


FallbackTrigger = Literal["timeout", "rate_limit", "5xx", "any_error"]


@dataclass
class FallbackEntry:
    """A single tier in the fallback chain.

    Args:
        model: any :class:`LLMClient` instance.
        triggers: which failure modes count as "fall through to the
            next entry". Empty tuple ``()`` means "never fall through" —
            useful as a hard barrier at any position in the chain.
            ``("any_error",)`` means "always fall through". ``None``
            (the default) means "unset": inside an
            :class:`LLMFallbackChain` the chain's ``default_triggers``
            are substituted; standalone it behaves as ``("any_error",)``.

            ``None`` and ``()`` are deliberately distinct. Sharing one
            sentinel would make a hard barrier indistinguishable from an
            unconfigured entry, and chain normalisation would silently
            overwrite the barrier with ``default_triggers``.
        provider: vendor label (``openai`` / ``anthropic`` / ``qwen`` /
            ``mirothinker``) — stamped onto ``response_metadata`` as
            ``provider_actually_used`` so per-call ``usage`` events can
            tell downstream billing which vendor served the request.
            Empty string when the construction site doesn't know.
    """
    model: Any  # LLMClient
    triggers: tuple[FallbackTrigger, ...] | None = None
    provider: str = ""

    def matches(self, exc: BaseException) -> bool:
        """Whether ``exc`` should trigger fall-through past this entry.

        An unset (``None``) trigger tuple falls back to ``any_error`` so a
        standalone entry keeps the permissive default; an explicitly empty
        tuple matches nothing, which is the hard-barrier contract.
        """
        triggers = ("any_error",) if self.triggers is None else self.triggers
        return any(_trigger_matches(trigger, exc) for trigger in triggers)


def _model_id(model: Any) -> str:
    """Best-effort short label for a model. Used for telemetry.

    ``model_name`` / ``model`` come off arbitrary duck-typed clients, so the
    result is coerced: an enum, a pydantic field or a ``Mock`` must not flow
    into ``self.model`` and every telemetry payload as a non-string.
    """
    return str(
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or type(model).__name__
    )


def _trigger_matches(trigger: FallbackTrigger, exc: BaseException) -> bool:
    """Decide whether ``exc`` matches the given trigger keyword."""
    if trigger == "any_error":
        return True

    name = type(exc).__name__.lower()
    msg = str(exc).lower()

    if trigger == "timeout":
        if isinstance(exc, asyncio.TimeoutError):
            return True
        return "timeout" in name or "timeout" in msg or "timed out" in msg

    if trigger == "rate_limit":
        status = getattr(exc, "status_code", None)
        if status == 429:
            return True
        return any(
            phrase in msg
            for phrase in (
                "rate limit",
                "rate_limit",
                "quota",
                "too many requests",
            )
        )

    if trigger == "5xx":
        status = getattr(exc, "status_code", None)
        if isinstance(status, int) and 500 <= status <= 599:
            return True
        # Many SDKs encode the code in the message.
        return any(
            phrase in msg
            for phrase in (
                "internal server error",
                "bad gateway",
                "service unavailable",
                "gateway timeout",
                "503",
                "502",
                "500",
                "504",
            )
        )

    return False


class CooldownFallbackLLM:
    """Retry a primary client, then use a fallback client during cooldown.

    This preserves the legacy two-model fallback behavior used by both host
    products. Product tracing is an optional async event hook, keeping the
    retry state machine independent of execution scopes and trace backends.

    ``stream`` stops retrying the primary once any delta has reached the
    consumer, because a retry would emit those deltas twice. Set
    ``replay_partial_stream=True`` to restore the historical duplicating
    behavior; prefer :class:`LLMFallbackChain` for rewind-safe semantics.

    Event hook contract
    -------------------
    Every event payload carries ``leg`` — ``"primary"`` or ``"fallback"`` —
    naming the client the event is *about*, so a tracing adapter never has to
    infer it. ``retry`` and ``abandon_stream_retry`` are primary-leg events:
    they describe what happened to the primary, not the leg that serves next.

    Events emitted:

    - ``request`` — a leg is about to be called. ``attempt`` on the primary
      leg; ``mode`` (``"cooldown"`` / ``"degraded"``) on the fallback leg.
    - ``error`` — that leg's call raised.
    - ``retry`` — the primary failed and will be retried after ``delay_s``.
    - ``abandon_stream_retry`` — the primary stream failed after deltas had
      already reached the consumer, so it degrades instead of replaying. A
      retry-shaped event: the ``degrade`` that follows describes the move.
    - ``degrade`` — traffic moves to the fallback leg (``leg="fallback"``);
      always carries ``reason``, ``degrade_from`` and ``degrade_to``.

    Stream-path events additionally carry ``streaming=True``.

    Logging
    -------
    Every event is also logged, independently of ``event_hook`` — a degrade
    moves traffic to a different model for ``cooldown_seconds`` and an operator
    has to be able to see that in logs whether or not a telemetry sink is
    wired. Retries, abandoned stream retries and degrades log at ``warning``; a
    failing fallback leg logs at ``error``; a primary-leg error logs at
    ``debug``, since the ``retry`` or ``degrade`` line that follows it carries
    the operational signal. Hosts should not re-log off the hook.
    """

    def __init__(
        self,
        primary: Any,
        fallback: Any,
        max_retries: int = 2,
        cooldown_seconds: int = 60,
        *,
        retryable: Callable[[Exception], bool] = legacy_retryable,
        event_hook: FallbackEventHook = _noop_event,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[], float] = random.random,
        replay_partial_stream: bool = False,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.max_retries = max_retries
        self.cooldown_seconds = cooldown_seconds
        self.model: str = _model_id(primary)
        self._retryable = retryable
        self._event_hook = event_hook
        self._clock = clock
        self._sleep = sleep
        self._jitter = jitter
        self._replay_partial_stream = replay_partial_stream
        self._cooldown_until = 0.0

    # ``model_name`` is derived, ``model`` is a snapshot, and the split is
    # forced: ``LLMClient`` declares ``model`` as a settable attribute (the
    # concrete clients subclass the Protocol and assign it in ``__init__``, so
    # it is a real descriptor slot), and a property there stops the class
    # satisfying that. ``model_name`` carries no such constraint, so it reads
    # through to the current primary — that is the label to use when a primary
    # may be lazily initialised or swapped by middleware after construction.
    @property
    def model_name(self) -> str:
        return f"fallback({_model_id(self.primary)})"

    def _log_event(self, name: str, payload: dict[str, Any]) -> None:
        """Operational log line for one event. See the class docstring."""
        if name == "error":
            if payload.get("leg") == "fallback":
                logger.error(
                    "CooldownFallbackLLM: fallback leg also failed: %s",
                    payload.get("error"),
                )
            else:
                logger.debug(
                    "CooldownFallbackLLM: primary leg failed: %s",
                    payload.get("error"),
                )
        elif name == "retry":
            logger.warning(
                "CooldownFallbackLLM: primary attempt %s/%s failed (%s), "
                "retrying in %.1fs",
                payload.get("attempt"),
                self.max_retries,
                payload.get("error_type"),
                float(payload.get("delay_s") or 0.0),
            )
        elif name == "abandon_stream_retry":
            logger.warning(
                "CooldownFallbackLLM: primary stream failed on attempt %s "
                "after emitting deltas; degrading instead of replaying them",
                payload.get("attempt"),
            )
        elif name == "degrade":
            logger.warning(
                "CooldownFallbackLLM: degrading %s -> %s (%s), cooldown %ss",
                payload.get("degrade_from"),
                payload.get("degrade_to"),
                payload.get("reason"),
                payload.get("cooldown_seconds", self.cooldown_seconds),
            )

    async def _emit(self, name: str, **payload: Any) -> None:
        self._log_event(name, payload)
        try:
            await self._event_hook(name, payload)
        except Exception:
            logger.debug("Fallback event hook failed", exc_info=True)

    def _kwargs(
        self,
        *,
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        max_tokens: int | None,
        extra_headers: dict[str, str] | None,
        timeout: float | None,
    ) -> dict[str, Any]:
        return {
            "tools": tools,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_headers": extra_headers,
            "timeout": timeout,
        }

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
        kwargs = self._kwargs(
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_headers=extra_headers,
            timeout=timeout,
        )
        if self._clock() < self._cooldown_until:
            await self._emit(
                "degrade",
                leg="fallback",
                reason="primary_cooldown",
                degrade_from=_model_id(self.primary),
                degrade_to=_model_id(self.fallback),
            )
            await self._emit("request", leg="fallback", mode="cooldown")
            return await self.fallback.chat(messages, **kwargs)

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                await self._emit("request", leg="primary", attempt=attempt + 1)
                return await self.primary.chat(messages, **kwargs)
            except Exception as error:
                last_error = error
                await self._emit(
                    "error",
                    leg="primary",
                    attempt=attempt + 1,
                    error=str(error),
                )
                if not self._retryable(error):
                    break
                if attempt + 1 >= self.max_retries:
                    # Last attempt: sleeping here only delays the degrade and
                    # emits a ``retry`` event that no retry follows.
                    break
                delay = min(0.5 * (2**attempt), 8.0) + self._jitter() * 0.25
                await self._emit(
                    "retry",
                    leg="primary",
                    attempt=attempt + 1,
                    delay_s=delay,
                    error_type=type(error).__name__,
                )
                await self._sleep(delay)

        self._cooldown_until = self._clock() + self.cooldown_seconds
        await self._emit(
            "degrade",
            leg="fallback",
            reason="primary_exhausted",
            degrade_from=_model_id(self.primary),
            degrade_to=_model_id(self.fallback),
            cooldown_seconds=self.cooldown_seconds,
            error=str(last_error) if last_error else "",
        )
        await self._emit("request", leg="fallback", mode="degraded")
        try:
            return await self.fallback.chat(messages, **kwargs)
        except Exception as fallback_error:
            await self._emit("error", leg="fallback", error=str(fallback_error))
            if last_error is None:
                raise
            raise last_error from fallback_error

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
        kwargs = self._kwargs(
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_headers=extra_headers,
            timeout=timeout,
        )
        if self._clock() < self._cooldown_until:
            await self._emit(
                "degrade",
                leg="fallback",
                reason="primary_stream_cooldown",
                degrade_from=_model_id(self.primary),
                degrade_to=_model_id(self.fallback),
                streaming=True,
            )
            await self._emit(
                "request", leg="fallback", mode="cooldown", streaming=True,
            )
            async for delta in self.fallback.stream(messages, **kwargs):
                yield delta
            return

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            yielded = False
            try:
                await self._emit(
                    "request",
                    leg="primary",
                    attempt=attempt + 1,
                    streaming=True,
                )
                async for delta in self.primary.stream(messages, **kwargs):
                    yielded = True
                    yield delta
                return
            except Exception as error:
                last_error = error
                await self._emit(
                    "error",
                    leg="primary",
                    attempt=attempt + 1,
                    streaming=True,
                    error=str(error),
                )
                if not self._retryable(error):
                    break
                if yielded and not self._replay_partial_stream:
                    # Deltas already reached the consumer; retrying the primary
                    # would duplicate them. Degrade to the fallback leg instead.
                    await self._emit(
                        "abandon_stream_retry",
                        leg="primary",
                        reason="primary_stream_partial",
                        attempt=attempt + 1,
                        streaming=True,
                        yielded=True,
                    )
                    break
                if attempt + 1 >= self.max_retries:
                    break
                delay = min(0.5 * (2**attempt), 8.0) + self._jitter() * 0.25
                await self._emit(
                    "retry",
                    leg="primary",
                    attempt=attempt + 1,
                    delay_s=delay,
                    streaming=True,
                    yielded=yielded,
                    error_type=type(error).__name__,
                )
                await self._sleep(delay)

        self._cooldown_until = self._clock() + self.cooldown_seconds
        await self._emit(
            "degrade",
            leg="fallback",
            reason="primary_stream_exhausted",
            degrade_from=_model_id(self.primary),
            degrade_to=_model_id(self.fallback),
            cooldown_seconds=self.cooldown_seconds,
            streaming=True,
        )
        await self._emit(
            "request", leg="fallback", mode="degraded", streaming=True,
        )
        try:
            async for delta in self.fallback.stream(messages, **kwargs):
                yield delta
        except Exception as fallback_error:
            await self._emit(
                "error",
                leg="fallback",
                streaming=True,
                error=str(fallback_error),
            )
            if last_error is None:
                raise
            raise last_error from fallback_error


@dataclass
class LLMFallbackChain:
    """Ordered list of :class:`LLMClient` entries with per-entry triggers.

    ``LLMFallbackChain([primary, fallback], default_triggers=("timeout","5xx"))``
    is the typical 2-tier setup. The chain's call surface is identical to
    a single :class:`LLMClient`: ``chat`` / ``stream`` delegate to whichever
    entry succeeds.

    Args:
        entries: ordered list of ``FallbackEntry``. The first entry is
            the primary; subsequent entries are tried in order on
            matching failure. Must be non-empty.
        default_triggers: applied to entries whose ``triggers`` field is
            left unset (``None``). Default is ``("any_error",)``. Entries
            that declare an explicit ``()`` keep it — that is the
            "never fall through" barrier and is not a missing value.

    Notes:
        - Streaming (``stream``) only fails over BEFORE the first chunk
          is yielded. Once the consumer has seen output we cannot rewind,
          so a mid-stream error propagates as-is. Token batching strategies
          (yield-once-complete) sidestep this; for true streaming
          robustness use a ``rate_limit`` / ``timeout`` trigger only and
          accept the mid-stream failure mode.
    """

    entries: list[FallbackEntry] = field(default_factory=list)
    default_triggers: tuple[FallbackTrigger, ...] = ("any_error",)
    model: str = field(init=False, default="")

    def __post_init__(self) -> None:
        if not self.entries:
            raise ValueError("LLMFallbackChain requires at least one entry")
        self.model = _model_id(self.entries[0].model)
        # Normalise default triggers onto entries that left them unset.
        # ``None`` means "unset" and inherits ``default_triggers``; an
        # explicit ``()`` is a hard barrier and MUST survive untouched, so
        # this tests for ``None`` rather than falsiness.
        for entry in self.entries:
            if entry.triggers is None:
                entry.triggers = self.default_triggers

    @classmethod
    def from_models(
        cls,
        models: Sequence[Any],
        *,
        triggers: tuple[FallbackTrigger, ...] = ("any_error",),
    ) -> LLMFallbackChain:
        """Convenience: build a chain where every entry uses the same triggers."""
        entries = [FallbackEntry(model=m, triggers=triggers) for m in models]
        return cls(entries=entries, default_triggers=triggers)

    @property
    def model_name(self) -> str:
        names = ",".join(_model_id(e.model) for e in self.entries)
        return f"llm_fallback_chain[{names}]"

    # ------------------------------------------------------------------
    # Non-streaming path — delegates to entries one at a time.
    # ------------------------------------------------------------------
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
        last_exc: BaseException | None = None
        for idx, entry in enumerate(self.entries):
            try:
                result = await entry.model.chat(
                    messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_headers=extra_headers,
                    timeout=timeout,
                )
                _stamp_metadata(result, idx, entry.model, entry.provider)
                return result
            except Exception as exc:
                last_exc = exc
                if idx == len(self.entries) - 1 or not entry.matches(exc):
                    raise
                logger.info(
                    "LLMFallbackChain: entry %d (%s) failed (%s); "
                    "falling through to entry %d",
                    idx, _model_id(entry.model), type(exc).__name__, idx + 1,
                )
        raise RuntimeError("LLMFallbackChain exhausted") from last_exc

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
        # Try entries in order. We can only fail over BEFORE any chunk
        # has been forwarded to the caller — once we yield, the consumer
        # has committed to that entry.
        last_exc: BaseException | None = None
        for idx, entry in enumerate(self.entries):
            yielded_any = False
            try:
                async for delta in entry.model.stream(
                    messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_headers=extra_headers,
                    timeout=timeout,
                ):
                    # ``StreamDelta`` has no ``response_metadata`` channel, but
                    # it does carry a ``provider`` slot: stamp the serving leg's
                    # vendor so the stream assembler can fold it into
                    # ``LLMResponse.response_metadata["provider_actually_used"]``
                    # (matching the non-streaming ``chat`` path's
                    # ``_stamp_metadata``). Constant for the whole stream —
                    # failover only fires before the first yield, so every
                    # delta below comes from this single committed entry.
                    # ``fallback_used`` / ``model_actually_used`` still have no
                    # streaming landing spot; only the billing-critical provider
                    # is carried here.
                    if entry.provider:
                        delta.provider = entry.provider
                    yielded_any = True
                    yield delta
                return
            except Exception as exc:
                last_exc = exc
                if yielded_any:
                    raise
                if idx == len(self.entries) - 1 or not entry.matches(exc):
                    raise
                logger.info(
                    "LLMFallbackChain stream: entry %d (%s) failed "
                    "before any chunk (%s); falling through",
                    idx, _model_id(entry.model), type(exc).__name__,
                )
        raise RuntimeError("LLMFallbackChain exhausted") from last_exc


def _stamp_metadata(
    result: LLMResponse, idx: int, model: Any, provider: str = "",
) -> None:
    """Record which entry served the call on the response metadata.

    Mutates ``LLMResponse.response_metadata`` in place — the proxy /
    tracing middleware reads ``fallback_used`` / ``model_actually_used``
    from there and forwards them into the ``llm_call_finished`` event.

    ``provider`` is stamped as ``provider_actually_used`` so downstream
    billing (``extract_usage`` → per-call ``usage`` event) can tell which
    vendor served a given call. Empty string is allowed when the chain
    construction site didn't know the vendor.
    """
    if not isinstance(result, LLMResponse):
        return
    result.response_metadata["fallback_used"] = idx
    result.response_metadata["model_actually_used"] = _model_id(model)
    if provider:
        result.response_metadata["provider_actually_used"] = provider


def with_provider_stamp(llm: Any, provider: str) -> Any:
    """Wrap any :class:`LLMClient` so responses carry ``provider_actually_used``.

    Returns ``llm`` unchanged when ``provider`` is empty (no-op fast path).
    Otherwise builds a 1-entry ``LLMFallbackChain`` with ``triggers=()``
    (the hard barrier — never falls through), making the wrap
    semantically transparent:

    - Successful response → ``response_metadata.provider_actually_used``
      gets stamped with ``provider`` (and ``model_actually_used`` /
      ``fallback_used=0`` come along — the chain's standard stamping).
    - Any exception → propagates as-is, since the single entry's empty
      trigger tuple matches nothing.

    Used by workflows that construct raw ``ChatOpenAI`` /
    ``ChatAnthropic`` outside the kernel's ``llm_fallback_chain`` config
    (swarm profiles, heavy_mode provider_chain attempts, SWE / apodex
    bases, sdk_cli's dev factory). Without this wrap the per-call
    ``usage`` event's ``provider`` field stays empty, blocking
    cross-vendor billing attribution.

    No-op for duck-typed test stubs that don't implement the native
    ``LLMClient`` dispatch surface — the chain routes through ``chat`` /
    ``stream``, so a stub lacking those would crash the call. The quiet
    skip keeps tests stable while still wrapping every real LLM instance
    in production.
    """
    if not provider:
        return llm
    if not (hasattr(llm, "chat") and hasattr(llm, "stream")):
        return llm
    return LLMFallbackChain(
        entries=[
            FallbackEntry(model=llm, triggers=(), provider=provider),
        ],
    )
