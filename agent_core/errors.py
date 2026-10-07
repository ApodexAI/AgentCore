"""Exception hierarchy for AgentCore."""

from __future__ import annotations

from typing import Any


class AgentCoreError(Exception):
    """Base exception for all AgentCore errors."""


# ── Kernel errors ───────────────────────────────────────────────────────────


class KernelError(AgentCoreError):
    """Errors originating from the OS kernel layer."""


class TaskNotFoundError(KernelError):
    def __init__(self, task_id: str) -> None:
        super().__init__(f"Task not found: {task_id}")
        self.task_id = task_id


class InvalidStateTransition(KernelError):
    def __init__(self, task_id: str, current: str, target: str) -> None:
        super().__init__(f"Invalid transition for {task_id}: {current} → {target}")


class ServiceNotRegistered(KernelError):
    def __init__(self, service_type: type) -> None:
        super().__init__(f"Service not registered: {service_type.__name__}")


class PermissionDenied(KernelError):
    def __init__(self, role: str, tool: str) -> None:
        super().__init__(f"Role '{role}' has no permission for tool '{tool}'")

# LLM request errors

class LLMError(AgentCoreError):
    """Errors from the LLM/provider layer."""


class LLMReasoningRunaway(LLMError):
    """A live stream spent its semantic budget on reasoning-only output.

    Unlike :class:`LLMStreamStalled`, the provider is healthy and actively
    emitting chunks. The failure is semantic: no non-whitespace visible text
    or tool-call delta appeared before the configured time/token guard fired.

    ``partial_response`` is intentionally carried separately from provider
    usage. Early stream cancellation often happens before the terminal usage
    chunk arrives, so its estimated reasoning tokens must never be presented
    as authoritative billing data.
    """

    def __init__(
        self,
        *,
        elapsed_s: float,
        estimated_tokens: int,
        trigger: str,
        partial_response: Any,
    ) -> None:
        self.elapsed_s = float(elapsed_s)
        self.estimated_tokens = int(estimated_tokens)
        self.trigger = trigger
        self.partial_response = partial_response
        super().__init__(
            "reasoning-only stream exceeded "
            f"{trigger} guard (elapsed={self.elapsed_s:.1f}s, "
            f"estimated_tokens={self.estimated_tokens})",
        )


class LLMEmptyCompletion(LLMError):
    """A streamed call completed cleanly and produced nothing at all.

    Not a stall (chunks did arrive, or the stream closed without ever going
    silent) and not a runaway (no reasoning was spent either). The provider
    simply ended the stream with no visible text, no tool call, no reasoning
    and — the tell — no usage, so there is nothing to bill and nothing to act
    on. :func:`agent_core.runtime.retriable.is_empty_completion` has always
    named this case ("reasoning-runaway, all-tokens-in-thinking, or an empty
    stream") and routes it through ``is_retriable_with_fallback``, which
    resamples on the same key and then advances the chain; what was missing is
    the raise, so the assembled empty response reached the loop as a VALID
    answer instead.

    Why that is worse than an error: an empty reply with no tool call is
    shaped exactly like "the model chose to stop talking", so under
    ``no_tool_behavior="stop"`` the run ENDS. Measured on ApodexHarness's
    2026-10-06 GDPval batch over llm-hub: 10 of 19 streamed trials died this
    way, six of them inside five minutes (one on turn 1 after 37s), each
    discarding every turn of work it had already done. The message wording is
    matched by ``_EMPTY_COMPLETION_PATTERNS`` so existing classification picks
    it up without a registry edit.
    """

    def __init__(self, *, chunks_seen: int, elapsed_s: float) -> None:
        self.chunks_seen = int(chunks_seen)
        self.elapsed_s = float(elapsed_s)
        super().__init__(
            "empty completion: stream returned no content, no tool call and "
            f"no usage (chunks_seen={self.chunks_seen}, "
            f"elapsed={self.elapsed_s:.1f}s)",
        )


class LLMTruncatedStream(LLMError):
    """An Anthropic stream ended without its protocol terminator.

    The Messages streaming protocol always closes with ``message_delta``
    (stop_reason + output usage) followed by ``message_stop``. A stream that
    ends before ``message_stop`` was cut off, whatever it carried so far, and
    the client cannot tell that from a deliberate stop: the SDK raises only on
    an SSE ``error`` event or a broken chunked body, and a gateway that loses
    its upstream mid-generation can do neither and simply finish the response
    body. What comes back is well-formed and short — thinking, no text, no tool
    call — and the loop reads it as "the model chose to stop" and ends the run.

    Measured on ApodexHarness's 2026-10-07 GDPval batch over llm-hub: 10 of 15
    finished trials ended this way, in three bursts where several unrelated
    in-flight streams were cut in the same second (02:42:20, 03:00:35 and
    03:07:46 UTC). Every one still carried ``message_start`` usage, so the
    empty-completion guard (which requires *no* usage) let it through.

    The wording matches ``_TRANSIENT_NETWORK_PATTERNS``: a dropped connection
    is retried on the same key with backoff, which is what clears it. The
    carried fields are the stream's end record, so the cause is visible in the
    retry log and attempt telemetry instead of only in a missing deliverable.
    """

    def __init__(
        self,
        *,
        last_event: str,
        events_seen: int,
        saw_message_delta: bool,
        block_types: list[str],
        elapsed_s: float,
    ) -> None:
        self.last_event = last_event
        self.events_seen = int(events_seen)
        self.saw_message_delta = bool(saw_message_delta)
        self.block_types = list(block_types)
        self.elapsed_s = float(elapsed_s)
        super().__init__(
            "truncated stream: ended without message_stop "
            f"(last_event={last_event or 'none'}, events_seen={self.events_seen}, "
            f"message_delta={'yes' if self.saw_message_delta else 'no'}, "
            f"blocks=[{','.join(self.block_types)}], "
            f"elapsed={self.elapsed_s:.1f}s)",
        )


class LLMStreamStalled(LLMError, TimeoutError):
    """A streaming LLM call went silent mid-flight.

    Subclasses ``asyncio.TimeoutError`` so every existing transient-
    timeout handler (retry/backoff in ``call_llm``, chain wrappers,
    classification) treats it identically without changes; carried
    fields make the distinct failure mode visible in logs and traces.
    """

    def __init__(
        self, stall_s: float, chunks_seen: int, elapsed_s: float,
    ) -> None:
        self.stall_s = stall_s
        self.chunks_seen = chunks_seen
        self.elapsed_s = elapsed_s
        super().__init__(
            f"stream stalled: no chunks for {stall_s:.0f}s "
            f"(chunks_seen={chunks_seen}, elapsed={elapsed_s:.0f}s)",
        )


class LLMDeadlineExceeded(LLMError, TimeoutError):
    """An LLM attempt was stopped by an enclosing runtime deadline.

    ``reason`` is deliberately carried on the underlying exception as well as
    on :class:`LLMCallExhausted`. Some callers unwrap ``last_exc`` before
    handing it to a provider-chain policy; a dedicated type prevents that
    policy from mistaking an exhausted run budget for an ordinary transient
    provider timeout.
    """

    def __init__(self, reason: str, detail: str) -> None:
        self.reason = reason
        super().__init__(f"{reason}: {detail}")


class LLMCallExhausted(LLMError, RuntimeError):
    """Raised by ``call_llm`` when retries are exhausted or the error is
    structurally unrecoverable (4xx without proxy-wrap, or a chain-aware
    fallback signal like ``model_not_found``).

    Wraps the last exception encountered so the caller (typically the
    product's agent loop) can surface it to a provider-chain wrapper for
    L1→L2→L3 rotation. Carries ``last_exc`` separately because
    ``raise from`` is too opaque for chain-aware classification — a chain
    wrapper calls ``classify_error(last_exc)`` directly.

    ``last_exc`` must always agree with ``reason``: it is the exception that
    *caused this raise*, not merely the most recent failure seen. A deadline
    refusal therefore carries :class:`LLMDeadlineExceeded` even when earlier
    attempts failed for unrelated reasons. The wrapper's ``reason`` remains
    authoritative, while the underlying exception preserves the same reason
    if a caller unwraps it before classification.

    ``prior_exc`` is where that earlier, superseded failure goes: diagnostic
    context for logs and post-mortems, deliberately outside the field
    classification reads.
    """

    def __init__(
        self,
        last_exc: BaseException,
        reason: str,
        *,
        prior_exc: BaseException | None = None,
    ) -> None:
        self.last_exc = last_exc
        self.reason = reason
        self.prior_exc = prior_exc
        detail = f"call_llm {reason}: {last_exc!r}"
        if prior_exc is not None and prior_exc is not last_exc:
            detail += f" (after {prior_exc!r})"
        super().__init__(detail)


__all__ = [
    "AgentCoreError",
    "InvalidStateTransition",
    "KernelError",
    "LLMCallExhausted",
    "LLMDeadlineExceeded",
    "LLMEmptyCompletion",
    "LLMError",
    "LLMReasoningRunaway",
    "LLMStreamStalled",
    "LLMTruncatedStream",
    "PermissionDenied",
    "ServiceNotRegistered",
    "TaskNotFoundError",
]
