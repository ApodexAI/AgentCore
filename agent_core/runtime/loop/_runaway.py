# pyright: reportPrivateUsage=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnusedFunction=false
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from agent_core.llm import LLMResponse
from agent_core.messages import Message
from agent_core.runtime.env import first_configured
from agent_core.runtime.llm_request_overrides import ThinkingRetryOverride
from agent_core.tokens import estimate_message_tokens

from ._bind import _ensure_bound
from ._response import _visible_response_text, extract_usage

logger = logging.getLogger(__name__)
# ── Reasoning-runaway detection (capped-empty completions) ───────────
#
# Some reasoning models behind OpenAI-compatible gateways can burn the entire
# ``max_tokens`` budget inside
# the reasoning channel and return a *successful* response with zero
# visible content and no tool calls (``finish_reason="length"``).
# ``call_llm`` treats that signature as retriable-same-key. The very first
# runaway switches the retry to a reduced ``max_tokens`` and appends a
# transient, throwaway reminder asking for concise reasoning plus visible
# output/tool use. The reminder never enters durable message history.
# If the retry budget is exhausted the response is returned as-is so the
# loop's existing no-tool nudge stays the behavioural floor — a runaway must
# never escalate into a fatal ``llm_error`` stop past turn 1.

_RUNAWAY_MIN_OUTPUT_TOKENS = 1024
# Ceiling for retry caps after a confirmed runaway. The actual bound cap
# is derived from the observed completion usage so low-cap profiles
# (e.g. CI smoke at 1024) are not accidentally raised to this value.
_RUNAWAY_RETRY_MAX_TOKENS = 8192
_RUNAWAY_EXPAND_FACTOR = 1.5
_RUNAWAY_CONTEXT_RESERVE_TOKENS = 1024
# The reminder rides as a ``user`` turn, NOT a trailing ``system`` one.
# Providers disagree on non-leading system messages: an Anthropic adapter
# only lifts a LEADING system message out of the array and maps anything
# else through the ``{"role": "user"}`` fallthrough, and chat templates on
# SGLang/vLLM commonly render only the first system block. A user turn means
# every provider sees the same instruction in the same position.
_RUNAWAY_EXPANDED_GUIDANCE = (
    "[system reminder] The previous attempt used its full private-reasoning "
    "budget without producing visible output. This task may legitimately "
    "require extended scientific or mathematical reasoning, so this retry "
    "has a larger thinking budget. Complete the analysis, while reserving "
    "enough output for either one valid tool call or a visible answer."
)
_RUNAWAY_RECOVERY_GUIDANCE = (
    "[system reminder] The expanded-thinking retry still produced no visible "
    "answer or tool call, or could not fit in the available context. Use only "
    "a short, bounded reasoning pass now, then promptly emit either one valid "
    "tool call or visible answer text. Do not re-derive the full plan."
)
_RUNAWAY_DIRECT_RECOVERY_GUIDANCE = (
    "[system reminder] Three consecutive attempts spent their budgets in "
    "private reasoning without producing a visible answer or tool call. "
    "Thinking is disabled for this retry. Immediately emit either one valid "
    "tool call that advances the task or a visible best-effort answer."
)


def _env_value(suffix: str) -> tuple[str, str] | None:
    """Return the first configured shared/legacy environment value.

    ``AGENT_CORE_*`` is the portable spelling. The product-prefixed names
    remain supported while the shared package is extracted. The order is
    fixed and owned by :data:`agent_core.runtime.env.ENV_PREFIXES` — a
    second copy here would let the two drift apart the moment a prefix is
    added or reordered in only one of them.
    """
    return first_configured(suffix)


def _env_int(suffix: str, default: int) -> int:
    configured = _env_value(suffix)
    if configured is None:
        return default
    name, raw = configured
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %d", name, raw, default)
        return default


def _env_float(suffix: str, default: float) -> float:
    configured = _env_value(suffix)
    if configured is None:
        return default
    name, raw = configured
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        return default


# Three semantic retries: expanded thinking → reduced thinking → thinking off.
# Read once at import; deployments may lower it as a cost-control knob.
_RUNAWAY_MAX_RETRIES = _env_int("RUNAWAY_MAX_RETRIES", 3)
_RUNAWAY_BACKOFF_S = 2.0
_RUNAWAY_DISABLE_THINKING_AFTER = 3
_RUNAWAY_THINKING_BUDGET_MAX = 4096
_RUNAWAY_THINKING_BUDGET_MIN = 512
# A lowered retry budget intentionally skips the expensive expanded phase.
_RUNAWAY_EXPAND_ENABLED = _RUNAWAY_MAX_RETRIES >= _RUNAWAY_DISABLE_THINKING_AFTER
_RUNAWAY_EXPANDED_OUTPUT_RESERVE = 0.25
# Key under which a caller threads cross-turn runaway state
# (a mutable dict) through its metadata into ``call_llm``. Contents:
#   consecutive_turns                   — diagnostic streak counter (log only)
#   last_call_runaway_responses         — surfaced on ``llm_finished``
#   last_call_runaway_reasoning_chars   — surfaced on ``llm_finished``
#   last_call_recovered / last_call_reason — surfaced on ``llm_finished``
RUNAWAY_STATE_KEY = "_runaway_state"

def _is_runaway_response(response: Any) -> bool:
    """True for a successful response whose budget went entirely to
    reasoning: no visible content, no tool calls, and either
    ``finish_reason="length"`` or a completion-token count too large to
    be a plain empty reply (gateways that drop ``finish_reason``)."""
    if not isinstance(response, LLMResponse):
        return False
    if response.tool_calls:
        return False
    if _visible_response_text(response):
        return False
    if response.finish_reason == "length":
        return True
    usage = extract_usage(response) or {}
    return int(usage.get("completion_tokens") or 0) >= _RUNAWAY_MIN_OUTPUT_TOKENS

# Continuation asked of a model whose previous reply was cut off mid-sentence.
# Deliberately does NOT reduce ``max_tokens`` the way the runaway path does: this
# model was producing real output when the cap hit, so giving it less room would
# truncate it again sooner. Brevity is requested in words instead.
TRUNCATION_CONTINUATION_GUIDANCE = (
    "[system reminder] Your previous reply hit the output token limit and was "
    "cut off mid-sentence. The partial text is above. Continue from exactly "
    "where it stopped — do not repeat what you already wrote, and do not start "
    "over. Be brief and reach a tool call or a complete answer this time."
)


def is_truncated_with_text(response: Any) -> bool:
    """True for a reply the token cap cut off *after* it had produced text.

    The other half of :func:`_is_runaway_response`, which handles the same
    ``finish_reason="length"`` with the visible text *empty*. Between them they
    cover the signal, and the split matters because the two need opposite
    treatment: a runaway gets resampled at a smaller cap, while this one already
    contains work worth keeping and needs to be continued.

    Nothing detected this case before. It fell through as an ordinary turn,
    reached ``if not parsed_calls`` and — under ``no_tool_behavior="stop"`` —
    ended the run on a sentence cut mid-token.

    Unlike the runaway detector, there is **no completion-token fallback** for
    gateways that drop ``finish_reason``. That heuristic reads "a large
    completion with nothing visible cannot be a plain empty reply", which is
    sound only while the text is empty. With text present a large completion is
    what a long legitimate answer looks like, so the same heuristic would
    declare every one of them truncated. An explicit ``finish_reason`` is the
    only evidence that can carry this.
    """
    if not isinstance(response, LLMResponse):
        return False
    if response.tool_calls:
        return False
    if response.finish_reason != "length":
        return False
    return bool(_visible_response_text(response))


def _runaway_retry_max_tokens(response: Any) -> int | None:
    """Return a retry cap that cannot exceed the observed runaway cap.

    Each successive runaway inside one call halves again (the second
    reduction is derived from a completion that was ALREADY capped), so the
    squeeze is progressive. ``_RUNAWAY_MIN_OUTPUT_TOKENS`` is the floor: below
    it a capped-empty completion is no longer even detectable as a runaway
    (see :func:`_is_runaway_response`), so shrinking past it would trade a
    diagnosable failure for a silent empty reply.
    """
    usage = extract_usage(response) or {}
    try:
        completion_tokens = int(usage.get("completion_tokens") or 0)
    except (TypeError, ValueError):
        completion_tokens = 0
    if completion_tokens <= 0:
        return None
    if completion_tokens <= _RUNAWAY_MIN_OUTPUT_TOKENS:
        return completion_tokens
    return max(
        _RUNAWAY_MIN_OUTPUT_TOKENS,
        min(_RUNAWAY_RETRY_MAX_TOKENS, completion_tokens // 2),
    )


def _runaway_retry_max_tokens_from_cap(active_cap: Any) -> int | None:
    """Derive a safe retry cap when an early-cancelled stream has no usage.

    The active request cap is authoritative for the upper bound. This helper
    must never raise a low-cap profile toward the normal 8K recovery ceiling.
    """
    try:
        cap = int(active_cap)
    except (TypeError, ValueError):
        return None
    if cap <= 0:
        return None
    if cap <= _RUNAWAY_MIN_OUTPUT_TOKENS:
        return cap
    return max(
        _RUNAWAY_MIN_OUTPUT_TOKENS,
        min(_RUNAWAY_RETRY_MAX_TOKENS, cap // 2),
    )


def _bind_reduced_max_tokens(
    llm: Any,
    response: Any | None = None,
    *,
    active_cap: Any = None,
) -> Any:
    """Bind the runaway-retry ``max_tokens`` cap for follow-up attempts.

    Mirrors :func:`bind_temperature` — falls back to the original LLM
    when no usable cap can be inferred.
    """
    retry_max_tokens = (
        _runaway_retry_max_tokens(response)
        if response is not None
        else _runaway_retry_max_tokens_from_cap(active_cap)
    )
    if retry_max_tokens is None:
        logger.debug(
            "Runaway max_tokens cap could not be inferred from usage; "
            "retrying at the existing budget.",
        )
        return llm
    return replace(_ensure_bound(llm), max_tokens=retry_max_tokens)


def _expanded_retry_max_tokens(
    active_cap: Any,
    messages: list[Message],
    context_token_limit_hint: int | None,
) -> int | None:
    """Return a safe 1.5× cap, or ``None`` when context cannot hold it."""
    try:
        cap = int(active_cap)
    except (TypeError, ValueError):
        return None
    if cap <= 0:
        return None
    expanded = int(cap * _RUNAWAY_EXPAND_FACTOR)
    if context_token_limit_hint:
        available = max(
            int(context_token_limit_hint)
            - sum(estimate_message_tokens(message) for message in messages)
            - _RUNAWAY_CONTEXT_RESERVE_TOKENS,
            0,
        )
        expanded = min(expanded, available)
    return expanded if expanded > cap else None


def _bind_expanded_max_tokens(
    llm: Any,
    *,
    active_cap: Any,
    messages: list[Message],
    context_token_limit_hint: int | None,
) -> Any | None:
    expanded = _expanded_retry_max_tokens(
        active_cap, messages, context_token_limit_hint,
    )
    if expanded is None:
        return None
    return replace(_ensure_bound(llm), max_tokens=expanded)


def _phase_reasoning_guard(
    retry_number: int,
    *,
    previous_cap: Any,
    next_cap: Any,
    timeout_s: float | None,
    max_tokens: int | None,
) -> tuple[float | None, int | None]:
    """Scale early-runaway guards with the expanded completion cap."""
    if retry_number != 1:
        return timeout_s, max_tokens
    try:
        ratio = int(next_cap) / int(previous_cap)
    except (TypeError, ValueError, ZeroDivisionError):
        return timeout_s, max_tokens
    if ratio <= 1.0:
        return timeout_s, max_tokens
    return (
        timeout_s * ratio if timeout_s else timeout_s,
        int(max_tokens * ratio) if max_tokens else max_tokens,
    )


def _runaway_retry_policy(
    retry_number: int,
    next_cap: Any,
) -> tuple[ThinkingRetryOverride, str, str]:
    """Choose the next retry's task-local thinking policy and guidance."""
    if retry_number == 1:
        try:
            cap = max(int(next_cap), 1)
        except (TypeError, ValueError):
            budget = None
        else:
            budget = max(
                int(cap * (1.0 - _RUNAWAY_EXPANDED_OUTPUT_RESERVE)),
                1,
            )
        return (
            ThinkingRetryOverride(
                mode="expanded",
                thinking_budget=budget,
                reasoning_effort="high",
            ),
            _RUNAWAY_EXPANDED_GUIDANCE,
            "retry_expanded_cap_and_thinking",
        )

    if retry_number >= _RUNAWAY_DISABLE_THINKING_AFTER:
        return (
            ThinkingRetryOverride(mode="disabled", reasoning_effort="low"),
            _RUNAWAY_DIRECT_RECOVERY_GUIDANCE,
            "retry_thinking_disabled",
        )

    try:
        cap = int(next_cap)
    except (TypeError, ValueError):
        cap = _RUNAWAY_THINKING_BUDGET_MAX * 2
    budget = max(
        _RUNAWAY_THINKING_BUDGET_MIN,
        min(_RUNAWAY_THINKING_BUDGET_MAX, max(cap, 1) // 2),
    )
    return (
        ThinkingRetryOverride(
            mode="reduced",
            thinking_budget=budget,
            reasoning_effort="low",
        ),
        _RUNAWAY_RECOVERY_GUIDANCE,
        "retry_reduced_cap_and_thinking",
    )
