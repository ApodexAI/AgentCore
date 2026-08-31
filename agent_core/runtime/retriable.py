"""Error classification for the LLM-call retry / chain-escalation machinery.

Every LLM call site — this package's ``call_llm``, a product's
provider-chain wrapper, any workflow-specific retry helper — needs to turn
a raw ``Exception`` into one of a small number of decisions:

- *Same key, sleep + retry* — transient network glitches, 429
  rate-limits, proxy-wrapped upstream blips.
- *Different key (or provider), no sleep* — overload, credit exhaustion,
  safety-filter rejection, model-not-hosted, and auth failure: a wrong or
  unauthorised key is fixed by the next leg, not by sleeping.
- *Short-circuit to salvage* — input exceeded the model's context.
- *Surface the enclosing deadline* — the run or logical call budget ended;
  neither sleeping nor rotating providers can buy more time.
- *Surface immediately* — everything left over, e.g. a malformed-request
  4xx that no key or provider can satisfy.

This module is the **single source of truth** for that classification. It
sits in the shared package rather than beside any one caller so the
generic retry loop and a product's chain helper cannot drift apart —
consistency here is what makes a sub-agent going through the generic
engine and a reporter going through the chain helper behave the same way.

Public predicates (each pure on a single ``Exception``):

- ``is_overloaded_error(err)``     — Anthropic 529, ``overloaded`` /
  ``capacity`` substrings, narrow OpenAI 503 *overload* shape.
  Deliberately does **not** match generic ``service_unavailable``
  because proxies / infra often emit that for transport problems
  unrelated to upstream capacity.
- ``is_credit_exhausted(err)``     — API key credit exhausted /
  insufficient quota; rotating to a different key fixes it.
- ``is_rate_limited(err)``         — per-key 429. Caller should
  back off on the same key, NOT escalate the chain layer.
- ``is_context_length_error(err)`` — input exceeds the model's
  context window. Caller must short-circuit to L4 salvage; no
  retry, no rotation can help.
- ``is_transient_network(err)``    — request-level transient failure
  that backoff-same-key fixes: ``timeout`` / ``timed out`` / connection
  reset, 5xx **without** an overload signature, and the common
  *proxy-wrapped* shape where an OpenAI-compatible gateway packages an
  upstream 5xx into a 400 envelope with
  ``code=bad_response_status_code`` / ``type=new_api_error``.
- ``is_safety_filter(err)``        — upstream content-moderation
  rejection (Aliyun PAI-EAS / DashScope ``DataInspectionFailed``,
  Anthropic ``input_filtered`` / ``output_filtered``, OpenAI
  ``content_policy_violation``). Same-key retry is hopeless — the
  filter is deterministic on the input — so this advances the chain
  layer if a different provider is available, otherwise surfaces.
- ``is_model_unavailable(err)``    — provider says it doesn't host
  this model (OpenRouter / new-api distributor ``model_not_found`` /
  ``no_such_model`` / ``no available channel``). The current
  provider can never serve the request; same-key retry just wastes
  time. Advance the chain to the next leg (a different distributor
  group / provider with the same canonical model) — that is what
  the chain machinery exists for.
- ``is_auth_failure(err)``         — 401 / AuthenticationError /
  ``invalid_api_key`` / ``Missing Authentication header``.
  Same-key retry can never succeed (the key is wrong / revoked /
  not whitelisted for this model). The next chain leg uses a
  different key (often a different provider entirely), so advancing
  is the only way forward. 403 ``forbidden`` is deliberately NOT in
  this set because it can mean "scope / region / model-access
  denied", which has the same root on a sibling provider.
- ``is_stream_stall(err)``         — the runtime's streaming watchdog
  already retried this endpoint/key up to its stall threshold and then
  surfaced for chain advance. Do not classify it as transient-network;
  same-key backoff would repeat the dead-stream budget.
- :class:`~agent_core.errors.LLMDeadlineExceeded` — carries either
  ``wall_deadline`` or ``logical_call_deadline`` and is deliberately excluded
  from transient-network retry.

Composed predicate:

- ``is_retriable_with_fallback(err)`` — ``overload`` OR
  ``credit_exhausted`` OR ``safety_filter`` OR ``model_unavailable``
  OR ``auth_failure`` OR ``empty_completion`` OR ``stream_stall``. The
  chain-escalation trigger a chain wrapper uses to advance L1 → L2 → L3.
  Intentionally narrower than "anything we might retry" — rate-limit and
  transient-network errors retry on the same key, so they are NOT
  included here.

Label helper:

- ``classify_error(err)`` — returns a deadline reason when given
  :class:`~agent_core.errors.LLMDeadlineExceeded`, otherwise one of ``"context_length"`` /
  ``"safety_filter"`` / ``"model_unavailable"`` / ``"auth_failure"`` /
  ``"empty_completion"`` / ``"overloaded"`` / ``"credit_exhausted"`` /
  ``"stream_stall"`` / ``"rate_limited"`` / ``"transient_network"`` /
  ``"other"`` for the ``report.fallback`` SSE payload +
  ``usage_summary.by_model[].outcome``.

All functions are pure (no I/O, no LLM). Safe to call from
anywhere — the generic engine retry loop, the chain helper, an
observer, a test.
"""

from __future__ import annotations

import re

from agent_core.errors import LLMDeadlineExceeded

_OVERLOAD_PATTERNS = (
    re.compile(r"overload", re.IGNORECASE),
    re.compile(r"capacity", re.IGNORECASE),
    re.compile(r"529", re.IGNORECASE),
    # Anthropic explicit error type from the SDK.
    re.compile(r"overloaded_error", re.IGNORECASE),
)

_CREDIT_PATTERNS = (
    re.compile(r"credit", re.IGNORECASE),
    re.compile(r"insufficient[_\s]*quota", re.IGNORECASE),
    re.compile(r"insufficient[_\s]*balance", re.IGNORECASE),
    re.compile(r"billing", re.IGNORECASE),
    re.compile(r"payment[_\s]*required", re.IGNORECASE),
    re.compile(r"\b402\b"),
)

_RATE_LIMIT_PATTERNS = (
    re.compile(r"rate[_\s]*limit", re.IGNORECASE),
    re.compile(r"\b429\b"),
)

_CONTEXT_LENGTH_PATTERNS = (
    re.compile(r"context[_\s]*length", re.IGNORECASE),
    re.compile(r"context_length_exceeded", re.IGNORECASE),
    re.compile(r"longer than the model", re.IGNORECASE),
    re.compile(r"maximum context", re.IGNORECASE),
)

# Transient network / proxy-wrap signatures. ``bad_response_status_code``
# + ``new_api_error`` are the new-api gateway's way of forwarding an
# upstream 5xx / timeout as a 400 envelope to the client — sleeping and
# retrying the same key is the right response, NOT raising a 400 as
# non-transient. Observed 2026-05-12 in multi-turn smokes against an
# OpenAI-compatible proxy.
_TRANSIENT_NETWORK_PATTERNS = (
    re.compile(r"\btimeout\b", re.IGNORECASE),
    re.compile(r"timed[\s_]*out", re.IGNORECASE),
    re.compile(
        r"connection[\s_]*(?:reset|refused|aborted|error|closed)",
        re.IGNORECASE,
    ),
    re.compile(r"bad_response_status_code", re.IGNORECASE),
    re.compile(r"new_api_error", re.IGNORECASE),
    # Upstream gateway timeouts that get text-wrapped before our status
    # extractor sees them.
    re.compile(r"gateway[\s_]*time[\s_]*out", re.IGNORECASE),
    re.compile(r"upstream[\s_]*(?:timeout|error)", re.IGNORECASE),
)

# Runtime stream watchdog. Matched by type name / message text instead of
# importing ``LLMStreamStalled`` from core to keep infra free of core imports.
_STREAM_STALL_PATTERNS = (
    re.compile(r"\bLLMStreamStalled\b"),
    re.compile(r"stream[_\s-]*stalled", re.IGNORECASE),
    re.compile(r"no chunks for", re.IGNORECASE),
)

# Upstream content-moderation rejections. Same-key retry is hopeless —
# the filter is deterministic on the same input — so these advance the
# chain to the next provider when one is configured. Patterns cover the
# four providers we have first-hand evidence of:
#
# - Aliyun PAI-EAS / DashScope wraps Qwen behind a ``数据安全检查`` layer;
#   rejection shape is 400 + ``code: data_inspection_failed`` +
#   ``type: data_inspection_failed`` (observed in a browse-comparison
#   smoke against Italian/political prompts).
# - Anthropic returns ``input_filtered`` / ``output_filtered`` blocks on
#   policy violations (rare on Claude 4.x but documented).
# - OpenAI ``content_policy_violation`` / ``content_filter`` on Azure +
#   o1/gpt-4 deployments with strict moderation enabled.
# - GPT-5.x specifically emits ``Invalid prompt: we've limited access to
#   this content for safety reasons. This type of information may be used
#   to benefit or to harm people...`` as both a pre-flight 400 and a
#   mid-stream error event (observed when reporter_v2 prompted gpt-5.x on
#   biomedical / dual-use research questions, 2026-05-29). The
#   distinctive substrings are unique enough to anchor on and they
#   survive minor wording tweaks.
_SAFETY_FILTER_PATTERNS = (
    re.compile(r"data[_\s]*inspection[_\s]*failed", re.IGNORECASE),
    re.compile(r"content[_\s]*policy[_\s]*violation", re.IGNORECASE),
    re.compile(r"content[_\s]*filter(?:ed)?", re.IGNORECASE),
    re.compile(r"input[_\s]*filtered", re.IGNORECASE),
    re.compile(r"output[_\s]*filtered", re.IGNORECASE),
    re.compile(r"inappropriate[_\s]*content", re.IGNORECASE),
    re.compile(r"prompt[_\s]*blocked", re.IGNORECASE),
    # GPT-5.x "Invalid prompt: we've limited access to this content for
    # safety reasons..." family. We match short substrings so the check
    # survives wording tweaks and translated variants.
    re.compile(r"limited[_\s]*access[_\s]*to[_\s]*this[_\s]*content[_\s]*for[_\s]*safety", re.IGNORECASE),
    re.compile(r"may[_\s]*be[_\s]*used[_\s]*to[_\s]*benefit[_\s]*or[_\s]*to[_\s]*harm", re.IGNORECASE),
    re.compile(r"violates[_\s]*our[_\s]*usage[_\s]*policies", re.IGNORECASE),
    re.compile(r"your[_\s]*request[_\s]*was[_\s]*blocked", re.IGNORECASE),
    # OpenRouter and general content moderation blocks:
    re.compile(r"content[_\s]*moderation", re.IGNORECASE),
    re.compile(r"moderation[_\s]*policy", re.IGNORECASE),
    re.compile(r"request[_\s]*blocked", re.IGNORECASE),
    re.compile(r"safety[_\s]*system", re.IGNORECASE),
)

# Provider says it doesn't host this model. Same-key retry is futile;
# advance the chain to the next leg. Patterns cover:
# - OpenRouter / new-api distributor: ``code=model_not_found`` body +
#   ``"No available channel for model ... under group ..."`` message
#   (observed in the 2026-05-16 heavy-mode e2e against
#   ``api.miromind.site``).
# - OpenAI-compatible gateways that surface a 404 / 400 with
#   ``no_such_model`` or ``model_not_supported``.
# - Anthropic: structured ``not_found_error`` body (snake_case literal in
#   ``body.type``) + the ``NotFoundError`` SDK class name surfaced via
#   ``type(err).__name__`` in :func:`_stringify` (both openai-python and
#   anthropic-python raise a ``NotFoundError`` class on 404). Patterns
#   are written precisely — a permissive ``not[_\s]*found[_\s]*error``
#   would false-match Python's builtin ``FileNotFoundError`` and silently
#   advance the chain on unrelated file-IO errors.
_MODEL_UNAVAILABLE_PATTERNS = (
    re.compile(r"model[_\s]*not[_\s]*found", re.IGNORECASE),
    re.compile(r"no[_\s]*such[_\s]*model", re.IGNORECASE),
    re.compile(r"model[_\s]*not[_\s]*supported", re.IGNORECASE),
    re.compile(r"no[_\s]*available[_\s]*channel", re.IGNORECASE),
    re.compile(r"unsupported[_\s]*model", re.IGNORECASE),
    re.compile(r"not_found_error"),
    re.compile(r"\bNotFoundError\b"),
    # OpenRouter (observed in chaos scenario 02, 2026-05-21): a 400
    # BadRequest with body ``"X is not a valid model ID"``. Same root
    # cause as model_not_found — the gateway refuses to route. Same-key
    # retry can't fix a typo'd model name; the next chain leg may use a
    # different canonical model spec and succeed.
    re.compile(r"not[_\s]*a[_\s]*valid[_\s]*model[_\s]*id", re.IGNORECASE),
    re.compile(r"\binvalid[_\s]*model[_\s]*id\b", re.IGNORECASE),
    re.compile(r"\bunknown[_\s]*model\b", re.IGNORECASE),
)

# Authentication failures from upstream providers. Discovered by a
# key-clobber chaos scenario (2026-05-21): a bare ``OPENROUTER_API_KEY``
# clobber surfaced ``openai.AuthenticationError`` with body
# ``{"error": {"message": "Missing Authentication header", "code": 401}}``.
# Without auth here, ``is_retriable_with_fallback`` returned False and the
# product's chain wrapper never
# advanced to L2 / L3 — the reporter crashed with exit 1. Real users
# hit this every time a key is revoked or scoped wrong; rotating to the
# next chain leg (different key OR different provider) is the only
# recovery, hence chain-advance is correct.
#
# Patterns cover the four wire shapes we have first-hand evidence of:
# - OpenAI / OpenRouter raw 401 body shapes (``invalid_api_key`` /
#   ``invalid_authentication`` / bare ``unauthorized``).
# - OpenRouter's specific "Missing Authentication header" surface
#   (happens when the SDK suppresses an obviously-bogus Bearer value).
# - The openai-python SDK class name surfaced via ``type(err).__name__``
#   in ``_stringify`` — both ``AuthenticationError`` (openai-python) and
#   the equivalent anthropic-python shape.
# - Bare ``401`` status code (the bottom-of-the-barrel fallback when an
#   upstream wrapper strips structured fields but keeps the status).
_AUTH_FAILURE_PATTERNS = (
    re.compile(r"\bAuthenticationError\b"),
    re.compile(r"\bauthentication[_\s]*failed", re.IGNORECASE),
    re.compile(r"\binvalid[_\s]*api[_\s]*key", re.IGNORECASE),
    re.compile(r"\binvalid[_\s]*authentication", re.IGNORECASE),
    re.compile(r"\bmissing[_\s]*authentication", re.IGNORECASE),
    re.compile(r"\bunauthorized\b", re.IGNORECASE),
    re.compile(r"\bunauthenticated\b", re.IGNORECASE),
    re.compile(r"\b401\b"),
)


def _stringify(err: BaseException) -> str:
    """Concatenate every signal an LLM SDK might surface."""
    parts: list[str] = [type(err).__name__, str(err)]
    for attr in ("status_code", "response", "body", "message"):
        val = getattr(err, attr, None)
        if val is not None:
            parts.append(str(val))
    return " | ".join(parts)


def get_status_code(err: BaseException) -> int | None:
    """Best-effort integer HTTP status extraction.

    Public so :mod:`agent_core.runtime.loop._call` can share this
    heuristic instead of keeping its own copy — both call sites need the
    exact same status-attribute lookup order.
    """
    for attr in ("status_code", "status", "code"):
        val = getattr(err, attr, None)
        if isinstance(val, int):
            return val
    return None


# Private alias for this module's own call site below.
_get_status_code = get_status_code


def is_overloaded_error(err: BaseException) -> bool:
    """True if ``err`` indicates the upstream provider is at capacity.

    Triggers fallback key rotation (the next key shares the provider so
    capacity is rarely fixed by rotation alone — but it's the cheapest
    signal we have and miroflow's prod observed that key-specific
    capacity quirks DO exist).
    """
    blob = _stringify(err)
    return any(p.search(blob) for p in _OVERLOAD_PATTERNS)


def is_credit_exhausted(err: BaseException) -> bool:
    """True if ``err`` indicates the current API key has run out of
    credit / quota. Rotating to a different key usually fixes it."""
    blob = _stringify(err)
    return any(p.search(blob) for p in _CREDIT_PATTERNS)


def is_rate_limited(err: BaseException) -> bool:
    """True if ``err`` is a per-key rate-limit (429). Rotating keys
    likely helps; backing off also helps."""
    blob = _stringify(err)
    return any(p.search(blob) for p in _RATE_LIMIT_PATTERNS)


def is_context_length_error(err: BaseException) -> bool:
    """True if the input exceeds the model's context window.

    Per spec §5 decision table: callers must short-circuit to L4
    salvage rather than retry or rotate keys — retrying will just hit
    the same wall.
    """
    blob = _stringify(err)
    return any(p.search(blob) for p in _CONTEXT_LENGTH_PATTERNS)


def is_transient_network(err: BaseException) -> bool:
    """True if ``err`` is a request-level transient (timeout / connection
    reset / upstream 5xx / proxy-wrapped upstream blip).

    Decision per spec §5: sleep + retry on the SAME key. Does NOT
    escalate chain layers — the next provider would see the same
    transient at roughly the same rate, so burning fallback keys here
    is counter-productive.

    The 5xx-without-overload branch catches the common case where a
    proxy hands back ``502 / 503 / 504`` without any overload substring
    — that's a network problem, not a capacity problem. ``is_overloaded_error``
    keeps priority so a 503 with ``overloaded_error`` in the body still
    routes to rotation rather than backoff.
    """
    if isinstance(err, LLMDeadlineExceeded):
        return False
    if is_stream_stall(err):
        return False
    if is_overloaded_error(err):
        return False
    # model_unavailable is also a 5xx (typically 503 from distributor
    # proxies) but the right response is "advance the chain", not
    # "backoff and retry same key" — same-key retries are guaranteed
    # to fail with the same model_not_found. Surrender precedence to
    # is_retriable_with_fallback here.
    if is_model_unavailable(err):
        return False
    status = _get_status_code(err)
    if status is not None and 500 <= status < 600:
        return True
    blob = _stringify(err)
    return any(p.search(blob) for p in _TRANSIENT_NETWORK_PATTERNS)


def is_stream_stall(err: BaseException) -> bool:
    """True when ``call_llm`` has surfaced a repeated stream watchdog stall.

    The watchdog has already spent the configured same-endpoint stall budget
    before this exception reaches an outer chain runner, so the correct chain
    decision is immediate key/provider advance rather than same-key backoff.
    """
    blob = _stringify(err)
    return any(p.search(blob) for p in _STREAM_STALL_PATTERNS)


def is_safety_filter(err: BaseException) -> bool:
    """True if ``err`` is an upstream content-moderation rejection.

    Retrying the same key is hopeless (filter is deterministic on the
    input). Caller should advance the chain to a different provider if
    one is configured; if not, the error surfaces to the user.
    """
    blob = _stringify(err)
    return any(p.search(blob) for p in _SAFETY_FILTER_PATTERNS)


def is_model_unavailable(err: BaseException) -> bool:
    """True if the provider doesn't host the requested model.

    Distributor proxies (OpenRouter aggregator, new-api / miromind.site
    gateway) return ``code=model_not_found`` (often with a 503 status
    when the upstream channel pool is empty) when the model name they
    received isn't routable to any backend. Same-key retry is pointless —
    the next provider leg in the chain may have a different upstream
    that DOES host the model, so advance instead.
    """
    blob = _stringify(err)
    return any(p.search(blob) for p in _MODEL_UNAVAILABLE_PATTERNS)


def is_auth_failure(err: BaseException) -> bool:
    """True if ``err`` is an upstream authentication failure (401).

    Catches the four observed shapes: ``openai.AuthenticationError`` SDK
    class, bare ``unauthorized`` text, ``invalid_api_key`` /
    ``invalid_authentication`` structured codes, and OpenRouter's
    "Missing Authentication header" wire surface. Status 403 is
    deliberately excluded — 403 means "key authenticated but not
    authorised for this resource", which often has the same root on a
    sibling provider (e.g. account scoped to specific model families).
    Surface 403 to the operator instead of silently advancing.

    Caller (chain wrapper) advances to the next leg on True. Same-key
    retry can never succeed because the rejection is deterministic on
    (current key, current model). See module docstring for the chaos
    test that discovered this gap (2026-05-21).
    """
    blob = _stringify(err)
    return any(p.search(blob) for p in _AUTH_FAILURE_PATTERNS)


# A bare ``ValueError("No generation chunks were returned")`` — the shape
# LangChain-based clients raise — arrives when an upstream stream completes
# but yields no usable *content*. That is the dominant failure shape for
# self-hosted reasoning models that run away in the ``reasoning_content``
# channel and never emit a content token before hitting ``max_tokens``.
# Observed 2026-05-29: a single empty completion among the ~150 LLM calls of
# one run was fatal because nothing classified it as recoverable, so every
# multi-call run eventually died on one. The HTTP call
# *succeeded* — this is not a timeout/network class — so it gets its own
# detector rather than folding into ``is_transient_network``.
_EMPTY_COMPLETION_PATTERNS = (
    re.compile(
        r"no[\s_]*generation[\s_]*chunks?[\s_]*(?:were[\s_]*)?returned",
        re.IGNORECASE,
    ),
    re.compile(
        r"no[\s_]*completion[\s_]*(?:tokens?|content)[\s_]*returned",
        re.IGNORECASE,
    ),
    re.compile(r"empty[\s_]*completion", re.IGNORECASE),
)


def is_empty_completion(err: BaseException) -> bool:
    """True if the upstream returned a successful response with no content.

    Distinct from a network/timeout error: the call succeeded at the HTTP
    layer but produced zero content tokens (reasoning-runaway,
    all-tokens-in-thinking, or an empty stream). Routed through
    :func:`is_retriable_with_fallback` so the caller retries the same key
    first (a temperature>0 resample frequently recovers) and then advances
    the chain to a different provider, which always recovers.
    """
    blob = _stringify(err)
    return any(p.search(blob) for p in _EMPTY_COMPLETION_PATTERNS)


def is_retriable_with_fallback(err: BaseException) -> bool:
    """The chain-escalation trigger.

    Per spec §5 decision table: overload, credit_exhausted,
    safety_filter, model_unavailable, AND auth_failure advance the
    chain layer. rate_limit + transient_network trigger backoff-same-key
    instead (handled by the caller, not this predicate). Each of these
    is deterministic on (current provider, current input) — only
    switching providers / keys can change the outcome.

    ``empty_completion`` also routes here: it isn't deterministic on the
    input (a temp>0 resample may recover), but the caller's same-key
    retry budget runs first, and advancing the chain afterwards is the
    guaranteed recovery — so it belongs to the same predicate.
    """
    return (
        is_overloaded_error(err)
        or is_credit_exhausted(err)
        or is_safety_filter(err)
        or is_model_unavailable(err)
        or is_auth_failure(err)
        or is_empty_completion(err)
        or is_stream_stall(err)
    )


def classify_error(err: BaseException) -> str:
    """Short reason label for the ``report.fallback`` SSE payload.

    Precedence (top wins):
      runtime deadline → ``context_length`` → ``safety_filter`` →
      ``model_unavailable`` →
      ``auth_failure`` → ``empty_completion`` → ``overloaded`` →
      ``credit_exhausted`` → ``stream_stall`` → ``rate_limited`` →
      ``transient_network`` → ``other``.

    Context-length wins outright because its caller behaviour differs
    (short-circuit to salvage). Safety-filter wins next because the
    operator dashboard needs to distinguish "model refused" from
    capacity issues. Model-unavailable wins over overload because the
    operator response is different — overload is "wait or fan out",
    model-unavailable is "fix the chain config". Auth-failure sits
    above overload/credit because the operator action is also a
    config fix (rotate / revoke key) — splitting it out from
    ``other`` makes dashboards immediately point at the right knob.
    ``empty_completion`` outranks overload/credit for the same reason in
    reverse: several gateways answer a capacity problem with a 200 and an
    empty body, so labelling it by its own shape keeps a silent-empty
    endpoint from being read as ordinary overload.
    """
    if isinstance(err, LLMDeadlineExceeded):
        return err.reason
    if is_context_length_error(err):
        return "context_length"
    if is_safety_filter(err):
        return "safety_filter"
    if is_model_unavailable(err):
        return "model_unavailable"
    if is_auth_failure(err):
        return "auth_failure"
    if is_empty_completion(err):
        return "empty_completion"
    if is_overloaded_error(err):
        return "overloaded"
    if is_credit_exhausted(err):
        return "credit_exhausted"
    if is_stream_stall(err):
        return "stream_stall"
    if is_rate_limited(err):
        return "rate_limited"
    if is_transient_network(err):
        return "transient_network"
    return "other"


__all__ = [
    "classify_error",
    "get_status_code",
    "is_auth_failure",
    "is_context_length_error",
    "is_credit_exhausted",
    "is_empty_completion",
    "is_model_unavailable",
    "is_overloaded_error",
    "is_rate_limited",
    "is_retriable_with_fallback",
    "is_safety_filter",
    "is_stream_stall",
    "is_transient_network",
]
