"""Unit tests for the reporter-fallback error classifier.

Pure-function tests on ``Exception`` instances. Mirrors the patterns
provider gateway's prod sees: Anthropic 529 overloaded_error, OpenAI 503
overloads, ``insufficient_quota`` from billing-exhausted keys, 429
rate-limits, and structural errors (which must NOT trigger fallback).
"""

from __future__ import annotations

import pytest

from agent_core.runtime.retriable import (
    classify_error,
    is_credit_exhausted,
    is_empty_completion,
    is_overloaded_error,
    is_rate_limited,
    is_retriable_with_fallback,
    is_stream_stall,
)

# ── is_overloaded_error ──────────────────────────────────────────────


@pytest.mark.parametrize("msg", [
    "Error code: 529 - Anthropic API is overloaded",
    "overloaded_error: capacity exceeded",
    "anthropic.OverloadedError: Overloaded",
    "OpenAI 503: model is overloaded, please retry",
    "Provider returned capacity exhausted",
])
def test_overload_signatures_detected(msg: str) -> None:
    assert is_overloaded_error(RuntimeError(msg))


@pytest.mark.parametrize("msg", [
    "invalid_request_error: messages.0.content must be a string",
    "rate_limit_error: too many requests",  # different signal — 429
    "schema validation failed",
    "json.JSONDecodeError: Expecting value",
    "service_unavailable",  # deliberately NOT matched (proxy noise)
])
def test_non_overload_signatures_not_matched(msg: str) -> None:
    assert not is_overloaded_error(RuntimeError(msg))


def test_overload_detected_from_status_code_attribute() -> None:
    """SDK exceptions often carry ``status_code`` separately from str()."""
    class _FakeAnthropicError(Exception):
        status_code = 529

        def __str__(self) -> str:
            return "request failed"

    assert is_overloaded_error(_FakeAnthropicError())


# ── is_credit_exhausted ──────────────────────────────────────────────


@pytest.mark.parametrize("msg", [
    "insufficient_quota: your credit is exhausted",
    "billing_error: account balance is insufficient",
    "402 Payment Required",
    "Insufficient Balance",
])
def test_credit_signatures_detected(msg: str) -> None:
    assert is_credit_exhausted(RuntimeError(msg))


def test_credit_not_confused_with_overload() -> None:
    assert not is_credit_exhausted(RuntimeError("overloaded_error"))


# ── is_rate_limited ──────────────────────────────────────────────────


@pytest.mark.parametrize("msg", [
    "rate_limit_error: too many requests",
    "Error code: 429 - rate limit exceeded",
    "RateLimitError: tokens per minute exceeded",
])
def test_rate_limit_signatures_detected(msg: str) -> None:
    assert is_rate_limited(RuntimeError(msg))


# ── is_retriable_with_fallback (union) ───────────────────────────────


def test_overload_is_retriable() -> None:
    assert is_retriable_with_fallback(RuntimeError("529 overloaded"))


def test_credit_is_retriable() -> None:
    assert is_retriable_with_fallback(RuntimeError("insufficient_quota"))


def test_rate_limit_is_retriable() -> None:
    # Per spec §5: rate_limit triggers backoff-same-key, NOT chain escalation.
    assert not is_retriable_with_fallback(RuntimeError("429 rate_limit"))


def test_stream_stall_is_retriable_with_fallback() -> None:
    err = RuntimeError(
        "LLMStreamStalled: stream stalled: no chunks for 180s",
    )
    assert is_stream_stall(err)
    assert is_retriable_with_fallback(err)
    assert classify_error(err) == "stream_stall"


def test_structural_error_not_retriable() -> None:
    """Bad JSON / schema mismatch should never trigger fallback —
    retrying with another key won't fix the request shape."""
    assert not is_retriable_with_fallback(
        ValueError("Expected str, got int at field 'response'"),
    )
    assert not is_retriable_with_fallback(
        RuntimeError("invalid_request_error: schema validation"),
    )


# ── is_empty_completion (reasoning-runaway / no-content recovery) ────


@pytest.mark.parametrize("msg", [
    "No generation chunks were returned",
    "no generation chunk returned",
    "No completion tokens returned",
    "empty completion",
])
def test_empty_completion_detected(msg: str) -> None:
    assert is_empty_completion(RuntimeError(msg))


def test_empty_completion_not_overmatched() -> None:
    # A normal validation error must not look like an empty completion.
    assert not is_empty_completion(ValueError("schema validation failed"))
    assert not is_empty_completion(RuntimeError("generation succeeded"))


def test_empty_completion_is_retriable_with_fallback() -> None:
    """The 2026-05-29 fix: a reasoning-runaway empty completion (LangChain's
    bare ``ValueError('No generation chunks were returned')``) must advance
    the chain instead of being fatal — otherwise one empty among a heavy
    run's ~150 calls kills the whole run."""
    assert is_retriable_with_fallback(
        ValueError("No generation chunks were returned"),
    )


# ── classify_error ───────────────────────────────────────────────────


def test_classify_returns_specific_reason() -> None:
    assert (
        classify_error(ValueError("No generation chunks were returned"))
        == "empty_completion"
    )
    assert classify_error(RuntimeError("529 overloaded")) == "overloaded"
    assert classify_error(RuntimeError("insufficient_quota")) == "credit_exhausted"
    assert classify_error(RuntimeError("429 too many requests")) == "rate_limited"
    assert classify_error(RuntimeError("validation failed")) == "other"


def test_classify_precedence_overload_before_credit() -> None:
    """If both signals are present, ``overloaded`` wins — it's the
    more specific provider-state signal."""
    err = RuntimeError("529 overloaded; also insufficient_quota mentioned")
    assert classify_error(err) == "overloaded"


# ── is_context_length_error (NEW per spec §5) ────────────────────────


@pytest.mark.parametrize("msg", [
    "context length exceeded",
    "context_length_exceeded",
    "this model's maximum context length is 200000 tokens",
    "input is longer than the model can handle",
    "maximum context size reached",
])
def test_context_length_signatures_detected(msg: str) -> None:
    from agent_core.runtime.retriable import is_context_length_error
    assert is_context_length_error(RuntimeError(msg))


@pytest.mark.parametrize("msg", [
    "rate_limit_error: too many requests",
    "overloaded_error: capacity exceeded",
    "json parse failed",
])
def test_context_length_not_misclassified(msg: str) -> None:
    from agent_core.runtime.retriable import is_context_length_error
    assert not is_context_length_error(RuntimeError(msg))


# ── is_retriable_with_fallback no longer matches rate_limit ──────────


def test_rate_limit_is_NOT_retriable_with_fallback() -> None:
    """Per spec §5 decision table: rate_limit means backoff, same key —
    do NOT escalate the chain layer."""
    from agent_core.runtime.retriable import is_retriable_with_fallback
    err = RuntimeError("rate_limit_error: too many requests")
    assert not is_retriable_with_fallback(err)


def test_overload_still_retriable_with_fallback() -> None:
    from agent_core.runtime.retriable import is_retriable_with_fallback
    err = RuntimeError("anthropic.OverloadedError: Overloaded")
    assert is_retriable_with_fallback(err)


def test_credit_exhausted_still_retriable_with_fallback() -> None:
    from agent_core.runtime.retriable import is_retriable_with_fallback
    err = RuntimeError("insufficient_quota")
    assert is_retriable_with_fallback(err)


def test_classify_error_context_length() -> None:
    """`classify_error` returns a new `context_length` label."""
    from agent_core.runtime.retriable import classify_error
    err = RuntimeError("context_length_exceeded")
    assert classify_error(err) == "context_length"


# ── is_transient_network ─────────────────────────────────────────────


@pytest.mark.parametrize("msg", [
    "Connection reset by peer",
    "Connection refused",
    "Connection aborted",
    "Connection error: ECONNRESET",
    "Connection closed unexpectedly",
    "asyncio.TimeoutError: timed out after 60s",
    "Read timeout",
    "504 Gateway Timeout",
    "Upstream error: provider unreachable",
    "Upstream timeout while reading response",
])
def test_transient_network_signatures_detected(msg: str) -> None:
    from agent_core.runtime.retriable import is_transient_network
    assert is_transient_network(RuntimeError(msg))


def test_transient_network_matches_proxy_wrapped_400() -> None:
    """The whole reason this predicate exists: an OpenAI-compatible proxy
    (e.g. new-api gateway) packages an upstream 5xx or timeout into a 400
    envelope. The literal HTTP status is 400 but the semantics are
    transient — sleeping + retrying the same key fixes it."""
    from agent_core.runtime.retriable import is_transient_network

    err = RuntimeError(
        "Error code: 400 - {'error': {'message': '(request id: foo)', "
        "'type': 'new_api_error', 'code': 'bad_response_status_code'}}",
    )
    # Status attribute would mark this as 400, but the body pattern wins.
    err.status_code = 400  # type: ignore[attr-defined]
    assert is_transient_network(err)


def test_transient_network_matches_5xx_without_overload() -> None:
    """A raw 502/503/504 with no overload wording is a transport problem,
    not capacity exhaustion — classify as transient so the caller sleeps
    the same key instead of burning fallback keys."""
    from agent_core.runtime.retriable import is_transient_network

    err = RuntimeError("502 Bad Gateway")
    err.status_code = 502  # type: ignore[attr-defined]
    assert is_transient_network(err)


def test_transient_network_yields_to_overload() -> None:
    """When the same 503 has ``overloaded`` in the body, overload wins —
    callers rotate keys instead of sleeping."""
    from agent_core.runtime.retriable import (
        is_overloaded_error,
        is_transient_network,
    )

    err = RuntimeError("503 Service Unavailable: model is overloaded")
    err.status_code = 503  # type: ignore[attr-defined]
    assert is_overloaded_error(err)
    assert not is_transient_network(err)


def test_transient_network_NOT_matched_by_plain_400() -> None:
    """A vanilla 400 Bad Request (schema mismatch, bad JSON, etc.) is
    structural — must NOT be classified as transient."""
    from agent_core.runtime.retriable import is_transient_network

    err = RuntimeError("Error code: 400 - invalid_request_error: bad JSON")
    err.status_code = 400  # type: ignore[attr-defined]
    assert not is_transient_network(err)


def test_transient_network_NOT_retriable_with_fallback() -> None:
    """Per spec §5: transient_network triggers backoff-same-key, NOT
    layer escalation. ``is_retriable_with_fallback`` (the chain-advance
    trigger) must therefore stay False for transient errors."""
    from agent_core.runtime.retriable import (
        is_retriable_with_fallback,
        is_transient_network,
    )

    err = RuntimeError("Connection reset")
    assert is_transient_network(err)
    assert not is_retriable_with_fallback(err)


def test_classify_error_returns_transient_network() -> None:
    from agent_core.runtime.retriable import classify_error
    err = RuntimeError("Connection reset by peer")
    assert classify_error(err) == "transient_network"


def test_classify_precedence_overload_before_transient() -> None:
    """503 + ``overloaded`` keeps the overloaded label, not transient."""
    from agent_core.runtime.retriable import classify_error
    err = RuntimeError("503 model is overloaded")
    err.status_code = 503  # type: ignore[attr-defined]
    assert classify_error(err) == "overloaded"


# ── is_safety_filter (Aliyun DataInspectionFailed + friends) ─────────


@pytest.mark.parametrize("msg", [
    "<400> ***.***.DataInspectionFailed: Input text data may contain "
    "inappropriate content. (request id: 202605121426525099564225sJgS6e6)",
    "data_inspection_failed",
    "content_policy_violation: prompt blocked",
    "content_filter: rejected",
    "content_filtered",
    "input_filtered: anthropic policy",
    "output_filtered",
    "the request contains inappropriate content",
    "prompt_blocked by safety system",
    # gpt-5.x mid-stream refusal family (added 2026-05-29 for the
    # reporter_v2 mid-stream continuation path):
    "Invalid prompt: we've limited access to this content for safety "
    "reasons. This type of information may be used to benefit or to "
    "harm people...",
    "Your request was blocked by our safety system",
    "This violates our usage policies",
    # OpenRouter content moderation blocks:
    "Request blocked: content moderation policy",
    "content moderation",
    "blocked by safety system",
    "moderation policy",
    "Request blocked by safety system",
])
def test_safety_filter_signatures_detected(msg: str) -> None:
    from agent_core.runtime.retriable import is_safety_filter
    assert is_safety_filter(RuntimeError(msg)), f"missed: {msg!r}"


@pytest.mark.parametrize("msg", [
    "rate_limit_error: too many requests",
    "529 overloaded_error",
    "insufficient_quota",
    "context_length_exceeded",
    "Connection reset by peer",
    "validation failed",
])
def test_safety_filter_not_misclassified(msg: str) -> None:
    """Capacity/quota/network errors must NOT trip the safety predicate
    — those route through their own branches with different recovery."""
    from agent_core.runtime.retriable import is_safety_filter
    assert not is_safety_filter(RuntimeError(msg))


def test_safety_filter_is_retriable_with_fallback() -> None:
    """Per design: same-key retry on a deterministic safety rejection is
    hopeless, so the chain must advance to a different provider."""
    from agent_core.runtime.retriable import is_retriable_with_fallback
    err = RuntimeError(
        "data_inspection_failed: Input text data may contain inappropriate content"
    )
    assert is_retriable_with_fallback(err)


def test_classify_safety_filter_label() -> None:
    from agent_core.runtime.retriable import classify_error
    err = RuntimeError("data_inspection_failed")
    assert classify_error(err) == "safety_filter"


def test_classify_precedence_safety_before_overload() -> None:
    """If both signals appear in the same envelope, ``safety_filter``
    wins — operators need to distinguish 'model refused' from 'overload'
    on the dashboard."""
    from agent_core.runtime.retriable import classify_error
    err = RuntimeError(
        "529 overloaded_error; also data_inspection_failed in body"
    )
    assert classify_error(err) == "safety_filter"


# ── is_model_unavailable (distributor "no available channel" / 503 model_not_found) ──


@pytest.mark.parametrize("msg", [
    # The exact shape from the 2026-05-16 heavy-mode e2e against
    # api.miromind.site:
    "Error code: 503 - {'error': {'code': 'model_not_found', 'message': "
    "'No available channel for model claude-sonnet-4-6 under group "
    "openrouter (distributor) (request id: foo)', 'type': 'new_api_error'}}",
    "model_not_found",
    "no_such_model: gpt-foo",
    "no available channel for model claude-bar under group anthropic",
    "model_not_supported by this provider",
    "unsupported_model: provider doesn't host this canonical name",
    # OpenRouter shape — chaos scenario 02, 2026-05-21:
    "openai.BadRequestError: Error code: 400 - {'error': {'message': "
    "'does-not-exist-chaos-9999 is not a valid model ID', 'code': 400}}",
    "invalid model ID",
    "unknown model: gpt-foo-9999",
])
def test_model_unavailable_signatures_detected(msg: str) -> None:
    from agent_core.runtime.retriable import is_model_unavailable
    assert is_model_unavailable(RuntimeError(msg)), f"missed: {msg!r}"


@pytest.mark.parametrize("msg", [
    "rate_limit_error: too many requests",
    "529 overloaded_error",
    "insufficient_quota",
    "context_length_exceeded",
    "Connection reset by peer",
    "validation failed",
    "Error code: 503 - service unavailable",  # bare 503 without model phrase
])
def test_model_unavailable_not_misclassified(msg: str) -> None:
    """Other 5xx / structural errors must not trip model_unavailable —
    they have their own recovery paths (backoff, salvage, raise)."""
    from agent_core.runtime.retriable import is_model_unavailable
    assert not is_model_unavailable(RuntimeError(msg))


def test_model_unavailable_is_retriable_with_fallback() -> None:
    """The whole reason this predicate exists: the next leg in the chain
    may host the model — advance instead of same-key retry."""
    from agent_core.runtime.retriable import is_retriable_with_fallback
    err = RuntimeError(
        "Error code: 503 - model_not_found: No available channel for model X"
    )
    assert is_retriable_with_fallback(err)


def test_model_unavailable_yields_transient_network() -> None:
    """A 503 model_not_found is also a 5xx, but we want chain advance
    rather than same-key backoff. ``is_transient_network`` must defer."""
    from agent_core.runtime.retriable import (
        is_model_unavailable,
        is_transient_network,
    )
    err = RuntimeError(
        "Error code: 503 - model_not_found: No available channel"
    )
    err.status_code = 503  # type: ignore[attr-defined]
    assert is_model_unavailable(err)
    assert not is_transient_network(err)


def test_classify_model_unavailable_label() -> None:
    from agent_core.runtime.retriable import classify_error
    err = RuntimeError("model_not_found")
    assert classify_error(err) == "model_unavailable"


def test_classify_precedence_model_unavailable_before_overload() -> None:
    """Operator wants 'fix the chain config' vs 'wait it out' distinction."""
    from agent_core.runtime.retriable import classify_error
    err = RuntimeError("529 overloaded_error; also model_not_found in body")
    assert classify_error(err) == "model_unavailable"


# ── is_auth_failure (chaos-discovered gap, 2026-05-21) ────────────────
# Background: scenarios/01 of scripts/chaos_heavy_mode.py set
# OPENROUTER_API_KEY to garbage and expected the chain to advance.
# Instead the run crashed exit=1 because the resulting
# openai.AuthenticationError ("Missing Authentication header", 401)
# wasn't in is_retriable_with_fallback. These tests pin the four wire
# shapes (SDK class name, OpenRouter body, invalid_api_key body, bare
# 401) and verify chain-advance + classify_error all agree.


@pytest.mark.parametrize("msg", [
    "openai.AuthenticationError: Error code: 401",
    "Error code: 401 - {'error': {'message': 'Missing Authentication header', 'code': 401}}",
    "invalid_api_key: the provided key is malformed",
    "invalid_authentication: token rejected",
    "401 Unauthorized",
    "unauthenticated",
    "authentication_failed",
])
def test_auth_failure_signatures_detected(msg: str) -> None:
    from agent_core.runtime.retriable import is_auth_failure
    assert is_auth_failure(RuntimeError(msg))


def test_auth_failure_detected_from_sdk_class_name() -> None:
    """openai-python raises ``AuthenticationError`` as a class — observed
    via ``type(err).__name__`` even when the body is sanitised."""
    from agent_core.runtime.retriable import is_auth_failure

    class AuthenticationError(Exception):
        pass

    assert is_auth_failure(AuthenticationError("rejected"))


def test_auth_failure_detected_from_status_code() -> None:
    """Status attribute fallback — wrappers sometimes strip the body
    but keep the status."""
    from agent_core.runtime.retriable import is_auth_failure

    class _Err(Exception):
        status_code = 401

        def __str__(self) -> str:
            return "request failed"

    assert is_auth_failure(_Err())


@pytest.mark.parametrize("msg", [
    "rate_limit_error: too many requests",  # 429, not 401
    "insufficient_quota",  # billing, not auth
    "model_not_found",
    "internal server error",
    "schema validation failed",
])
def test_auth_failure_not_misclassified(msg: str) -> None:
    from agent_core.runtime.retriable import is_auth_failure
    assert not is_auth_failure(RuntimeError(msg))


def test_403_forbidden_is_NOT_auth_failure() -> None:
    """403 means "key authenticated but not authorised for this resource"
    (e.g. account scoped to specific model families). The root often
    repeats on sibling providers in the same chain, so advancing isn't
    useful — surface to the operator instead."""
    from agent_core.runtime.retriable import is_auth_failure

    class _Err(Exception):
        status_code = 403

        def __str__(self) -> str:
            return "403 Forbidden: insufficient permissions"

    assert not is_auth_failure(_Err())


def test_auth_failure_is_retriable_with_fallback() -> None:
    """The whole point of the chaos finding: 401 must trigger chain
    advance so the next leg's (different key, often different provider)
    can take over. Reporter L1 OpenRouter 401 → L2 anthropic-direct →
    L3 fallback_models[0] must walk."""
    from agent_core.runtime.retriable import is_retriable_with_fallback
    err = RuntimeError(
        "Error code: 401 - {'error': {'message': 'Missing Authentication "
        "header', 'code': 401}}",
    )
    assert is_retriable_with_fallback(err)


def test_classify_returns_auth_failure_label() -> None:
    from agent_core.runtime.retriable import classify_error
    err = RuntimeError(
        "Error code: 401 - {'error': {'message': 'Missing Authentication "
        "header', 'code': 401}}",
    )
    assert classify_error(err) == "auth_failure"


def test_classify_precedence_model_unavailable_before_auth_failure() -> None:
    """If a chain returns both 401 AND model_not_found (e.g. distributor
    proxy rejected the model name before checking the key),
    ``model_unavailable`` wins — the actionable fix is the chain config,
    not the credentials."""
    from agent_core.runtime.retriable import classify_error
    err = RuntimeError("401 unauthorized; also model_not_found in body")
    assert classify_error(err) == "model_unavailable"
