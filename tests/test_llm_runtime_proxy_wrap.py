"""Regression for the proxy-wrapped-400 transient handling.

OpenAI-compatible proxies (new-api, etc.) sometimes package an upstream
5xx or timeout as a 400 envelope. The literal HTTP status is 400 but the
semantics are transient — sleeping + retrying the same key fixes it.

Until this fix, ``call_llm`` treated every 400 as non-transient and
returned ``None`` immediately, surfacing the failure to the caller as
"sub-agent's LLM call failed; report is partial." Observed in
``temp/2026-05-12_heavy_mode_sdk_multiturn.md`` §"Side observations".

The fix delegates the classification to
``agent_core.infra.retriable.is_transient_network``, which both this
call site and ``workflows/heavy_mode/utils/provider_chain.py`` share.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agent_core.llm import LLMResponse
from agent_core.messages import user_msg
from agent_core.runtime.loop.llm_client import LLMCallExhausted, call_llm


class _ProxyWrapped400(Exception):
    """Mimics the shape new-api / OpenAI-proxy gateways surface.

    Carries ``status_code=400`` (literal HTTP status from the proxy) but
    the body string flags it as the proxy's own wrap of an upstream
    failure.
    """

    def __init__(self) -> None:
        super().__init__(
            "Error code: 400 - {'error': {'message': '(request id: foo)', "
            "'type': 'new_api_error', 'code': 'bad_response_status_code'}}",
        )
        self.status_code = 400


class _Vanilla400(Exception):
    """Genuine bad-request: schema validation, malformed JSON, etc.

    These ARE non-transient — must continue to return ``None`` without
    retry."""

    def __init__(self) -> None:
        super().__init__(
            "Error code: 400 - invalid_request_error: schema validation failed",
        )
        self.status_code = 400


@pytest.mark.asyncio
async def test_proxy_wrapped_400_is_retried(monkeypatch):
    """A 400 carrying ``bad_response_status_code`` must trigger backoff +
    retry, NOT immediate ``return None``."""
    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def _record(duration):
        sleeps.append(duration)
        await real_sleep(0)

    monkeypatch.setattr("asyncio.sleep", _record)

    call_count = 0

    async def _chat(_messages, **_kw):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _ProxyWrapped400()
        return LLMResponse(content="recovered")

    fake_llm = SimpleNamespace(chat=_chat)

    result = await call_llm(
        fake_llm,
        [user_msg("hi")],
        timeout=10,
        max_retries=3,
        turn=0,
    )

    assert result is not None
    assert result.content == "recovered"
    assert call_count == 2  # initial fail → 1 backoff → success
    backoffs = [s for s in sleeps if s > 0]
    assert backoffs, f"expected a backoff sleep on proxy 400, got {sleeps!r}"


@pytest.mark.asyncio
async def test_vanilla_400_surfaces_as_LLMCallExhausted(monkeypatch):
    """Schema-level 400s remain non-transient — must surface immediately
    so a chain wrapper can decide whether to advance to the next leg
    (different provider may have stricter / looser schema) instead of
    burning retries on a deterministic failure."""
    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def _record(duration):
        sleeps.append(duration)
        await real_sleep(0)

    monkeypatch.setattr("asyncio.sleep", _record)

    call_count = 0

    async def _chat(_messages, **_kw):
        nonlocal call_count
        call_count += 1
        raise _Vanilla400()

    fake_llm = SimpleNamespace(chat=_chat)

    with pytest.raises(LLMCallExhausted) as exc_info:
        await call_llm(
            fake_llm,
            [user_msg("hi")],
            timeout=10,
            max_retries=3,
            turn=0,
        )

    assert exc_info.value.reason == "non_transient"
    assert isinstance(exc_info.value.last_exc, _Vanilla400)
    # No retry — call_count is 1 and no real backoff was scheduled.
    assert call_count == 1
    backoffs = [s for s in sleeps if s > 0]
    assert not backoffs, (
        f"vanilla 400 should not trigger backoff, got {backoffs!r}"
    )


@pytest.mark.asyncio
async def test_401_403_404_classification(monkeypatch):
    """The 400-transient escape hatch is narrowly scoped to 400 + body
    pattern. 401 / 403 / 404 each surface immediately without same-key
    retry, but split on whether a chain advance might recover:

    - **401** → ``chain_advance``. The auth_failure predicate matches
      (``\\b401\\b``) and ``is_retriable_with_fallback`` includes it
      since ``501e2c645`` — the next chain leg uses a different key
      (often a different provider), so retrying *there* can succeed
      where same-key retry cannot.
    - **403** → ``non_transient``. Explicitly excluded from
      ``is_auth_failure`` because 403 means "authenticated but not
      authorised", typically a scoping / region / model-access
      problem that recurs on sibling providers — surface to the
      operator instead of silently advancing.
    - **404** → ``non_transient``. A bare "Error code: 404" carries
      no ``model_not_found`` / ``no_such_model`` signal, so the
      chain-advance predicate doesn't match — falls through to the
      generic 400-family non-transient branch.

    Either way ``call_count == 1`` — neither classification retries on
    the same key.
    """
    real_sleep = asyncio.sleep

    async def _no_op(_duration):
        await real_sleep(0)

    monkeypatch.setattr("asyncio.sleep", _no_op)

    cases = [
        (401, "chain_advance"),
        (403, "non_transient"),
        (404, "non_transient"),
    ]
    for status, expected_reason in cases:
        call_count = 0

        class _AuthError(Exception):
            pass

        err = _AuthError(f"Error code: {status}")
        err.status_code = status  # type: ignore[attr-defined]

        async def _chat(_messages, _e=err, **_kw):
            nonlocal call_count
            call_count += 1
            raise _e

        fake_llm = SimpleNamespace(chat=_chat)
        with pytest.raises(LLMCallExhausted) as exc_info:
            await call_llm(
                fake_llm,
                [user_msg("hi")],
                timeout=10,
                max_retries=3,
                turn=0,
            )
        assert exc_info.value.reason == expected_reason, (
            f"status {status} should be classified as {expected_reason}, "
            f"got {exc_info.value.reason}"
        )
        assert call_count == 1, (
            f"status {status} should not retry; got {call_count} calls"
        )
