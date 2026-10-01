"""Failed Responses results retain error codes for the runtime's retry routing."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agent_core.errors import LLMError
from agent_core.messages import user_msg
from agent_core.providers import openai_responses as rc
from agent_core.runtime.loop.llm_client import LLMCallExhausted, call_llm
from agent_core.runtime.retriable import is_retriable_with_fallback, is_transient_network

ERROR_CODES = [
    ("rate_limit_exceeded", 429),
    ("server_error", 500),
    ("vector_store_timeout", 504),
    ("invalid_prompt", 400),
    ("data_residency_mismatch", 400),
    ("bio_policy", 400),
    ("invalid_image", 400),
    ("invalid_image_format", 400),
    ("invalid_base64_image", 400),
    ("invalid_image_url", 400),
    ("image_too_large", 400),
    ("image_too_small", 400),
    ("image_parse_error", 400),
    ("image_content_policy_violation", 400),
    ("invalid_image_mode", 400),
    ("image_file_too_large", 400),
    ("unsupported_image_media_type", 400),
    ("empty_image_file", 400),
    ("failed_to_download_image", 400),
    ("image_file_not_found", 400),
    ("future_error", None),
    ("", None),
    (None, None),
]


@pytest.mark.parametrize("code,status", ERROR_CODES)
@pytest.mark.parametrize("as_object", [False, True])
def test_failed_output_retains_error_code_and_status(code, status, as_object):
    error = {"code": code, "message": "original failure message"}
    raw = {"status": "failed", "error": error, "output": [{"type": "function_call"}]}
    if as_object:
        raw = SimpleNamespace(**{**raw, "error": SimpleNamespace(**error)})

    with pytest.raises(LLMError) as caught:
        rc._parse_responses_output(raw)

    assert caught.value.code == (code or "")
    assert caught.value.status_code == status
    assert "original failure message" in str(caught.value)
    if code:
        assert code in str(caught.value)


@pytest.mark.parametrize("error", [None, {}, {"code": "invalid_prompt", "message": ""}])
def test_failed_output_without_message_uses_fallback(error):
    with pytest.raises(LLMError, match="Responses request failed") as caught:
        rc._parse_responses_output({"status": "failed", "error": error})
    assert caught.value.code == (error or {}).get("code", "")
    assert caught.value.status_code == (400 if error else None)


ROUTES = [
    (
        code, "request failed",
        "chain_advance" if code == "image_content_policy_violation"
        else "non_transient" if status == 400 else "exhausted",
        30 if status == 429 else 2,
        status in (500, 504),
    )
    for code, status in ERROR_CODES
] + [
    ("bio_policy", "Your request was blocked for safety reasons", "chain_advance", 2, False),
    ("invalid_prompt", "We've limited access to this content for safety reasons",
     "chain_advance", 2, False),
    ("invalid_prompt", "upstream timeout", "exhausted", 2, True),
    ("server_error", "server overloaded", "chain_advance", 2, False),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("code,message,reason,base,transient", ROUTES)
async def test_failed_output_routes_through_call_llm(
    monkeypatch, code, message, reason, base, transient,
):
    with pytest.raises(LLMError) as caught:
        rc._parse_responses_output({
            "status": "failed", "error": {"code": code, "message": message},
        })
    error = caught.value
    sleeps = []
    real_sleep = asyncio.sleep

    async def record_sleep(duration):
        sleeps.append(duration)
        await real_sleep(0)

    monkeypatch.setattr("asyncio.sleep", record_sleep)
    calls = 0

    async def chat(_messages, **_kwargs):
        nonlocal calls
        calls += 1
        raise error

    with pytest.raises(LLMCallExhausted) as exhausted:
        await call_llm(
            SimpleNamespace(chat=chat), [user_msg("hi")],
            timeout=10, max_retries=3, turn=0, chain_fallback_active=lambda: True,
        )

    assert exhausted.value.reason == reason
    assert exhausted.value.last_exc is error
    backoffs = [duration for duration in sleeps if duration > 0]
    if reason == "exhausted":
        assert calls == 3
        assert len(backoffs) == 2
        for duration, expected in zip(backoffs, [base, base * 2], strict=True):
            assert expected * 0.75 <= duration <= expected * 1.25
    else:
        assert calls == 1
        assert backoffs == []
    assert is_transient_network(error) is transient
    assert is_retriable_with_fallback(error) is (reason == "chain_advance")


@pytest.mark.asyncio
@pytest.mark.parametrize("code,status", ERROR_CODES)
async def test_stream_failed_event_raises_same_error(code, status):
    raw = SimpleNamespace(
        status="failed", error=SimpleNamespace(code=code, message="stream failure"),
    )

    async def events():
        yield SimpleNamespace(type="response.output_text.delta", delta="partial")
        yield SimpleNamespace(type="response.failed", response=raw)
        yield SimpleNamespace(type="response.completed", response=SimpleNamespace())

    async def create(**kwargs):
        assert kwargs["stream"] is True
        return events()

    client = rc.OpenAIResponsesClient("gpt-x", api_key="x")
    client._client = SimpleNamespace(responses=SimpleNamespace(create=create))
    deltas = []
    with pytest.raises(LLMError) as streamed:
        async for delta in client.stream([user_msg("hi")]):
            deltas.append(delta)
    with pytest.raises(LLMError) as parsed:
        rc._parse_responses_output(raw)

    assert len(deltas) == 1 and deltas[0].content == "partial"
    assert type(streamed.value) is type(parsed.value)
    assert streamed.value.code == (code or "")
    assert streamed.value.status_code == status
    assert str(streamed.value) == str(parsed.value)
