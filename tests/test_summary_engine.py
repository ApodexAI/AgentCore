from __future__ import annotations

from agent_core.providers.summary import (
    build_summary_payload,
    describe_summary_candidates,
    normalize_summary_endpoint,
    truncate_summary_fallback,
)


def test_summary_endpoint_normalization() -> None:
    assert normalize_summary_endpoint("https://host/v1/") == (
        "https://host/v1/chat/completions"
    )
    assert normalize_summary_endpoint("https://host/v1/chat/completions") == (
        "https://host/v1/chat/completions"
    )


def test_summary_payload_dialects() -> None:
    assert "max_completion_tokens" in build_summary_payload("gpt-5", "prompt")
    qwen = build_summary_payload("qwen-3", "prompt")
    assert qwen["chat_template_kwargs"] == {"enable_thinking": False}
    assert qwen["temperature"] == 1.0


def test_candidate_description_redacts_api_key() -> None:
    rendered = describe_summary_candidates([{
        "endpoint": "https://host/v1/chat/completions",
        "model": "model",
        "provider": "provider",
        "api_key": "top-secret-key",
    }])
    assert "top-secret-key" not in rendered
    assert "len=14" in rendered


def test_truncate_fallback() -> None:
    assert truncate_summary_fallback("short", 10) == "short"
    assert truncate_summary_fallback("01234567890", 10).startswith("0123456789")


import asyncio  # noqa: E402
from typing import Any  # noqa: E402

import httpx  # noqa: E402
import pytest  # noqa: E402

from agent_core.providers import summary as summary_mod  # noqa: E402
from agent_core.providers.summary import (  # noqa: E402
    SummaryLLMEngine,
    default_summary_retryable,
)


class _Response:
    def __init__(self, status: int, body: str) -> None:
        self.status_code = status
        self.text = body
        self._body = body

    def json(self) -> Any:
        import json

        return json.loads(self._body)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"{self.status_code}",
                request=httpx.Request("POST", "https://host/v1/chat/completions"),
                response=httpx.Response(self.status_code, text=self._body),
            )


def _install(monkeypatch: pytest.MonkeyPatch, responses: list[_Response]) -> list[int]:
    posts: list[int] = []

    class FakeClient:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *_args: Any) -> bool:
            return False

        async def post(self, *_args: Any, **_kwargs: Any) -> _Response:
            posts.append(1)
            return responses[min(len(posts) - 1, len(responses) - 1)]

    monkeypatch.setattr(summary_mod.httpx, "AsyncClient", FakeClient)
    return posts


def test_permanent_status_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bad key must not burn max_retries attempts on every candidate."""
    posts = _install(monkeypatch, [_Response(401, '{"error":"bad key"}')])
    sleeps: list[float] = []

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    engine = SummaryLLMEngine(sleep=sleep, fallback_limit=10)
    out = asyncio.run(
        engine.summarize(
            "some long content",
            "focus",
            [
                {"endpoint": "https://a/v1/chat/completions", "model": "m1"},
                {"endpoint": "https://b/v1/chat/completions", "model": "m2"},
            ],
        ),
    )
    # One request per candidate, no backoff, then the truncation fallback.
    assert len(posts) == 2
    assert sleeps == []
    assert out.endswith("[Content truncated...]")


def test_transient_status_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    posts = _install(monkeypatch, [_Response(503, "overloaded")])
    sleeps: list[float] = []

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    engine = SummaryLLMEngine(max_retries=3, sleep=sleep, fallback_limit=10)
    asyncio.run(
        engine.summarize(
            "content",
            "focus",
            [{"endpoint": "https://a/v1/chat/completions", "model": "m1"}],
        ),
    )
    assert len(posts) == 3
    assert sleeps == [1.0, 2.0]


def test_successful_summary_records_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    body = (
        '{"choices":[{"message":{"content":"the summary"}}],'
        '"usage":{"prompt_tokens":11,"completion_tokens":5,'
        '"prompt_tokens_details":{"cached_tokens":7}}}'
    )
    _install(monkeypatch, [_Response(200, body)])
    recorded: list[dict[str, Any]] = []
    engine = SummaryLLMEngine(usage_recorder=lambda **kw: recorded.append(kw))
    out = asyncio.run(
        engine.summarize(
            "content",
            "focus",
            [{
                "endpoint": "https://a/v1/chat/completions",
                "model": "m1",
                "provider": "prov",
                "api_key": "k",
            }],
        ),
    )
    assert out == "the summary"
    assert recorded == [{
        "model": "m1",
        "provider": "prov",
        "prompt_tokens": 11,
        "completion_tokens": 5,
        "cache_read_tokens": 7,
    }]


def test_context_length_truncation_uses_the_full_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The retry ladder must measure the original content, not the shortened one."""
    posts = _install(
        monkeypatch,
        [_Response(400, "maximum context length exceeded")],
    )
    engine = SummaryLLMEngine(
        max_retries=4,
        truncate_step=40_960,
        sleep=lambda _d: _noop(),
        fallback_limit=10,
    )
    asyncio.run(
        engine.summarize(
            "x" * 100_000,
            "focus",
            [{"endpoint": "https://a/v1/chat/completions", "model": "m1"}],
        ),
    )
    # 40 960 / 81 920 both fit inside 100 000; the third step (122 880) does not.
    assert len(posts) == 3


async def _noop() -> None:
    return None


def test_default_retryable_classification() -> None:
    def err(status: int) -> httpx.HTTPStatusError:
        return httpx.HTTPStatusError(
            "boom",
            request=httpx.Request("POST", "https://h/"),
            response=httpx.Response(status),
        )

    assert not default_summary_retryable(err(401))
    assert not default_summary_retryable(err(404))
    assert default_summary_retryable(err(429))
    assert default_summary_retryable(err(503))
    assert default_summary_retryable(TimeoutError("timeout"))
