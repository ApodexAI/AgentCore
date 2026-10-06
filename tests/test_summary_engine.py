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


def test_error_response_body_is_logged(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A bare "400 Bad Request" hides why the provider rejected the call."""
    reason = '{"error":{"message":"invalid image_url in content"}}'
    _install(monkeypatch, [_Response(400, reason)])

    async def sleep(_delay: float) -> None:
        return None

    engine = SummaryLLMEngine(sleep=sleep, fallback_limit=10)
    with caplog.at_level("WARNING", logger=summary_mod.__name__):
        asyncio.run(
            engine.summarize(
                "content",
                "focus",
                [{"endpoint": "https://a/v1/chat/completions", "model": "m1"}],
            ),
        )
    assert any(
        "HTTP 400 response body" in r.getMessage() and reason in r.getMessage()
        for r in caplog.records
    )


def test_error_body_excerpt_truncates() -> None:
    long = "x" * (summary_mod._ERROR_BODY_LOG_LIMIT + 50)
    out = summary_mod._error_body_excerpt(httpx.Response(400, text=long))
    assert out.endswith("[truncated 50 chars]")
    assert summary_mod._error_body_excerpt(httpx.Response(400, text="")) == "(empty)"


@pytest.mark.asyncio
async def test_slow_drip_body_hits_total_timeout_and_advances_candidate(monkeypatch):
    requests, cancelled, closed = [], [], []
    original_client = httpx.AsyncClient

    class Drip(httpx.AsyncByteStream):
        async def __aiter__(self):
            try:
                for _ in range(100):
                    yield b" "
                    await asyncio.sleep(0.01)
                yield b'{"choices":[{"message":{"content":"too late"}}]}'
            finally:
                cancelled.append(True)

        async def aclose(self):
            closed.append(True)

    def respond(request):
        requests.append(str(request.url))
        if request.url.host == "slow":
            return httpx.Response(200, stream=Drip())
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    def client(**kwargs):
        return original_client(**kwargs, transport=httpx.MockTransport(respond))

    monkeypatch.setattr(summary_mod.httpx, "AsyncClient", client)
    engine = SummaryLLMEngine(request_timeout=0.03, max_retries=1)
    result = await asyncio.wait_for(engine.summarize("content", "focus", [
        {"endpoint": "https://slow/chat/completions", "model": "slow"},
        {"endpoint": "https://fast/chat/completions", "model": "fast"},
    ]), 0.3)
    assert result == "ok" and len(requests) == 2
    assert cancelled == [True] and closed == [True]


@pytest.mark.asyncio
async def test_summary_external_cancellation_does_not_retry(monkeypatch):
    started, closed = asyncio.Event(), asyncio.Event()
    posts = []

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            closed.set()

        async def post(self, *args, **kwargs):
            posts.append(True)
            started.set()
            await asyncio.Event().wait()

    monkeypatch.setattr(summary_mod.httpx, "AsyncClient", Client)
    task = asyncio.create_task(SummaryLLMEngine(max_retries=3).summarize(
        "content", "focus", [{"endpoint": "https://slow", "model": "slow"}],
    ))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(closed.wait(), 0.2)
    assert posts == [True]
