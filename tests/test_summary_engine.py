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
