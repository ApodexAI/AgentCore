from __future__ import annotations

from typing import Any

import pytest

from agent_core.providers.aux_builder import AuxLLMFactory


def _capture(kind: str, calls: list[tuple[str, dict[str, Any]]]):
    def build(**kwargs: Any) -> dict[str, Any]:
        calls.append((kind, kwargs))
        return kwargs

    return build


def test_openai_aux_request_shape_and_host_hooks() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    factory = AuxLLMFactory(
        openai_factory=_capture("openai", calls),
        anthropic_factory=_capture("anthropic", calls),
        provider_type=lambda _provider: "openai_compat",
        session_headers=lambda _provider, section: {
            "x-session": str(section.get("session_suffix", "")),
        },
        decorate=lambda client, provider, _model: {**client, "provider": provider},
    )
    client = factory.build({
        "provider": "gateway",
        "model": "qwen",
        "api_key": "key",
        "enable_thinking": True,
        "thinking_budget": 123,
        "session_suffix": ":dag",
        "extra_headers": {"X-Inspection": "on"},
        "max_tokens": 99,
    })
    assert calls[0][0] == "openai"
    assert client["provider"] == "gateway"
    assert client["max_completion_tokens"] == 99
    assert client["default_headers"] == {
        "x-session": ":dag",
        "X-Inspection": "on",
    }
    assert client["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": True,
        "preserve_thinking": False,
        "thinking_budget": 123,
    }


def test_anthropic_and_bedrock_request_shape() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    factory = AuxLLMFactory(
        openai_factory=_capture("openai", calls),
        anthropic_factory=_capture("anthropic", calls),
        provider_type=lambda _provider: "bedrock",
    )
    result = factory.build({
        "provider": "aws",
        "model": "global.anthropic.claude-sonnet",
        "api_key": "key",
        "thinking": {"type": "adaptive"},
        "effort": "high",
        "max_completion_tokens": 500,
    })
    assert calls[0][0] == "anthropic"
    assert result["bedrock"] is True
    assert result["thinking"] == {"type": "adaptive"}
    assert result["max_tokens"] == 500


def test_native_anthropic_rejects_non_claude_model() -> None:
    factory = AuxLLMFactory(
        openai_factory=lambda **_kwargs: None,
        anthropic_factory=lambda **_kwargs: None,
        provider_type=lambda _provider: "anthropic",
    )
    with pytest.raises(ValueError, match="requires a Claude model"):
        factory.build({"provider": "anthropic", "model": "qwen"})


def test_explicit_enable_thinking_false_is_emitted() -> None:
    """Qwen3/SGLang default thinking on: disabling it must send the key."""
    calls: list[tuple[str, dict[str, Any]]] = []
    factory = AuxLLMFactory(
        openai_factory=_capture("openai", calls),
        anthropic_factory=_capture("anthropic", calls),
        provider_type=lambda _provider: "openai_compat",
    )
    client = factory.build({
        "provider": "gateway",
        "model": "qwen3",
        "api_key": "key",
        "enable_thinking": False,
        # A budget without an enable flag must not leak through.
        "thinking_budget": 512,
    })
    template = client["extra_body"]["chat_template_kwargs"]
    assert template == {"enable_thinking": False}


def test_absent_enable_thinking_leaves_the_key_unset() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    factory = AuxLLMFactory(
        openai_factory=_capture("openai", calls),
        anthropic_factory=_capture("anthropic", calls),
        provider_type=lambda _provider: "openai_compat",
    )
    client = factory.build({
        "provider": "gateway",
        "model": "qwen3",
        "api_key": "key",
        "thinking_budget": 512,
    })
    template = client["extra_body"]["chat_template_kwargs"]
    assert "enable_thinking" not in template
    assert template["thinking_budget"] == 512


def test_numeric_string_thinking_budget_is_coerced() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []
    factory = AuxLLMFactory(
        openai_factory=_capture("openai", calls),
        anthropic_factory=_capture("anthropic", calls),
        provider_type=lambda _provider: "openai_compat",
    )
    client = factory.build({
        "provider": "gateway",
        "model": "qwen3",
        "api_key": "key",
        "enable_thinking": True,
        "thinking": {"budget_tokens": "1024"},
    })
    template = client["extra_body"]["chat_template_kwargs"]
    assert template["thinking_budget"] == 1024
