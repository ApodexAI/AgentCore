from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from agent_core.model_capabilities import (
    MODEL_CAPABILITIES,
    ModelCapabilities,
    normalize_thinking_mode,
    resolve_model_capabilities,
)
from agent_core.providers.anthropic import AnthropicClient
from agent_core.providers.protocol_client import build_protocol_client
from agent_core.runtime.loop.model_profile import ModelProfile


@pytest.mark.parametrize("model,effort", [("claude-fable-5-1", "high"), ("claude-opus-5-5", "medium")])
def test_current_model_facts_are_shared_with_profiles(model, effort):
    capabilities = resolve_model_capabilities(model, protocol="anthropic")
    profile = ModelProfile(model_id=model, provider="anthropic", protocol="anthropic", context_window=64_000)
    assert profile.request_capabilities is capabilities
    assert profile.context_window == 64_000  # operational budget stays host-owned
    assert capabilities.thinking_required is True
    assert capabilities.default_effort == effort
    assert capabilities.source_urls and capabilities.verified_on == "2026-10-05"
    assert capabilities.max_output_tokens == 128_000
    assert capabilities.thinking_signature_binding == "conversation_prefix"


@pytest.mark.parametrize("model,protocol", [
    ("claude-opus-5-5-next", "anthropic"),
    ("gateway-alias", "anthropic"),
    ("claude-opus-5-5", "chat_completions"),
    ("gpt-5.5", "responses"),
])
def test_unknown_and_other_protocols_do_not_inherit_native_rules(model, protocol):
    capabilities = resolve_model_capabilities(model, protocol=protocol)
    assert capabilities == ModelCapabilities()
    capabilities.validate_request(model=model, thinking={"type": "future"}, effort="custom", max_tokens=500_000)


@pytest.mark.parametrize("model", [
    "anthropic.claude-opus-5-5", "us.anthropic.claude-opus-5-5",
    "eu.anthropic.claude-opus-5-5-v1:0", "global.anthropic.claude-opus-5-5",
])
def test_bedrock_model_ids_resolve_without_rewriting_request_model(model):
    assert resolve_model_capabilities(model, protocol="bedrock") is MODEL_CAPABILITIES["claude-opus-5-5"]
    assert resolve_model_capabilities(model, protocol="anthropic") == ModelCapabilities()


def test_overrides_distinguish_unknown_from_unsupported_and_are_local():
    baseline = resolve_model_capabilities("claude-opus-5-5", protocol="anthropic")
    overrides = {"thinking_modes": ["enabled", "disabled"], "thinking_required": False,
                 "effort_levels": [], "default_effort": None, "max_output_tokens": None}
    patched = resolve_model_capabilities("claude-opus-5-5", protocol="anthropic", overrides=overrides)
    assert patched.thinking_modes == frozenset({"enabled", "disabled"})
    assert patched.effort_levels == frozenset()
    assert patched.max_output_tokens is None
    assert patched.overridden_fields == frozenset(overrides)
    assert resolve_model_capabilities("claude-opus-5-5", protocol="anthropic") is baseline
    overrides["thinking_modes"].append("adaptive")
    assert "adaptive" not in patched.thinking_modes
    with pytest.raises(FrozenInstanceError):
        patched.thinking_required = True
    with pytest.raises(TypeError):
        MODEL_CAPABILITIES["custom"] = patched


@pytest.mark.parametrize("overrides", [
    {"typo": True}, {"effort_levels": "high"}, {"effort_levels": [1]},
    {"thinking_required": "false"}, {"max_output_tokens": True},
    {"max_input_tokens": 0}, {"thinking_signature_binding": "typo"},
    {"default_effort": 5}, {"effort_levels": []},
    {"thinking_modes": ["disabled"]},
])
def test_invalid_overrides_are_rejected(overrides):
    with pytest.raises(ValueError):
        resolve_model_capabilities("claude-opus-5-5", protocol="anthropic", overrides=overrides)


@pytest.mark.parametrize("thinking,effort,limit", [
    ({"type": "disabled"}, "", 4096), ({"type": "enabled", "budget_tokens": 1024}, "", 4096),
    ({"type": ["adaptive"]}, "", 4096), (None, "minimal", 4096),
    (None, "", 128_001),
])
def test_direct_client_rejects_known_invalid_requests_before_sdk_creation(thinking, effort, limit, monkeypatch):
    import anthropic

    def unexpected_client(**kwargs):
        raise AssertionError("invalid request must fail before opening an SDK client")

    monkeypatch.setattr(anthropic, "AsyncAnthropic", unexpected_client)
    with pytest.raises(ValueError):
        AnthropicClient("claude-opus-5-5", api_key="x", thinking=thinking, effort=effort, max_tokens=limit)


@pytest.mark.parametrize("model,mode", [
    ("claude-fable-5-1", "adaptive"), ("claude-opus-5-5", "adaptive"),
    ("claude-sonnet-4-5-20250929", "enabled"), ("claude-haiku-4-5", "enabled"),
    ("claude-x", "adaptive"),
])
async def test_native_builder_uses_model_aware_defaults(model, mode):
    client = build_protocol_client({"protocol": "anthropic", "model": model, "max_tokens": 4096}, title="test")
    try:
        kwargs = client._build_kwargs([{"role": "user", "content": "hi"}], tools=None, temperature=None, max_tokens=None, extra_headers=None, timeout=None)
        assert kwargs["thinking"]["type"] == mode
        if mode == "enabled":
            assert 1024 <= kwargs["thinking"]["budget_tokens"] < kwargs["max_tokens"]
    finally:
        await client._client.close()


async def test_per_call_limit_is_validated_as_well_as_constructor_default():
    client = AnthropicClient("claude-opus-5-5", api_key="x")
    try:
        with pytest.raises(ValueError, match="max_tokens"):
            client._build_kwargs([], tools=None, temperature=None, max_tokens=128_001, extra_headers=None, timeout=None)
    finally:
        await client._client.close()


async def test_builder_override_can_disable_thinking_configuration_entirely():
    client = build_protocol_client({
        "protocol": "anthropic", "model": "claude-opus-5-5",
        "model_capabilities": {"thinking_modes": [], "thinking_required": False},
    }, title="test")
    try:
        assert client._thinking is None
    finally:
        await client._client.close()


async def test_manual_thinking_can_forward_effort_when_model_supports_it():
    client = build_protocol_client({"protocol": "anthropic", "model": "claude-opus-4-5", "effort": "medium"}, title="test")
    try:
        kwargs = client._build_kwargs([], tools=None, temperature=None, max_tokens=None, extra_headers=None, timeout=None)
        assert kwargs["thinking"]["type"] == "enabled"
        assert kwargs["extra_body"]["output_config"]["effort"] == "medium"
    finally:
        await client._client.close()


@pytest.mark.parametrize("mode", [None, "", "   "])
async def test_unset_thinking_type_uses_model_default(mode):
    client = build_protocol_client({"protocol": "anthropic", "model": "claude-haiku-4-5", "thinking_type": mode}, title="test")
    try:
        assert client._thinking["type"] == "enabled"
    finally:
        await client._client.close()


def test_auxiliary_factory_forwards_overrides_to_the_same_capability_resolver():
    from agent_core.providers.aux_builder import AuxLLMFactory

    factory = AuxLLMFactory(openai_factory=lambda **kw: kw, anthropic_factory=lambda **kw: kw,
                           provider_type=lambda _: "anthropic")
    kwargs = factory.build({"provider": "gateway", "model": "claude-opus-5-5", "api_key": "x",
                            "model_capabilities": {"max_output_tokens": 32_000}})
    assert kwargs["capabilities"].max_output_tokens == 32_000
    assert MODEL_CAPABILITIES["claude-opus-5-5"].max_output_tokens == 128_000


@pytest.mark.parametrize("thinking", [{"type": "disabled"}, {"type": "off"}, {"type": "enabled", "budget_tokens": 1024}])
def test_auxiliary_client_checks_explicit_modes_before_legacy_conversion(thinking):
    from agent_core.providers.aux_builder import AuxLLMFactory

    def unexpected_factory(**kwargs):
        raise AssertionError("configuration must fail before client construction")

    factory = AuxLLMFactory(openai_factory=unexpected_factory, anthropic_factory=unexpected_factory,
                           provider_type=lambda _: "anthropic")
    with pytest.raises(ValueError, match="thinking mode"):
        factory.build({"provider": "anthropic", "model": "claude-opus-5-5", "api_key": "x", "thinking": thinking})


def test_known_legacy_model_rejects_effort_instead_of_silently_dropping_it():
    with pytest.raises(ValueError, match="effort"):
        build_protocol_client({"protocol": "anthropic", "model": "claude-sonnet-4-5", "effort": "high"}, title="test")


async def test_profile_and_client_resolve_the_same_deployment_overrides():
    overrides = {"max_output_tokens": 32_000}
    client = build_protocol_client({
        "protocol": "anthropic", "model": "claude-opus-5-5", "max_tokens": 16_000,
        "model_capabilities": overrides,
    }, title="test")
    try:
        profile = ModelProfile(model_id="claude-opus-5-5", provider="anthropic",
                               protocol="anthropic", model_capabilities=overrides)
        assert profile.request_capabilities == client.capabilities
        assert profile.request_capabilities.max_output_tokens == 32_000
    finally:
        await client._client.close()


@pytest.mark.parametrize("spelling", ["disabled", "off", "none", "False"])
async def test_main_and_auxiliary_builders_share_disabled_spellings(spelling):
    from agent_core.providers.aux_builder import AuxLLMFactory

    client = build_protocol_client({"protocol": "anthropic", "model": "claude-haiku-4-5", "thinking_type": spelling}, title="test")
    try:
        assert client._thinking == {"type": "disabled"}
    finally:
        await client._client.close()
    factory = AuxLLMFactory(openai_factory=lambda **kw: kw, anthropic_factory=lambda **kw: kw,
                           provider_type=lambda _: "anthropic")
    kwargs = factory.build({"provider": "anthropic", "model": "claude-haiku-4-5", "api_key": "x",
                            "thinking": {"type": spelling}})
    assert kwargs["thinking"] is None


def test_main_and_auxiliary_builders_reject_the_same_unknown_mode():
    from agent_core.providers.aux_builder import AuxLLMFactory

    with pytest.raises(ValueError, match="unknown thinking type"):
        build_protocol_client({"protocol": "anthropic", "model": "claude-x", "thinking_type": "auto"}, title="test")
    factory = AuxLLMFactory(openai_factory=lambda **kw: kw, anthropic_factory=lambda **kw: kw,
                           provider_type=lambda _: "anthropic")
    with pytest.raises(ValueError, match="unknown thinking type"):
        factory.build({"provider": "anthropic", "model": "claude-x", "api_key": "x", "thinking": {"type": "auto"}})


@pytest.mark.parametrize("model_id", [
    "jp.anthropic.claude-opus-5-5", "au.anthropic.claude-opus-5-5-v1:0", "us-gov.anthropic.claude-opus-5-5",
])
def test_additional_bedrock_regional_prefixes_resolve(model_id):
    assert resolve_model_capabilities(model_id, protocol="bedrock") is MODEL_CAPABILITIES["claude-opus-5-5"]


@pytest.mark.parametrize("value,expected", [
    (None, None), ("", None), ("   ", None),
    (" Adaptive ", "adaptive"), ("ENABLED", "enabled"),
    ("disabled", "disabled"), ("Off", "disabled"), ("none", "disabled"), ("false", "disabled"),
])
def test_normalize_thinking_mode_accepts_documented_spellings(value, expected):
    assert normalize_thinking_mode(value) == expected


@pytest.mark.parametrize("value", ["auto", "on", "true", "budget"])
def test_normalize_thinking_mode_rejects_unknown_values(value):
    with pytest.raises(ValueError, match="unknown thinking type"):
        normalize_thinking_mode(value)


@pytest.mark.parametrize("value", [False, 0, {"type": "adaptive"}])
def test_normalize_thinking_mode_rejects_non_strings(value):
    with pytest.raises(ValueError, match="must be a string"):
        normalize_thinking_mode(value)


def test_opus_4_5_alias_and_dated_id_share_one_record():
    alias = MODEL_CAPABILITIES["claude-opus-4-5"]
    assert alias is MODEL_CAPABILITIES["claude-opus-4-5-20251101"]
    assert alias.effort_levels == frozenset({"low", "medium", "high"})
    assert alias.thinking_modes == frozenset({"enabled", "disabled"})
    assert "https://platform.claude.com/docs/en/build-with-claude/effort" in alias.source_urls


@pytest.mark.parametrize("model_id", [
    "anthropic.claude-opus-5-5", "us.anthropic.claude-opus-5-5-v1:0",
    "eu.anthropic.claude-opus-5-5", "apac.anthropic.claude-opus-5-5",
    "global.anthropic.claude-opus-5-5",
])
def test_documented_bedrock_prefixes_still_resolve(model_id):
    assert resolve_model_capabilities(model_id, protocol="bedrock") is MODEL_CAPABILITIES["claude-opus-5-5"]


@pytest.mark.parametrize("model_id", [
    "xx.anthropic.claude-opus-5-5",
    "arn:aws:bedrock:us-east-1:123:inference-profile/us.anthropic.claude-opus-5-5-v1:0",
    "jp.anthropic.claude-opus-5-5",  # bedrock form, but resolved over the native protocol
])
def test_unrecognized_bedrock_forms_stay_unknown(model_id):
    protocol = "anthropic" if model_id.startswith("jp.") else "bedrock"
    assert resolve_model_capabilities(model_id, protocol=protocol) == ModelCapabilities()


async def test_per_call_limit_uses_the_deployment_override():
    client = build_protocol_client({
        "protocol": "anthropic", "model": "claude-opus-5-5", "max_tokens": 16_000,
        "model_capabilities": {"max_output_tokens": 32_000},
    }, title="test")
    try:
        ok = client._build_kwargs([], tools=None, temperature=None, max_tokens=32_000, extra_headers=None, timeout=None)
        assert ok["max_tokens"] == 32_000
        with pytest.raises(ValueError, match="max_tokens=32001"):
            client._build_kwargs([], tools=None, temperature=None, max_tokens=32_001, extra_headers=None, timeout=None)
    finally:
        await client._client.close()


async def test_per_call_default_limit_is_validated_against_the_record():
    client = AnthropicClient("claude-opus-5-5", api_key="x", max_tokens=128_000)
    try:
        kwargs = client._build_kwargs([], tools=None, temperature=None, max_tokens=None, extra_headers=None, timeout=None)
        assert kwargs["max_tokens"] == 128_000
    finally:
        await client._client.close()


def test_profile_prefers_an_explicit_record_over_override_mapping():
    record = resolve_model_capabilities("claude-opus-5-5", protocol="anthropic", overrides={"max_output_tokens": 8_000})
    profile = ModelProfile(model_id="claude-opus-5-5", provider="anthropic", protocol="anthropic",
                           capabilities=record, model_capabilities={"max_output_tokens": 64_000})
    assert profile.request_capabilities is record


def test_profile_rejects_malformed_overrides_like_the_client():
    profile = ModelProfile(model_id="claude-opus-5-5", provider="anthropic", protocol="anthropic",
                           model_capabilities={"max_output_tokens": -1})
    with pytest.raises(ValueError, match="positive integer"):
        _ = profile.request_capabilities
    with pytest.raises(ValueError, match="positive integer"):
        build_protocol_client({"protocol": "anthropic", "model": "claude-opus-5-5",
                               "model_capabilities": {"max_output_tokens": -1}}, title="test")


def test_profile_overrides_do_not_apply_to_other_protocols_facts():
    profile = ModelProfile(model_id="claude-opus-5-5", provider="gateway", protocol="chat_completions",
                           model_capabilities={"max_output_tokens": 32_000})
    caps = profile.request_capabilities
    assert caps.max_output_tokens == 32_000
    assert caps.thinking_modes is None  # native facts are not inherited over chat completions
    assert caps.overridden_fields == frozenset({"max_output_tokens"})


def test_descriptive_fields_are_exposed_and_overridable():
    caps = resolve_model_capabilities("claude-fable-5-1", protocol="anthropic", overrides={
        "tool_choice_modes": ["auto", "none", "any"], "sampling_parameters": None,
        "thinking_signature_binding": "model", "max_input_tokens": 200_000,
    })
    assert caps.tool_choice_modes == frozenset({"auto", "none", "any"})
    assert caps.sampling_parameters is None
    assert caps.thinking_signature_binding == "model"
    assert caps.max_input_tokens == 200_000
    # Descriptive facts never affect request validation.
    caps.validate_request(model="claude-fable-5-1", thinking={"type": "adaptive"}, effort="high", max_tokens=128_000)
