"""Typed model request constraints, separate from credentials and deployment catalogs.

None means unknown, an empty set means unsupported. Resolution is explicit and
local to a client/profile: exact model facts, then per-deployment host overrides.
No network access or mutable process-wide registration is performed here.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal, cast

WireProtocol = Literal["chat_completions", "anthropic", "responses", "bedrock"]
ThinkingMode = Literal["adaptive", "enabled", "disabled"]
SignatureBinding = Literal["none", "model", "conversation_prefix"]


@dataclass(frozen=True)
class ModelCapabilities:
    """Request facts for one model, resolved for one client/profile.

    Enforced before API calls by :meth:`validate_request`: ``thinking_modes``,
    ``thinking_required``, ``effort_levels`` and ``max_output_tokens``.
    Descriptive only, for hosts to read: ``default_effort`` (an omitted effort
    keeps the provider default), ``sampling_parameters`` (the Anthropic adapter
    omits sampling for every model regardless), ``tool_choice_modes`` (no
    adapter sends ``tool_choice`` yet), ``thinking_signature_binding`` and
    ``max_input_tokens`` (``ModelProfile.context_window`` stays the operational
    budget). A field moves to the enforced list only together with the adapter
    code that consumes it.
    """

    thinking_modes: frozenset[str] | None = None
    thinking_required: bool | None = None
    effort_levels: frozenset[str] | None = None
    default_effort: str | None = None
    sampling_parameters: frozenset[str] | None = None
    tool_choice_modes: frozenset[str] | None = None
    thinking_signature_binding: SignatureBinding | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    source_urls: tuple[str, ...] = ()
    verified_on: str = ""
    overridden_fields: frozenset[str] = frozenset()

    def validate_request(
        self, *, model: str, thinking: Mapping[str, object] | None,
        effort: str = "", max_tokens: int | None = None,
    ) -> None:
        """Reject known unsupported settings; leave unknown capabilities alone."""
        mode = thinking.get("type") if thinking is not None else None
        if mode is not None and self.thinking_modes is not None and (not isinstance(mode, str) or mode not in self.thinking_modes):
            raise ValueError(f"{model}: thinking mode {mode!r} is unsupported; use {sorted(self.thinking_modes)}")
        if mode == "disabled" and self.thinking_required is True:
            raise ValueError(f"{model}: thinking is always enabled")
        if effort and self.effort_levels is not None and effort not in self.effort_levels:
            raise ValueError(f"{model}: effort {effort!r} is unsupported; use {sorted(self.effort_levels)}")
        if max_tokens is not None and self.max_output_tokens is not None and max_tokens > self.max_output_tokens:
            raise ValueError(f"{model}: max_tokens={max_tokens} exceeds the model limit {self.max_output_tokens}")


_CURRENT_CLAUDE = ModelCapabilities(
    thinking_modes=frozenset({"adaptive"}), thinking_required=True,
    effort_levels=frozenset({"low", "medium", "high", "xhigh", "max"}),
    default_effort="high", sampling_parameters=frozenset(),
    tool_choice_modes=frozenset({"auto", "none"}),
    thinking_signature_binding="conversation_prefix",
    max_input_tokens=1_000_000, max_output_tokens=128_000,
    source_urls=(
        "https://platform.claude.com/docs/en/models/fable-5-1/migration-guide",
        "https://platform.claude.com/docs/en/models/fable-5-1/overview",
        "https://platform.claude.com/docs/en/build-with-claude/effort",
    ), verified_on="2026-10-05",
)
_LEGACY_CLAUDE = ModelCapabilities(
    thinking_modes=frozenset({"enabled", "disabled"}), thinking_required=False,
    effort_levels=frozenset(),
    source_urls=("https://platform.claude.com/docs/en/build-with-claude/extended-thinking",),
    verified_on="2026-10-05",
)
# The only manual-thinking model with effort; it combines with budget_tokens.
_OPUS_4_5 = replace(
    _LEGACY_CLAUDE, effort_levels=frozenset({"low", "medium", "high"}), default_effort="high",
    source_urls=(*_LEGACY_CLAUDE.source_urls, "https://platform.claude.com/docs/en/build-with-claude/effort"),
)

# These are request facts, not an endpoint, credential, price, or routing catalog.
# Only documented exact IDs/aliases match; a new version never inherits a guessed
# capability merely because its model name starts with an older one.
MODEL_CAPABILITIES: Mapping[str, ModelCapabilities] = MappingProxyType({
    "claude-fable-5-1": _CURRENT_CLAUDE,
    "claude-opus-5-5": replace(
        _CURRENT_CLAUDE, default_effort="medium", source_urls=(
            "https://platform.claude.com/docs/en/models/opus-5-5/migration-guide",
            "https://platform.claude.com/docs/en/models/opus-5-5/overview",
            "https://platform.claude.com/docs/en/build-with-claude/effort",
        ),
    ),
    "claude-sonnet-4-5": _LEGACY_CLAUDE,
    "claude-sonnet-4-5-20250929": _LEGACY_CLAUDE,
    "claude-haiku-4-5": _LEGACY_CLAUDE,
    "claude-haiku-4-5-20251001": _LEGACY_CLAUDE,
    "claude-opus-4-5": _OPUS_4_5,
    "claude-opus-4-5-20251101": _OPUS_4_5,
})

_DISABLED_ALIASES = frozenset({"disabled", "off", "none", "false"})


def normalize_thinking_mode(value: object) -> ThinkingMode | None:
    """Normalize a configured thinking type; None/blank means "not configured".

    Shared by every native builder so one spelling means the same thing on the
    main, auxiliary, and direct paths. Unknown values raise instead of silently
    falling back to a mode the operator did not choose.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("thinking type must be a string or null")
    mode = value.strip().lower()
    if not mode:
        return None
    if mode in _DISABLED_ALIASES:
        return "disabled"
    if mode in ("adaptive", "enabled"):
        return cast(ThinkingMode, mode)
    raise ValueError(f"unknown thinking type {value!r}; use adaptive, enabled, or disabled")


_SET_FIELDS = frozenset({"thinking_modes", "effort_levels", "sampling_parameters", "tool_choice_modes"})
_LIMIT_FIELDS = frozenset({"max_input_tokens", "max_output_tokens"})
_OVERRIDE_FIELDS = _SET_FIELDS | _LIMIT_FIELDS | {
    "thinking_required", "default_effort", "thinking_signature_binding",
}


def _string_set(key: str, value: object) -> frozenset[str]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise ValueError(f"{key} must be a collection of strings or null")
    members = cast(Iterable[object], value)
    strings: set[str] = set()
    for member in members:
        if not isinstance(member, str):
            raise ValueError(f"{key} must be a collection of strings or null")
        strings.add(member)
    return frozenset(strings)


def _canonical_model_id(model_id: str, protocol: WireProtocol) -> str:
    if protocol != "bedrock":
        return model_id
    # Bedrock inference-profile regional prefixes and documented version suffix.
    # Arbitrary proxy aliases and ARNs require explicit host overrides.
    match = re.fullmatch(r"(?:(?:us|us-gov|eu|apac|jp|au|global)\.)?anthropic\.(claude-[a-z0-9-]+?)(?:-v\d+:\d+)?", model_id)
    return match.group(1) if match else model_id


def resolve_model_capabilities(
    model_id: str, *, protocol: WireProtocol,
    overrides: Mapping[str, object] | None = None,
) -> ModelCapabilities:
    """Resolve native model facts and a validated, per-deployment override.

    A Claude name served over Chat Completions does not imply native Messages
    capabilities. Proxies and custom model aliases must declare their overrides.
    Explicit None clears a known fact back to unknown.
    """
    capabilities = ModelCapabilities()
    if protocol in ("anthropic", "bedrock"):
        capabilities = MODEL_CAPABILITIES.get(_canonical_model_id(model_id, protocol), capabilities)
    if overrides is None:
        return capabilities
    unknown = set(overrides) - _OVERRIDE_FIELDS
    if unknown:
        raise ValueError(f"unknown model capability fields: {sorted(unknown)}")
    changes: dict[str, object] = {}
    for key, value in overrides.items():
        if value is None:
            changes[key] = None
        elif key in _SET_FIELDS:
            changes[key] = _string_set(key, value)
        elif key in _LIMIT_FIELDS:
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{key} must be a positive integer or null")
            changes[key] = value
        elif key == "thinking_required":
            if not isinstance(value, bool):
                raise ValueError("thinking_required must be a boolean or null")
            changes[key] = value
        elif key == "thinking_signature_binding":
            if value not in ("none", "model", "conversation_prefix"):
                raise ValueError("thinking_signature_binding must be none, model, conversation_prefix or null")
            changes[key] = value
        elif key == "default_effort":
            if not isinstance(value, str):
                raise ValueError("default_effort must be a string or null")
            changes[key] = value
    result = replace(capabilities, **changes, overridden_fields=frozenset(overrides))
    if result.default_effort is not None and result.effort_levels is not None and result.default_effort not in result.effort_levels:
        raise ValueError("default_effort must belong to effort_levels; override both fields together")
    if result.thinking_required is True and result.thinking_modes is not None and "disabled" in result.thinking_modes:
        raise ValueError("thinking_required conflicts with disabled thinking mode")
    return result


__all__ = [
    "MODEL_CAPABILITIES", "ModelCapabilities", "SignatureBinding", "ThinkingMode",
    "WireProtocol", "normalize_thinking_mode", "resolve_model_capabilities",
]
