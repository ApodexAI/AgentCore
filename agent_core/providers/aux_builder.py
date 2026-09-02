"""Configurable factory for profile-defined auxiliary LLM clients."""

from __future__ import annotations

# pyright: basic
import logging
from collections.abc import Callable, Mapping
from typing import Any

logger = logging.getLogger(__name__)

type ClientFactory = Callable[..., Any]
type ProviderTypeResolver = Callable[[str], str]
type SessionHeadersResolver = Callable[[str, Mapping[str, Any]], Mapping[str, str]]
type ClientDecorator = Callable[[Any, str, str], Any]
type APIKeyResolver = Callable[[Mapping[str, Any], str], str]

_DUMMY_KEY_WARNED: set[tuple[str, str]] = set()


def _resolve_api_key(section: Mapping[str, Any], provider: str) -> str:
    key = section.get("api_key") or ""
    if key and str(key).strip():
        return str(key)
    model = str(section.get("model") or "")
    cache_key = (provider, model)
    if cache_key not in _DUMMY_KEY_WARNED:
        _DUMMY_KEY_WARNED.add(cache_key)
        logger.warning(
            "Aux LLM provider=%r model=%r has no api_key; using the legacy "
            "'dummy' token",
            provider or "<unknown>",
            model or "<unknown>",
        )
    return "dummy"


def _as_int(value: object) -> int:
    """Coerce a profile-supplied budget, tolerating numeric strings."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return int(float(str(value).strip().replace("_", "")))


def _thinking_budget(section: Mapping[str, Any]) -> object | None:
    for key in ("thinking_budget", "thinking_budget_tokens"):
        value = section.get(key)
        if value is not None:
            return value
    thinking = section.get("thinking")
    if isinstance(thinking, Mapping):
        for key in ("budget", "budget_tokens", "max_tokens"):
            value = thinking.get(key)
            if value is not None:
                return value
    return None


def _anthropic_thinking(section: Mapping[str, Any]) -> dict[str, Any] | None:
    raw = section.get("thinking")
    if not isinstance(raw, Mapping):
        return None
    thinking = {str(key): value for key, value in raw.items()}
    kind = str(thinking.get("type") or "").strip().lower()
    if kind in {"", "disabled", "off", "none", "false"}:
        return None
    return thinking


def _headers(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): str(item)
        for key, item in value.items()
        if item is not None
    }


class AuxLLMFactory:
    """Build OpenAI-compatible, Anthropic, or Bedrock auxiliary clients.

    Host products provide catalog resolution, concrete constructors, session
    headers, and post-build decoration. All request-shape normalization stays
    here so profile-driven DAG/report/summary clients cannot drift.
    """

    def __init__(
        self,
        *,
        openai_factory: ClientFactory,
        anthropic_factory: ClientFactory,
        provider_type: ProviderTypeResolver,
        session_headers: SessionHeadersResolver | None = None,
        decorate: ClientDecorator | None = None,
        api_key_resolver: APIKeyResolver = _resolve_api_key,
    ) -> None:
        self._openai_factory = openai_factory
        self._anthropic_factory = anthropic_factory
        self._provider_type = provider_type
        self._session_headers = session_headers
        self._decorate = decorate
        self._api_key_resolver = api_key_resolver

    def build(self, section: Mapping[str, Any]) -> Any:
        provider = str(
            section.get("_provider_label") or section.get("provider") or "",
        )
        provider_type = self._provider_type(provider).lower()
        model = str(section.get("model") or "")
        if provider_type in {"anthropic", "bedrock"}:
            if "claude" not in model.strip().lower():
                raise ValueError(
                    f"provider {provider!r} uses {provider_type!r} transport, "
                    f"which requires a Claude model; got {model!r}",
                )
            client = self._build_anthropic(
                section,
                provider,
                bedrock=provider_type == "bedrock",
            )
        else:
            client = self._build_openai(section, provider)
        if self._decorate is not None:
            client = self._decorate(client, provider, model)
        return client

    def _build_anthropic(
        self,
        section: Mapping[str, Any],
        provider: str,
        *,
        bedrock: bool,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": section["model"],
            "api_key": self._api_key_resolver(section, provider),
            "temperature": section.get("temperature", 0.0),
            "timeout": float(section.get("llm_timeout_s", 120)),
            "thinking": _anthropic_thinking(section),
            "effort": str(section.get("effort") or ""),
            "bedrock": bedrock,
        }
        if section.get("base_url"):
            kwargs["base_url"] = section["base_url"]
        default_headers = _headers(section.get("extra_headers"))
        if default_headers:
            kwargs["default_headers"] = default_headers
        maximum = section.get("max_completion_tokens") or section.get("max_tokens")
        if maximum is not None:
            kwargs["max_tokens"] = int(maximum)
        return self._anthropic_factory(**kwargs)

    def _build_openai(
        self,
        section: Mapping[str, Any],
        provider: str,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": section["model"],
            "api_key": self._api_key_resolver(section, provider),
            "base_url": section.get("base_url") or None,
            "temperature": section.get("temperature", 0.0),
            "timeout": float(section.get("llm_timeout_s", 120)),
        }
        maximum = section.get("max_completion_tokens") or section.get("max_tokens")
        if maximum is not None:
            kwargs["max_completion_tokens"] = int(maximum)

        extra_body_value = section.get("extra_body")
        extra_body = dict(extra_body_value) if isinstance(extra_body_value, Mapping) else {}
        template_value = extra_body.get("chat_template_kwargs")
        template = dict(template_value) if isinstance(template_value, Mapping) else {}
        # Distinguish "absent" from an explicit false: models such as Qwen3 and
        # SGLang default thinking on, so a profile disabling it must emit the
        # key rather than fall through silently.
        enable_thinking = section.get("enable_thinking")
        if enable_thinking is not None:
            template["enable_thinking"] = bool(enable_thinking)
            if enable_thinking:
                template.setdefault("preserve_thinking", False)
        if enable_thinking is None or enable_thinking:
            budget = _thinking_budget(section)
            if budget is not None:
                template["thinking_budget"] = _as_int(budget)
        if template:
            extra_body["chat_template_kwargs"] = template
        if extra_body:
            kwargs["extra_body"] = extra_body

        default_headers: dict[str, str] = {}
        if self._session_headers is not None:
            default_headers.update(self._session_headers(provider, section))
        default_headers.update(_headers(section.get("extra_headers")))
        if default_headers:
            kwargs["default_headers"] = default_headers
        return self._openai_factory(**kwargs)


__all__ = ["AuxLLMFactory"]
