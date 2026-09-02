"""Per-profile ``protocol`` → native LLM client selection.

Workflow LLM builders read ``llm.protocol`` and, for the reasoning-capturing
protocols, hand off here instead of building the default ``OpenAIClient``:

- ``anthropic``   → native Anthropic Messages API + extended thinking
  (:class:`~agent_core.providers.anthropic.AnthropicClient`).
- ``responses``   → OpenAI Responses API + encrypted reasoning
  (:class:`~agent_core.providers.openai_responses.OpenAIResponsesClient`).
- ``chat_completions`` (default / anything else) → returns ``None`` so the
  caller keeps its existing Chat Completions ``OpenAIClient`` path unchanged.

Both reasoning protocols return content as a typed BLOCK LIST → the model
profile's ``thinking_format`` is ``content_block`` (see
:func:`thinking_format_for_protocol`) so the parser keeps the verbatim blocks
(signatures / encrypted_content) for faithful multi-turn replay.
"""

from __future__ import annotations

# pyright: basic, reportPrivateImportUsage=false
import logging
from typing import Any, get_args

from agent_core.llm import LLMClient

# Single source of truth for the valid ``llm.protocol`` values. Duplicating the
# set here would let the two drift, which is how a protocol becomes buildable
# but has no ``thinking_format`` (or vice versa).
from agent_core.runtime.loop.model_profile import (
    ThinkingFormat,
    WireProtocol,
    is_wire_protocol,
)

logger = logging.getLogger(__name__)

# Anthropic's floor for ``thinking.budget_tokens`` on the legacy ``enabled``
# shape. The API also requires ``budget_tokens < max_tokens``, so the two
# constraints together make any ``max_tokens <= 1024`` unsatisfiable.
_MIN_THINKING_BUDGET = 1024


def protocol_of(cfg: dict[str, Any]) -> WireProtocol:
    """Normalised ``llm.protocol`` (default ``chat_completions``).

    An unusable value — wrong type, or a string that names no known protocol —
    normalises to ``chat_completions`` and is LOGGED. The value comes from
    hand-written YAML, and a typo (``anthropc``) otherwise degrades in total
    silence: :func:`build_protocol_client` returns ``None``,
    :func:`thinking_format_for_protocol` returns ``None``, and the profile ends
    up on a Chat Completions client with tag-format thinking. Requests keep
    succeeding while the signed reasoning is quietly dropped, and nothing points
    at the profile. An ABSENT or empty value is the documented default and is
    not logged.
    """
    raw = cfg.get("protocol")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return "chat_completions"
    if isinstance(raw, str):
        lowered = raw.lower()
        if is_wire_protocol(lowered):
            return lowered
    logger.warning(
        "unknown llm.protocol %r for model %r; falling back to "
        "chat_completions. Valid values: %s",
        raw, cfg.get("model", "?"), ", ".join(sorted(get_args(WireProtocol))),
    )
    return "chat_completions"


def provider_label(cfg: dict[str, Any]) -> str:
    """Usage-attribution provider label for a profile ``llm`` block.

    Prefers the explicit ``_provider_label`` / ``provider``; otherwise DERIVES
    it from ``protocol`` so native reasoning profiles that omit ``provider``
    aren't mislabelled ``openai`` in billing/usage traces: ``anthropic`` →
    ``anthropic``, ``bedrock`` → ``bedrock``, ``responses`` / ``chat_completions``
    → ``openai`` (both are OpenAI-wire)."""
    explicit = cfg.get("_provider_label") or cfg.get("provider")
    if explicit:
        return str(explicit)
    proto = protocol_of(cfg)
    if proto == "anthropic":
        return "anthropic"
    if proto == "bedrock":
        return "bedrock"
    return "openai"


def thinking_format_for_protocol(protocol: str) -> ThinkingFormat | None:
    """``content_block`` for the reasoning protocols, else ``None`` (caller
    falls back to explicit YAML / model-id inference).

    Both this and :func:`protocol_of` return the NARROW alias, not a bare
    ``str``: their outputs feed ``ModelProfile.thinking_format`` /
    ``ModelProfile.protocol``, which are ``Literal`` unions, so widening to
    ``str`` makes a value this module just validated unassignable at the very
    field it was validated for. This file is checked at ``basic`` level, so
    nothing here catches that — a strict host does."""
    return (
        "content_block"
        if protocol in ("anthropic", "responses", "bedrock")
        else None
    )


def _effort_str(cfg: dict[str, Any]) -> str:
    effort = cfg.get("effort")
    return effort.strip() if isinstance(effort, str) else ""


def _build_anthropic(cfg: dict[str, Any], *, bedrock: bool = False) -> LLMClient:
    """Native Anthropic Messages API with extended thinking.

    Responses carry thinking + signature blocks, kept verbatim
    (raw_content_blocks) and replayed unmodified. ``temperature`` is OMITTED
    (the client drops it when thinking is on). ``base_url`` posts to
    ``{base_url}/v1/messages`` (direct) or ``{base_url}/model/{id}/invoke``
    (``bedrock=True``, AWS Bedrock runtime, Bearer API-key auth + the
    ``anthropic_version`` body stamp). Optional ``effort`` →
    ``output_config.effort``.

    ``thinking_type`` selects the request shape (default ``adaptive``). Live-
    verified against api.anthropic.com + Bedrock 2026-07-09 (see
    ``temp/2026-07-09_reasoning-protocol-live-verification.md``); matches the
    official matrix at platform.claude.com/docs/en/build-with-claude/adaptive-thinking:

    - ``adaptive`` (DEFAULT) — the RECOMMENDED mode for all current Claude
      (Opus 4.6/4.7/4.8, Sonnet 4.6/5, Fable/Mythos), and the ONLY mode on the
      newest (Opus 4.7/4.8, Sonnet 5) — ``enabled`` is rejected there with 400.
      Emits ``thinking={"type":"adaptive"}`` + ``thinking_display`` (default
      ``summarized`` so the readable thinking text is captured; the newest
      models default ``display`` to ``omitted`` = empty ``thinking`` field with
      the ``signature`` still present for replay). ``effort`` is forwarded only
      in this mode (it is an adaptive-only knob; the oldest models 400 on
      ``enabled``+effort).
    - ``enabled`` — LEGACY opt-in for models older than Opus 4.6 / Sonnet 4.6
      (Sonnet 4.5, Opus 4.5, …), which reject ``adaptive`` with 400. Emits
      ``thinking={"type":"enabled","budget_tokens":N}`` (``N`` from
      ``thinking_budget_tokens``, default 8192, clamped to ``[1024, max_tokens-1]``
      since Anthropic requires ``budget_tokens < max_tokens``). When the
      configured ``max_tokens`` is too small for the 1024 floor, ``max_tokens``
      is RAISED to ``budget + 1`` rather than emitting an invalid pair — see
      :func:`_enabled_thinking_budget`. ``effort`` is
      NOT sent (budget_tokens is the control knob here; oldest models 400 on it).
      Deprecated on Opus 4.6 / Sonnet 4.6 per Anthropic.
    """
    from agent_core.providers.anthropic import AnthropicClient

    max_tokens = int(cfg.get("max_tokens", 32768))
    ttype = str(cfg.get("thinking_type", "adaptive")).strip().lower()
    if ttype == "enabled":
        budget, max_tokens = _enabled_thinking_budget(
            int(cfg.get("thinking_budget_tokens", 8192)), max_tokens,
        )
        thinking: dict[str, Any] = {"type": "enabled", "budget_tokens": budget}
        effort = ""
    else:
        thinking = {"type": "adaptive"}
        display = cfg.get("thinking_display", "summarized")
        if isinstance(display, str) and display.strip():
            thinking["display"] = display.strip()
        effort = _effort_str(cfg)
    return AnthropicClient(
        model=cfg["model"],
        api_key=cfg.get("api_key", "dummy"),
        base_url=cfg.get("base_url") or None,
        max_tokens=max_tokens,
        thinking=thinking,
        effort=effort,
        bedrock=bedrock,
    )


def _enabled_thinking_budget(budget: int, max_tokens: int) -> tuple[int, int]:
    """Reconcile ``thinking.budget_tokens`` with ``max_tokens``.

    Anthropic imposes two constraints on the legacy ``enabled`` thinking shape:
    ``budget_tokens >= 1024`` and ``budget_tokens < max_tokens``. They are
    jointly unsatisfiable whenever ``max_tokens <= 1024``, so a naive
    ``max(1024, min(budget, max_tokens - 1))`` clamp silently emits a pair the
    API rejects — ``max_tokens=512`` produced ``budget_tokens=1024``, i.e. a
    budget LARGER than the response cap, and a 400 on every call.

    The floor wins, because it is the API's own minimum and cannot be
    negotiated: when ``max_tokens`` cannot accommodate it, ``max_tokens`` is
    raised to ``budget + 1`` and the adjustment logged. Returns the
    ``(budget, max_tokens)`` pair to send.
    """
    budget = max(_MIN_THINKING_BUDGET, min(budget, max_tokens - 1))
    if budget >= max_tokens:
        raised = budget + 1
        logger.warning(
            "thinking_type=enabled requires budget_tokens (%d) < max_tokens, "
            "but max_tokens is %d; raising max_tokens to %d. Set max_tokens "
            "above %d in the profile to silence this.",
            budget, max_tokens, raised, _MIN_THINKING_BUDGET,
        )
        max_tokens = raised
    return budget, max_tokens


def _build_responses(cfg: dict[str, Any], title: str) -> LLMClient:
    """OpenAI Responses API with encrypted reasoning.

    ``reasoning`` is built from ``effort`` + ``reasoning_summary`` (default
    ``auto`` → the response carries a readable reasoning summary; opt out with
    ``reasoning_summary: ""`` or set ``reasoning: {...}`` verbatim). The client
    always sends ``include=['reasoning.encrypted_content']`` + ``store=False``.
    ``temperature`` is only sent when the profile sets it (reasoning models
    reject non-default values).
    """
    from agent_core.providers.openai_responses import OpenAIResponsesClient

    reasoning = cfg.get("reasoning")
    if reasoning is None:
        reasoning = {}
        if cfg.get("effort"):
            reasoning["effort"] = cfg["effort"]
        summary = cfg.get("reasoning_summary", "auto")
        if isinstance(summary, str) and summary.strip():
            reasoning["summary"] = summary.strip()
    return OpenAIResponsesClient(
        model=cfg["model"],
        api_key=cfg.get("api_key", "dummy"),
        base_url=cfg.get("base_url"),
        temperature=cfg.get("temperature"),
        max_output_tokens=cfg.get("max_tokens", 32768),
        default_headers={"X-Title": title},
        reasoning=reasoning or None,
        store=False,
    )


def build_protocol_client(cfg: dict[str, Any], *, title: str) -> LLMClient | None:
    """Build the native client for ``cfg['protocol']``.

    Returns ``None`` for ``chat_completions`` (the caller builds its usual
    ``OpenAIClient``); an :class:`AnthropicClient` / :class:`OpenAIResponsesClient`
    otherwise. ``title`` is the ``X-Title`` header stamped on Responses calls.

    An unrecognised ``protocol`` also lands here as ``chat_completions`` →
    ``None``; :func:`protocol_of` is what logs it.
    """
    protocol = protocol_of(cfg)
    if protocol == "anthropic":
        return _build_anthropic(cfg)
    if protocol == "bedrock":
        return _build_anthropic(cfg, bedrock=True)
    if protocol == "responses":
        return _build_responses(cfg, title)
    return None


__all__ = [
    "build_protocol_client",
    "protocol_of",
    "provider_label",
    "thinking_format_for_protocol",
]
