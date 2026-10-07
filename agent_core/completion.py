"""Provider-neutral completion signals, before display/history normalization."""
from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, cast

from agent_core.llm import LLMResponse, StreamDelta


def get_recovery_hook(client: Any, name: str) -> Callable[..., Any] | None:
    """Require explicit opt-in; transparent __getattr__ must not unwrap proxies."""
    if not callable(inspect.getattr_static(client, name, None)):
        return None
    hook = getattr(client, name, None)
    return cast("Callable[..., Any]", hook) if callable(hook) else None


def _mapping(value: Any) -> dict[str, Any]:
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}


def response_rejection_reason(response: Any) -> str:
    """Return the explicit provider rejection marker, never infer from silence."""
    metadata = _mapping(getattr(response, "response_metadata", None))
    extra = _mapping(getattr(response, "additional_kwargs", None))
    for mapping in (metadata, extra):
        details = _mapping(mapping.get("stop_details"))
        if mapping.get("refusal") or (
            details.get("type") == "refusal"
        ):
            return "refusal"
    if getattr(response, "refusal", None):
        return "refusal"
    content = getattr(response, "content", None)
    if isinstance(content, list) and any(
        _mapping(block).get("type") == "refusal" for block in cast("list[Any]", content)
    ):
        return "refusal"
    reason = str(getattr(response, "finish_reason", "") or "").strip().lower()
    if reason in ("refusal", "content_filter"):
        return reason
    reason = str(metadata.get("stop_reason") or "").strip().lower()
    if reason in ("refusal", "content_filter"):
        return reason
    return ""


def _has_reported_usage(response: Any) -> bool:
    metadata = _mapping(getattr(response, "response_metadata", None))
    source = getattr(response, "usage_source", "") or (
        metadata.get("usage_source", "")
    )
    if source == "estimated":
        return False
    candidates = [getattr(response, "usage", None), getattr(response, "usage_metadata", None)]
    candidates.extend((metadata.get("token_usage"), metadata.get("usage")))
    # Real reported zeros count as a signal. Estimates and wrapper placeholders
    # do not: products should retain their provenance instead of inventing a
    # provider report. Unmarked legacy usage is treated conservatively.
    return any(
        usage and not (_mapping(usage).get("estimated"))
        for usage in candidates
    )


def is_wholly_empty_response(response: Any) -> bool:
    """No content, tool calls, reasoning, reported usage or explicit rejection."""
    if getattr(response, "tool_calls", None) or response_rejection_reason(response):
        return False
    if str(getattr(response, "reasoning_content", "") or "").strip():
        return False
    extra = _mapping(getattr(response, "additional_kwargs", None))
    if str(extra.get("reasoning_content") or "").strip():
        return False
    if _has_reported_usage(response):
        return False
    metadata = _mapping(getattr(response, "response_metadata", None))
    if metadata.get("stop_details"):
        return False
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return not content.strip()
    if isinstance(content, list):
        for block in cast("list[Any]", content):
            if isinstance(block, str):
                if block.strip():
                    return False
            elif isinstance(block, dict):
                block = cast("dict[str, Any]", block)
                if (
                    str(block.get("text") or block.get("content") or "").strip()
                    or block.get("type") not in ("text", "")
                ):
                    return False
            elif block is not None:
                return False
        return True
    return content is None


def stream_delta_has_completion_signal(delta: StreamDelta) -> bool:
    """Signals for provider wrappers that cannot assemble signed history."""
    if any(call.get("name") for call in delta.tool_call_deltas):
        return True
    return not is_wholly_empty_response(LLMResponse(
        content=delta.reasoning_blocks or delta.content,
        reasoning_content=delta.reasoning_content,
        usage=delta.usage, usage_source=delta.usage_source,
        finish_reason=delta.finish_reason,
        response_metadata={
            "stop_details": delta.stop_details, "stop_reason": delta.stop_reason,
            "refusal": delta.refusal,
        },
    ))
