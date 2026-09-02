"""Truncation signals must reach the runaway/continuation checks as ``length``."""

from __future__ import annotations

from typing import Any

from agent_core.llm import LLMResponse
from agent_core.providers.anthropic import _to_llm_response
from agent_core.providers.finish_reason import (
    normalize_finish_reason,
    responses_finish_reason,
)
from agent_core.providers.openai_responses import _parse_responses_output
from agent_core.runtime.loop._runaway import (
    _is_runaway_response,
    is_truncated_with_text,
)


def test_only_truncation_markers_are_rewritten() -> None:
    assert normalize_finish_reason("max_tokens") == "length"
    assert normalize_finish_reason("max_output_tokens") == "length"
    assert normalize_finish_reason("length") == "length"
    # Provider-meaningful values are preserved: hosts and tests read them.
    assert normalize_finish_reason("tool_use") == "tool_use"
    assert normalize_finish_reason("end_turn") == "end_turn"
    assert normalize_finish_reason("stop") == "stop"
    assert normalize_finish_reason("") == ""
    assert normalize_finish_reason(None) == ""


def test_responses_status_folds_the_nested_reason() -> None:
    # ``status`` alone can never express truncation.
    assert responses_finish_reason("incomplete", "max_output_tokens") == "length"
    assert responses_finish_reason("completed", None) == "completed"
    assert responses_finish_reason("incomplete", "content_filter") == "content_filter"


class _Raw:
    def __init__(self, **kw: Any) -> None:
        self.__dict__.update(kw)


def test_anthropic_max_tokens_reaches_the_truncation_check() -> None:
    raw = _Raw(
        content=[_Raw(type="text", text="a half sentence that stops")],
        stop_reason="max_tokens",
        model="claude-x",
        usage=None,
        id="m1",
    )
    response = _to_llm_response(raw)
    assert response.finish_reason == "length"
    # The whole point: recovery now fires for anthropic/bedrock.
    assert is_truncated_with_text(response) is True


def test_anthropic_tool_use_is_still_passed_through() -> None:
    raw = _Raw(
        content=[_Raw(type="text", text="calling")],
        stop_reason="tool_use",
        model="claude-x",
        usage=None,
        id="m2",
    )
    assert _to_llm_response(raw).finish_reason == "tool_use"


def test_responses_incomplete_reaches_the_truncation_check() -> None:
    raw = {
        "output": [{
            "type": "message",
            "content": [{"type": "output_text", "text": "cut off here"}],
        }],
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "model": "gpt-5",
        "id": "r1",
    }
    response = _parse_responses_output(raw)
    assert response.finish_reason == "length"
    assert is_truncated_with_text(response) is True


def test_responses_empty_truncated_reply_is_a_runaway() -> None:
    raw = {
        "output": [],
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "model": "gpt-5",
        "id": "r2",
    }
    response = _parse_responses_output(raw)
    assert response.finish_reason == "length"
    assert _is_runaway_response(response) is True


def test_completed_response_is_not_treated_as_truncated() -> None:
    raw = {
        "output": [{
            "type": "message",
            "content": [{"type": "output_text", "text": "a full answer."}],
        }],
        "status": "completed",
        "model": "gpt-5",
        "id": "r3",
    }
    response = _parse_responses_output(raw)
    assert response.finish_reason == "completed"
    assert is_truncated_with_text(response) is False
    assert isinstance(response, LLMResponse)


def test_unmapped_marker_would_be_invisible() -> None:
    """Documents the coupling this normalization exists to satisfy.

    ``_runaway`` matches the literal string, so a raw provider marker reaching
    ``LLMResponse`` unmapped disables truncation recovery entirely.
    """
    raw_marker = LLMResponse(content="cut off", finish_reason="max_tokens")
    assert is_truncated_with_text(raw_marker) is False
    normalized = LLMResponse(
        content="cut off",
        finish_reason=normalize_finish_reason("max_tokens"),
    )
    assert is_truncated_with_text(normalized) is True
