"""``LLMSummaryCompactor`` unit tests.

Cover the four flavours of behaviour:

- Happy path: summary LLM returns text → output is
  ``[system, user_msg(summary), recent]``.
- Rollback on LLM raise: output is
  ``[system, placeholder, last_user_query]`` and the emitter receives
  ``rolled_back=True`` with a reason.
- Rollback on empty summary: same shape; reason is ``empty_summary``.
- No-LLM fallthrough: when ``summary_llm=None`` the compactor delegates
  to the string-slice path (no LLM call, no rollback).

The agent-loop integration is covered separately by an integration test
in ``tests/test_agent_loop_pre_llm_compact.py`` (existing) once the
async-aware call site lands.
"""
from __future__ import annotations

import pytest

from agent_core.llm import LLMResponse
from agent_core.messages import (
    Message,
    assistant_msg,
    system_msg,
    tool_msg,
    user_msg,
)
from agent_core.runtime.loop.compact_llm import LLMSummaryCompactor


class _StaticSummaryLLM:
    """Fake LLM whose ``chat`` returns a configured string."""

    def __init__(self, content: str = "## summary\n- entity X confirmed") -> None:
        self.model = "static-summary"
        self._content = content
        self.calls: list[list] = []

    async def chat(self, messages, **kwargs) -> LLMResponse:
        self.calls.append(list(messages))
        return LLMResponse(content=self._content)


class _FailingSummaryLLM:
    def __init__(self, exc: Exception | None = None) -> None:
        self.model = "failing-summary"
        self._exc = exc or RuntimeError("primary down")

    async def chat(self, messages, **kwargs):
        raise self._exc


def _long_history(n: int = 8) -> list[Message]:
    """Build a history with system + n turns of user/AI/tool messages."""
    msgs: list[Message] = [system_msg("be helpful")]
    for i in range(n):
        msgs.append(user_msg(f"user query #{i}"))
        msgs.append(assistant_msg(f"thinking #{i}"))
        msgs.append(tool_msg(f"tool result #{i}", f"t{i}"))
    msgs.append(user_msg("final user query"))
    return msgs


@pytest.mark.asyncio
async def test_happy_path_returns_system_summary_recent() -> None:
    history = _long_history(n=6)
    summary_llm = _StaticSummaryLLM(content="ROLLUP-CONTENT")

    captured: list[dict] = []

    async def _emit(payload):
        captured.append(payload)

    compactor = LLMSummaryCompactor(
        summary_llm=summary_llm, emit_event=_emit,
    )
    out = await compactor.compact(history, keep_recent=4)

    assert out[0].get("role") == "system"
    # The single rollup user message carries the LLM-supplied summary.
    assert out[1].get("role") == "user"
    assert "ROLLUP-CONTENT" in out[1].get("content")
    # ``keep_recent`` recent messages must follow verbatim.
    assert out[-1] is history[-1]

    assert len(summary_llm.calls) == 1
    assert len(captured) == 1
    assert captured[0]["rolled_back"] is False
    assert captured[0]["messages_before"] == len(history)
    assert captured[0]["messages_after"] == len(out)
    assert captured[0]["compactor"] == "llm"
    assert captured[0]["attempts"] == 1
    assert captured[0]["summary"] == "ROLLUP-CONTENT"


@pytest.mark.asyncio
async def test_rollback_on_llm_failure_keeps_only_system_and_last_user() -> None:
    history = _long_history(n=6)
    last_user = history[-1]

    captured: list[dict] = []

    async def _emit(payload):
        captured.append(payload)

    compactor = LLMSummaryCompactor(
        summary_llm=_FailingSummaryLLM(RuntimeError("502 backend")),
        emit_event=_emit,
    )
    out = await compactor.compact(history, keep_recent=4)

    assert out[0].get("role") == "system"
    assert out[1].get("role") == "user"
    assert "Compaction failed" in out[1].get("content")
    assert "llm_error" in out[1].get("content")
    # Last user query preserved so the next turn has something to answer.
    assert out[-1] is last_user

    assert captured[0]["rolled_back"] is True
    assert captured[0]["rollback_reason"] == "llm_error"
    assert "502 backend" in captured[0]["error"]


@pytest.mark.asyncio
async def test_rollback_on_empty_summary() -> None:
    history = _long_history(n=6)
    captured: list[dict] = []

    async def _emit(payload):
        captured.append(payload)

    compactor = LLMSummaryCompactor(
        summary_llm=_StaticSummaryLLM(content="   \n  \t "),
        emit_event=_emit,
    )
    out = await compactor.compact(history, keep_recent=4)

    assert out[1].get("role") == "user"
    assert "Compaction failed" in out[1].get("content")
    assert captured[0]["rollback_reason"] == "empty_summary"


@pytest.mark.asyncio
async def test_no_llm_falls_through_to_string_slice() -> None:
    """No summarizer → behaves as the deterministic string-slice path."""
    history = _long_history(n=6)
    compactor = LLMSummaryCompactor(summary_llm=None)
    out = await compactor.compact(history, keep_recent=4)

    # String-slice keeps system + ONE compact summary user message + recent.
    assert out[0].get("role") == "system"
    assert out[1].get("role") == "user"
    assert "Compacted" in out[1].get("content")
    # Importantly: no rollback placeholder text.
    assert "Compaction failed" not in out[1].get("content")


@pytest.mark.asyncio
async def test_short_history_below_keep_recent_is_unchanged() -> None:
    history = [
        system_msg("be helpful"),
        user_msg("hi"),
        assistant_msg("hello"),
    ]
    compactor = LLMSummaryCompactor(summary_llm=_StaticSummaryLLM())
    out = await compactor.compact(history, keep_recent=4)

    assert out == history
