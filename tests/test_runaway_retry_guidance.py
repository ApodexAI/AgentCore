"""Runaway-retry guidance must describe the retries that actually ran.

With expansion disabled (a lowered ``RUNAWAY_MAX_RETRIES``) or impossible, the
first retry goes straight to the reduced phase. Its reminder used to say "The
expanded-thinking retry still produced no visible answer", describing an
attempt the model never saw.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agent_core.llm import LLMResponse
from agent_core.runtime.loop import _call
from agent_core.runtime.loop._runaway import _runaway_retry_policy


def test_reduced_guidance_names_the_expanded_retry_only_when_it_ran() -> None:
    _, after_expanded, _ = _runaway_retry_policy(2, 4096, expanded_attempted=True)
    _, without_expanded, _ = _runaway_retry_policy(2, 4096, expanded_attempted=False)

    assert "expanded-thinking retry" in after_expanded
    assert "expanded" not in without_expanded
    assert "short, bounded reasoning pass" in without_expanded


class _RunawayThenAnswer:
    """First reply spends the whole budget on reasoning; the retry answers."""

    model = "fake"

    def __init__(self) -> None:
        self.requests: list[list[Any]] = []

    async def chat(self, messages: list[Any], **_kwargs: Any) -> LLMResponse:
        self.requests.append(list(messages))
        if len(self.requests) == 1:
            return LLMResponse(
                content="",
                reasoning_content="x" * 5_000,
                finish_reason="length",
                usage={"prompt_tokens": 10, "completion_tokens": 2_048},
            )
        return LLMResponse(content="done", finish_reason="stop")


def _retry_reminder(llm: _RunawayThenAnswer) -> str:
    assert len(llm.requests) == 2
    last = llm.requests[1][-1]
    assert last["role"] == "user"
    return str(last["content"])


def test_skipped_expansion_does_not_mention_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_call, "_RUNAWAY_EXPAND_ENABLED", False)
    monkeypatch.setattr(_call, "_RUNAWAY_BACKOFF_S", 0.0)
    llm = _RunawayThenAnswer()

    response = asyncio.run(
        _call.call_llm(llm, [{"role": "user", "content": "q"}], timeout=30, max_retries=3, turn=1)
    )

    assert response is not None and response.content == "done"
    reminder = _retry_reminder(llm)
    assert reminder.startswith("[system reminder]")
    assert "expanded" not in reminder
