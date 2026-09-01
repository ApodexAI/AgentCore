"""Tests for the cooperative sub-agent stop signal (registry + observer).

The heavyweight end-to-end loop test from the source branch is langchain-bound;
this covers the deterministic core on the current API: the registry is one-shot,
and ``StopSignalObserver`` injects a rollback intervention exactly when (and
only when) a stop is queued for its session.
"""

from __future__ import annotations

import pytest

from agent_core.components.agent_bus.stop_signal import (
    SubAgentStopRegistry,
    get_stop_registry,
)
from agent_core.components.observers.stop_signal_observer import (
    STOP_SIGNAL_PROMPT,
    StopSignalObserver,
)
from agent_core.loop_types import TurnContext


def _make_ctx(turn: int = 2) -> TurnContext:
    return TurnContext(
        turn=turn,
        max_turns=8,
        task_id="t1",
        role_id="swarm_sub",
        ai_text="",
        thinking="",
        tool_calls=[],
        messages=[],
        usage=None,
        metadata={},
    )


# ---------------------------------------------------------------------------
# SubAgentStopRegistry
# ---------------------------------------------------------------------------


def test_request_then_consume_is_one_shot() -> None:
    reg = SubAgentStopRegistry()
    reg.request_stop("t1::alice")
    assert reg.is_requested("t1::alice") is True
    assert reg.consume("t1::alice") is True
    # Second consume returns False — the prompt is injected only once.
    assert reg.consume("t1::alice") is False
    assert reg.is_requested("t1::alice") is False


def test_blank_session_is_ignored() -> None:
    reg = SubAgentStopRegistry()
    reg.request_stop("   ")
    assert reg.consume("") is False


def test_get_stop_registry_is_singleton() -> None:
    assert get_stop_registry() is get_stop_registry()


def test_clear_stale_drops_prior_job() -> None:
    # A stop queued for job-1 that was never consumed must not survive into
    # job-2 on the same reused session.
    reg = SubAgentStopRegistry()
    reg.request_stop("t1::alice", "job-1")
    assert reg.clear_stale("t1::alice", "job-2") is True
    assert reg.is_requested("t1::alice") is False


def test_clear_stale_keeps_current_job() -> None:
    # A stop for the job that is actually starting must be preserved.
    reg = SubAgentStopRegistry()
    reg.request_stop("t1::alice", "job-2")
    assert reg.clear_stale("t1::alice", "job-2") is False
    assert reg.consume("t1::alice") is True


# ---------------------------------------------------------------------------
# StopSignalObserver
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_observer_no_signal_returns_none() -> None:
    obs = StopSignalObserver(session_id="t1::bob", registry=SubAgentStopRegistry())
    assert await obs.on_llm_response(_make_ctx()) is None


@pytest.mark.asyncio
async def test_observer_injects_rollback_on_stop() -> None:
    reg = SubAgentStopRegistry()
    obs = StopSignalObserver(session_id="t1::bob", registry=reg)
    reg.request_stop("t1::bob")

    iv = await obs.on_llm_response(_make_ctx())
    assert iv is not None
    assert iv.inject_messages == [STOP_SIGNAL_PROMPT]
    # Rollback: drop the exploration tool_calls this turn chose, jump to next.
    assert iv.pop_last_message is True
    assert iv.continue_to_next_turn is True

    # One-shot: a follow-up turn with no new request does nothing.
    assert await obs.on_llm_response(_make_ctx(turn=3)) is None


@pytest.mark.asyncio
async def test_observer_isolated_per_session() -> None:
    reg = SubAgentStopRegistry()
    a = StopSignalObserver(session_id="t1::alice", registry=reg)
    b = StopSignalObserver(session_id="t1::bob", registry=reg)
    reg.request_stop("t1::alice")
    # Stopping alice must not affect bob.
    assert await b.on_llm_response(_make_ctx()) is None
    assert await a.on_llm_response(_make_ctx()) is not None
