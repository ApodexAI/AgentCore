"""Tests for BudgetObserver."""

from __future__ import annotations

import pytest

from agent_core.components.observers.budget_observer import BudgetObserver
from agent_core.loop_types import Intervention, TurnContext


def _make_ctx(usage: dict | None = None, metadata: dict | None = None) -> TurnContext:
    return TurnContext(
        turn=1,
        max_turns=50,
        task_id="t1",
        role_id="react_solver",
        ai_text="",
        thinking="",
        tool_calls=[],
        messages=[],
        usage=usage,
        metadata=metadata if metadata is not None else {},
    )


# ---------------------------------------------------------------------------
# 1. critical flag
# ---------------------------------------------------------------------------


def test_is_critical() -> None:
    assert BudgetObserver.critical is True


# ---------------------------------------------------------------------------
# 2. Token tracking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tracks_token_usage() -> None:
    obs = BudgetObserver(max_tokens=100_000)
    ctx = _make_ctx(usage={"input_tokens": 1000, "output_tokens": 500})
    await obs.on_llm_response(ctx)
    assert obs.tokens_used == 1500


@pytest.mark.asyncio
async def test_accumulates_across_turns() -> None:
    obs = BudgetObserver(max_tokens=100_000)
    ctx1 = _make_ctx(usage={"input_tokens": 1000, "output_tokens": 500})
    ctx2 = _make_ctx(usage={"input_tokens": 2000, "output_tokens": 300})
    await obs.on_llm_response(ctx1)
    await obs.on_llm_response(ctx2)
    assert obs.tokens_used == 3800


@pytest.mark.asyncio
async def test_no_usage_metadata_skipped() -> None:
    obs = BudgetObserver(max_tokens=100_000)
    ctx = _make_ctx(usage=None)
    await obs.on_llm_response(ctx)
    assert obs.tokens_used == 0


# ---------------------------------------------------------------------------
# 3. Intervention on exhaustion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stops_when_exhausted() -> None:
    obs = BudgetObserver(max_tokens=5000)
    ctx = _make_ctx(usage={"input_tokens": 4000, "output_tokens": 2000})
    await obs.on_llm_response(ctx)
    result = await obs.on_turn_end(ctx)
    assert isinstance(result, Intervention)
    assert result.stop_reason == "budget_exhausted"
    assert result.inject_messages is not None
    assert len(result.inject_messages) > 0


@pytest.mark.asyncio
async def test_warns_near_budget() -> None:
    obs = BudgetObserver(max_tokens=10_000, warn_ratio=0.8)
    ctx = _make_ctx(usage={"input_tokens": 7000, "output_tokens": 1500})
    await obs.on_llm_response(ctx)
    result = await obs.on_turn_end(ctx)
    # 8500 / 10000 = 85% >= 80% → warn only, no stop
    assert isinstance(result, Intervention)
    assert result.stop_reason is None
    assert result.inject_messages is not None
    assert len(result.inject_messages) > 0


@pytest.mark.asyncio
async def test_no_intervention_under_budget() -> None:
    obs = BudgetObserver(max_tokens=100_000)
    ctx = _make_ctx(usage={"input_tokens": 1000, "output_tokens": 500})
    await obs.on_llm_response(ctx)
    result = await obs.on_turn_end(ctx)
    assert result is None


# ---------------------------------------------------------------------------
# 4. Metadata exposure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_exposes_usage_in_metadata() -> None:
    obs = BudgetObserver(max_tokens=50_000)
    metadata: dict = {}
    ctx = _make_ctx(usage={"input_tokens": 1000, "output_tokens": 500}, metadata=metadata)
    await obs.on_llm_response(ctx)
    assert ctx.metadata["budget_tokens_used"] == 1500
    assert ctx.metadata["budget_tokens_limit"] == 50_000
