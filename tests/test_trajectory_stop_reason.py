"""A trajectory must record WHY a turn ended, not only what it produced.

Three turn endings used to be indistinguishable in a JSONL trajectory: an
answer the model finished, an answer the output cap cut off, and a request a
safety classifier declined. The first two differ only in ``finish_reason``;
the third arrives with empty ``content`` and no tool calls, so it reads as a
turn where the model simply chose to say nothing, and the loop's no-tool exit
ends the run looking clean.

The 2026-10-05 gdpval triage hit exactly this: 20 runs ended in ``llm_error``
and 25 more in ``no_tool`` with an empty deliverable, and nothing on disk —
trajectory, job log, or breaker state — recorded which of those were declines.
These pin the fields that answer the question.
"""

from __future__ import annotations

import json

import pytest

from agent_core.components.observers.trajectory import TrajectoryFileObserver
from agent_core.loop_types import TurnContext


def _ctx(**overrides) -> TurnContext:
    base = dict(
        turn=1, max_turns=10, task_id="t", role_id="r",
        ai_text="", thinking="", tool_calls=[], messages=[],
        usage=None, metadata={},
    )
    base.update(overrides)
    return TurnContext(**base)


def _llm_records(tmp_path) -> list[dict]:
    lines = (tmp_path / "t.jsonl").read_text().splitlines()
    return [r for r in (json.loads(x) for x in lines) if r.get("t") == "llm"]


async def _record(tmp_path, ctx: TurnContext) -> dict:
    obs = TrajectoryFileObserver(tmp_path, filename="t", formats=["jsonl"])
    await obs.on_llm_response(ctx)
    records = _llm_records(tmp_path)
    assert len(records) == 1
    return records[0]


@pytest.mark.asyncio
async def test_a_refusal_is_recorded_as_a_refusal(tmp_path):
    """The shape that used to look like "the model said nothing"."""
    record = await _record(tmp_path, _ctx(
        finish_reason="refusal",
        stop_details={"type": "refusal", "category": "bio",
                      "explanation": "declined"},
    ))

    assert record["content"] == ""
    assert record["tool_calls"] == []
    assert record["finish_reason"] == "refusal"
    assert record["stop_details"]["category"] == "bio"


@pytest.mark.asyncio
async def test_a_truncated_turn_is_distinguishable_from_a_finished_one(tmp_path):
    truncated = await _record(tmp_path / "a", _ctx(
        ai_text="half a sent", finish_reason="length",
    ))
    finished = await _record(tmp_path / "b", _ctx(
        ai_text="a whole answer", finish_reason="end_turn",
    ))

    assert truncated["finish_reason"] == "length"
    # ``end_turn`` carries no information and would be on nearly every line.
    assert "finish_reason" not in finished


@pytest.mark.asyncio
async def test_an_ordinary_turn_stays_lean(tmp_path):
    """No empty keys on the overwhelmingly common path."""
    record = await _record(tmp_path, _ctx(
        ai_text="hello", finish_reason="end_turn",
    ))

    assert "finish_reason" not in record
    assert "stop_details" not in record


@pytest.mark.asyncio
async def test_a_producer_that_sets_nothing_is_no_worse_than_before(tmp_path):
    """The fields default, so an un-migrated caller still writes a valid line."""
    record = await _record(tmp_path, _ctx(ai_text="hello"))

    assert record["content"] == "hello"
    assert "finish_reason" not in record
    assert "stop_details" not in record


@pytest.mark.asyncio
async def test_tool_use_turns_record_their_stop_reason(tmp_path):
    """Useful for telling a tool-call turn from a text turn when replaying."""
    record = await _record(tmp_path, _ctx(
        tool_calls=[{"name": "bash", "args": {}}], finish_reason="tool_use",
    ))

    assert record["finish_reason"] == "tool_use"
