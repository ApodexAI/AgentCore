"""Tests for ReactStepTracker observer."""
from __future__ import annotations

import json

import pytest

from agent_core.components.observers.react_step_tracker import ReactStepTracker
from agent_core.loop_types import ToolResult, TurnContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(
    turn=1, thinking="I should search", tool_calls=None, metadata=None
):
    return TurnContext(
        turn=turn,
        max_turns=50,
        task_id="t1",
        role_id="react_solver",
        ai_text="text",
        thinking=thinking,
        tool_calls=tool_calls or [],
        messages=[],
        usage=None,
        metadata=metadata if metadata is not None else {},
    )


def _make_tool_result(
    name="web_search", args=None, result="ok", duration_ms=100
):
    return ToolResult(
        name=name,
        args=args or {"query": "test"},
        result=result,
        duration_ms=duration_ms,
        tool_call_id="call_1",
        is_error=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_records_tool_call_as_step():
    tracker = ReactStepTracker()
    ctx = _make_ctx()
    tr = _make_tool_result()

    await tracker.on_tool_result(ctx, tr)

    steps = ctx.metadata["react_steps"]
    assert len(steps) == 1
    step = steps[0]
    assert step["turn"] == 1
    assert step["thinking"] == "I should search"
    assert step["tool_name"] == "web_search"
    assert '"query": "test"' in step["tool_args"]
    assert step["tool_result"] == "ok"
    assert step["duration_ms"] == 100
    assert step["is_error"] is False


@pytest.mark.asyncio
async def test_records_multiple_tools_per_turn():
    tracker = ReactStepTracker()
    ctx = _make_ctx(turn=2)
    tr1 = _make_tool_result(name="web_search", result="result1")
    tr2 = _make_tool_result(name="web_fetch", result="result2")

    await tracker.on_tool_result(ctx, tr1)
    await tracker.on_tool_result(ctx, tr2)

    steps = ctx.metadata["react_steps"]
    assert len(steps) == 2
    assert steps[0]["tool_name"] == "web_search"
    assert steps[1]["tool_name"] == "web_fetch"
    assert steps[0]["turn"] == 2
    assert steps[1]["turn"] == 2


@pytest.mark.asyncio
async def test_fields_under_cap_ride_verbatim():
    """Below the caps nothing is rewritten — the common case must be
    byte-identical to the pre-cap behaviour."""
    tracker = ReactStepTracker()
    thinking = "x" * 1000
    result = "y" * 1000
    ctx = _make_ctx(thinking=thinking)
    tr = _make_tool_result(result=result)

    await tracker.on_tool_result(ctx, tr)

    step = ctx.metadata["react_steps"][0]
    assert step["thinking"] == thinking
    assert step["tool_result"] == result


@pytest.mark.asyncio
async def test_oversized_result_and_thinking_are_capped():
    """The OOM guardrail: one observer instance exists per agent and
    ``react_steps`` lives for the whole run, so a 150K tool body must not
    be retained verbatim (see the module docstring)."""
    tracker = ReactStepTracker()
    long_thinking = "x" * 50_000
    long_result = "y" * 150_000
    ctx = _make_ctx(thinking=long_thinking)
    tr = _make_tool_result(result=long_result)

    await tracker.on_tool_result(ctx, tr)

    step = ctx.metadata["react_steps"][0]
    assert len(step["tool_result"]) < len(long_result)
    assert len(step["thinking"]) < len(long_thinking)
    # Head is preserved so the field stays useful, and the drop is explicit
    # rather than looking like a genuinely short result.
    assert step["tool_result"].startswith("y" * 100)
    assert "truncated 145904 of 150000 chars" in step["tool_result"]
    assert step["tool_result"].isascii()
    assert "truncated" in step["thinking"]


@pytest.mark.asyncio
async def test_caps_can_be_disabled_via_env(monkeypatch):
    """``0`` restores the historical unbounded behaviour, for callers that
    genuinely need the full body in ``react_steps``."""
    from agent_core.components.observers import react_step_tracker as rst

    monkeypatch.setenv("MIROHARNESS_REACT_STEP_RESULT_MAX_CHARS", "0")
    monkeypatch.setenv("MIROHARNESS_REACT_STEP_THINKING_MAX_CHARS", "0")
    rst._caps.cache_clear()
    try:
        tracker = ReactStepTracker()
        long_thinking = "x" * 50_000
        long_result = "y" * 150_000
        ctx = _make_ctx(thinking=long_thinking)
        tr = _make_tool_result(result=long_result)

        await tracker.on_tool_result(ctx, tr)

        step = ctx.metadata["react_steps"][0]
        assert step["tool_result"] == long_result
        assert step["thinking"] == long_thinking
    finally:
        rst._caps.cache_clear()


@pytest.mark.asyncio
async def test_oversized_tool_args_stay_json_parseable():
    """Over-cap ``tool_args`` becomes a valid JSON *envelope*, never a
    mid-string slice — downstream consumers ``json.loads`` this field, and
    invalid JSON silently pushes them onto regex fallbacks."""
    tracker = ReactStepTracker()
    big_args = {"blob": "z" * 100_000}
    ctx = _make_ctx()
    tr = _make_tool_result(name="create_subagent", args=big_args)

    await tracker.on_tool_result(ctx, tr)

    step = ctx.metadata["react_steps"][0]
    decoded = json.loads(step["tool_args"])  # must not raise
    assert decoded["_truncated"] is True
    assert decoded["_original_chars"] > 100_000
    assert decoded["_preview"].startswith('{"blob": "zzz')


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("locator_key", "locator_value"),
    [
        ("path", "/workspace/report.md"),
        ("file_path", "/workspace/report.md"),
        ("image_path_or_url", "/workspace/chart.png"),
    ],
)
async def test_oversized_tool_args_preserve_artifact_locator(
    locator_key, locator_value,
):
    """A valid truncation envelope must not hide paths from legacy API
    consumers that successfully parse JSON and then use ``args.get``."""
    tracker = ReactStepTracker()
    big_args = {locator_key: locator_value, "content": "x" * 20_000}
    ctx = _make_ctx()
    tr = _make_tool_result(name="write_file", args=big_args)

    await tracker.on_tool_result(ctx, tr)

    decoded = json.loads(ctx.metadata["react_steps"][0]["tool_args"])
    assert decoded["_truncated"] is True
    assert decoded[locator_key] == locator_value
    assert "content" not in decoded


@pytest.mark.asyncio
async def test_tool_args_not_truncated_preserves_json_parseability():
    """``tool_args`` is JSON-stringified without truncation so the
    downstream ``json.loads`` consumers (agent_runner, session_io,
    agent_intent) round-trip nested args verbatim."""
    tracker = ReactStepTracker()
    big_args = {
        "agents": [
            {"name": "lit_search", "system_prompt": "x" * 2000},
            {"name": "fact_checker", "system_prompt": "y" * 2000},
        ],
    }
    ctx = _make_ctx()
    tr = _make_tool_result(name="create_subagent", args=big_args)

    await tracker.on_tool_result(ctx, tr)

    step = ctx.metadata["react_steps"][0]
    assert json.loads(step["tool_args"]) == big_args


@pytest.mark.asyncio
async def test_dict_result_does_not_crash():
    """Structured tool results (dict / list) must not raise — the
    observer is ``critical=True``, so a TypeError here used to take
    out the agent loop. Native structures JSON-encode into the preview."""
    tracker = ReactStepTracker()
    ctx = _make_ctx()
    structured_result = {"hits": [{"url": "u1"}, {"url": "u2"}], "total": 2}
    tr = _make_tool_result(name="web_search", result=structured_result)

    await tracker.on_tool_result(ctx, tr)

    step = ctx.metadata["react_steps"][0]
    assert isinstance(step["tool_result"], str)
    assert json.loads(step["tool_result"]) == structured_result


@pytest.mark.asyncio
async def test_error_tool_recorded():
    tracker = ReactStepTracker()
    ctx = _make_ctx()
    tr = ToolResult(
        name="bash",
        args={"command": "rm -rf /"},
        result="Permission denied",
        duration_ms=50,
        tool_call_id="call_err",
        is_error=True,
    )

    await tracker.on_tool_result(ctx, tr)

    step = ctx.metadata["react_steps"][0]
    assert step["is_error"] is True
    assert step["tool_name"] == "bash"
    assert step["tool_result"] == "Permission denied"


@pytest.mark.asyncio
async def test_critical_is_true():
    tracker = ReactStepTracker()
    assert tracker.critical is True
