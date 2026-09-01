"""Tests for SSEObserver."""
from __future__ import annotations

from typing import ClassVar
from unittest.mock import AsyncMock

import pytest

from agent_core.components.observers.sse_observer import SSEObserver
from agent_core.events import EventType
from agent_core.loop_types import ToolResult, TurnContext
from agent_core.runtime.loop.model_profile import HistoryPolicy


def _make_ctx(turn: int = 1, thinking: str = "some thinking") -> TurnContext:
    return TurnContext(
        turn=turn,
        max_turns=50,
        task_id="task-001",
        role_id="react_solver",
        ai_text="The answer is 42.",
        thinking=thinking,
        tool_calls=[],
        messages=[],
        usage={"input_tokens": 10, "output_tokens": 5},
        metadata={},
    )


def _make_tool_result() -> ToolResult:
    return ToolResult(
        name="web_search",
        args={"query": "AI chips 2025"},
        result="NVIDIA leads...",
        duration_ms=350,
        tool_call_id="tc-abc",
        is_error=False,
    )


# ---------------------------------------------------------------------------
# Test 1: passive flag
# ---------------------------------------------------------------------------


def test_is_passive():
    obs = SSEObserver(event_store=AsyncMock(), task_id="task-001")
    assert obs.critical is False


# ---------------------------------------------------------------------------
# Test 2: on_llm_response emits react_think
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_llm_response_emits_react_think():
    mock_es = AsyncMock()
    obs = SSEObserver(event_store=mock_es, task_id="task-001")
    ctx = _make_ctx()

    await obs.on_llm_response(ctx)

    mock_es.append.assert_awaited_once()
    args = mock_es.append.call_args[0]
    assert args[0] == "task-001"
    assert args[1] == EventType.AGENT_ACTION
    payload = args[2]
    assert payload["trace_type"] == "react_think"
    assert payload["agent"] == "react_solver"
    assert payload["turn"] == 1


# ---------------------------------------------------------------------------
# Test 3: on_tool_result emits react_tool_call with tool_name
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_tool_result_emits_react_tool_call():
    mock_es = AsyncMock()
    obs = SSEObserver(event_store=mock_es, task_id="task-001")
    ctx = _make_ctx()
    result = _make_tool_result()

    await obs.on_tool_result(ctx, result)

    mock_es.append.assert_awaited_once()
    args = mock_es.append.call_args[0]
    assert args[1] == EventType.AGENT_ACTION
    payload = args[2]
    assert payload["trace_type"] == "react_tool_call"
    assert payload["tool_name"] == "web_search"
    assert "duration_ms" in payload


# ---------------------------------------------------------------------------
# Test 4: thinking included when policy allows
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_thinking_emitted_when_policy_allows():
    mock_es = AsyncMock()
    policy = HistoryPolicy(thinking_in_sse=True)
    obs = SSEObserver(event_store=mock_es, task_id="task-001", history_policy=policy)
    ctx = _make_ctx(thinking="deep chain-of-thought here")

    await obs.on_llm_response(ctx)

    args = mock_es.append.call_args[0]
    payload = args[2]
    assert "thinking" in payload
    assert payload["thinking"] == "deep chain-of-thought here"


# ---------------------------------------------------------------------------
# Test 5: thinking hidden when policy denies
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_thinking_hidden_when_policy_denies():
    mock_es = AsyncMock()
    policy = HistoryPolicy(thinking_in_sse=False)
    obs = SSEObserver(event_store=mock_es, task_id="task-001", history_policy=policy)
    ctx = _make_ctx(thinking="secret reasoning")

    await obs.on_llm_response(ctx)

    args = mock_es.append.call_args[0]
    payload = args[2]
    assert "thinking" not in payload


# ---------------------------------------------------------------------------
# Test 6: skill_loaded emitted once when read_text hits skills/<id>/SKILL.md
# ---------------------------------------------------------------------------


def _make_skill_read(path: str, is_error: bool = False) -> ToolResult:
    return ToolResult(
        name="read_text",
        args={"path": path},
        result="---\nname: Chart Visualization\n---\n# body" if not is_error else "Access denied",
        duration_ms=5,
        tool_call_id="tc-skill",
        is_error=is_error,
    )


@pytest.mark.asyncio
async def test_skill_loaded_emitted_on_skill_read():
    mock_es = AsyncMock()
    obs = SSEObserver(event_store=mock_es, task_id="task-001")
    ctx = _make_ctx()

    await obs.on_tool_result(ctx, _make_skill_read("skills/chart-visualization/SKILL.md"))

    # Two appends: react_tool_call + skill_loaded.
    assert mock_es.append.await_count == 2
    second_payload = mock_es.append.await_args_list[1][0][2]
    assert second_payload["trace_type"] == "skill_loaded"
    assert second_payload["skill_id"] == "chart-visualization"
    assert second_payload["action"] == "skill_loaded"
    assert second_payload["agent"] == "react_solver"


@pytest.mark.asyncio
async def test_skill_loaded_deduped_per_session():
    mock_es = AsyncMock()
    obs = SSEObserver(event_store=mock_es, task_id="task-001")
    ctx = _make_ctx()
    path = "skills/chart-visualization/SKILL.md"

    await obs.on_tool_result(ctx, _make_skill_read(path))
    await obs.on_tool_result(ctx, _make_skill_read(path))

    # 2 react_tool_call + 1 skill_loaded (second read is suppressed).
    assert mock_es.append.await_count == 3
    trace_types = [call[0][2]["trace_type"] for call in mock_es.append.await_args_list]
    assert trace_types.count("skill_loaded") == 1


@pytest.mark.asyncio
async def test_skill_loaded_skipped_on_error():
    mock_es = AsyncMock()
    obs = SSEObserver(event_store=mock_es, task_id="task-001")
    ctx = _make_ctx()

    await obs.on_tool_result(
        ctx,
        _make_skill_read("skills/chart-visualization/SKILL.md", is_error=True),
    )

    assert mock_es.append.await_count == 1  # only react_tool_call


@pytest.mark.asyncio
async def test_skill_loaded_ignores_non_skill_reads():
    mock_es = AsyncMock()
    obs = SSEObserver(event_store=mock_es, task_id="task-001")
    ctx = _make_ctx()

    await obs.on_tool_result(
        ctx,
        ToolResult(
            name="read_text",
            args={"path": "data/reports/foo.md"},
            result="hello",
            duration_ms=3,
            tool_call_id="tc-x",
            is_error=False,
        ),
    )

    assert mock_es.append.await_count == 1
    assert mock_es.append.await_args[0][2]["trace_type"] == "react_tool_call"


# ---------------------------------------------------------------------------
# skill_loaded resolves the display name through a registered SkillLoader
# ---------------------------------------------------------------------------


class _FakeSkill:
    skill_id = "chart-visualization"
    name = "Chart Visualization"
    description = "charts"
    version = "1.0.0"
    tags: ClassVar[list[str]] = []
    allowed_tools: ClassVar[list[str]] = []
    content = ""
    root_dir = "/skills/chart-visualization"
    enabled = True


class _FakeSkillLoader:
    """Structural ``SkillLoader`` — registered under its own concrete type."""

    def list_skills(self):
        return [_FakeSkill()]

    def get_skill(self, skill_id: str):
        return _FakeSkill() if skill_id == _FakeSkill.skill_id else None

    def get_enabled_skills(self):
        return [_FakeSkill()]

    def toggle_skill(self, skill_id: str, enabled: bool) -> bool:
        return False

    def reload(self) -> None:
        return None


@pytest.mark.asyncio
async def test_skill_loaded_uses_registered_loader_display_name():
    from agent_core.runtime.registries import services as registry
    from agent_core.runtime.registries.scope import use_scope

    mock_es = AsyncMock()
    obs = SSEObserver(event_store=mock_es, task_id="task-001")
    ctx = _make_ctx()

    with use_scope(name="sse-skill-name", fallback_to_process=False):
        registry.register(_FakeSkillLoader, _FakeSkillLoader())
        await obs.on_tool_result(
            ctx, _make_skill_read("skills/chart-visualization/SKILL.md"),
        )

    payload = mock_es.append.await_args_list[1][0][2]
    assert payload["skill_id"] == "chart-visualization"
    assert payload["skill_name"] == "Chart Visualization"
    assert payload["detail"] == "Loaded skill: Chart Visualization"


@pytest.mark.asyncio
async def test_skill_loaded_falls_back_to_id_without_loader():
    from agent_core.runtime.registries.scope import use_scope

    mock_es = AsyncMock()
    obs = SSEObserver(event_store=mock_es, task_id="task-001")
    ctx = _make_ctx()

    with use_scope(name="sse-no-loader", fallback_to_process=False):
        await obs.on_tool_result(
            ctx, _make_skill_read("skills/chart-visualization/SKILL.md"),
        )

    payload = mock_es.append.await_args_list[1][0][2]
    assert payload["skill_name"] == "chart-visualization"
