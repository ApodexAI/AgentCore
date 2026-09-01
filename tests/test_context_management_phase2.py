from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_core.loop_types import TurnContext
from agent_core.messages import assistant_msg, tool_msg, user_msg
from agent_core.runtime.loop.budget_consistency import check_context_budget
from agent_core.runtime.loop.compact import INPUT_ESTIMATE_KEY
from agent_core.runtime.loop.compact_llm import LLMSummaryCompactor
from agent_core.runtime.loop.summary_prompt import (
    HANDOFF_COMPACTION_PROMPT,
    RESEARCH_COMPACTION_PROMPT,
    compaction_prompt,
    format_conversation_for_summary,
)
from agent_core.runtime.loop.tiered_compact import (
    InputTokenGauge,
    InputTokenThresholdPolicy,
    TieredCompactor,
    compaction_trigger_tokens,
)
from agent_core.runtime.spill import SpillStore


def _call(name: str, arguments: dict[str, str], call_id: str = "call-1") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def test_summary_renderer_preserves_tool_arguments() -> None:
    rendered = format_conversation_for_summary([
        assistant_msg("", tool_calls=[_call("web_search", {"q": "exact query"})]),
        tool_msg("result without the query", "call-1"),
    ])

    assert "web_search" in rendered
    assert "exact query" in rendered


def test_prompt_style_is_explicit_and_auto_requires_host_metadata() -> None:
    messages = [assistant_msg("", tool_calls=[_call("bash", {"cmd": "pytest"})])]

    assert compaction_prompt(messages, style="research") == RESEARCH_COMPACTION_PROMPT
    assert compaction_prompt(messages, style="handoff") == HANDOFF_COMPACTION_PROMPT
    assert compaction_prompt(messages, style="auto") == RESEARCH_COMPACTION_PROMPT
    assert compaction_prompt(
        messages,
        style="auto",
        tool_category=lambda _name: "compute",
    ) == HANDOFF_COMPACTION_PROMPT


def test_auto_dispatch_is_the_documented_default() -> None:
    """The default has to be ``auto``, or the routing never fires for any caller.

    Both compactors pass ``prompt_builder=compaction_prompt`` with no style, so a
    ``research`` default made the measured handoff routing unreachable while the
    docstring said it was on. It is behaviour-preserving: without a
    tool-category callback ``auto`` returns the research prompt.
    """
    machine_work = [assistant_msg("", tool_calls=[_call("bash", {"cmd": "pytest"})])]

    assert compaction_prompt(machine_work) == RESEARCH_COMPACTION_PROMPT
    assert compaction_prompt(
        machine_work,
        tool_category=lambda _name: "compute",
    ) == HANDOFF_COMPACTION_PROMPT


@pytest.mark.asyncio
async def test_summary_compactor_uses_host_prompt_builder() -> None:
    class CaptureLLM:
        def __init__(self) -> None:
            self.prompt = ""

        async def chat(self, messages):
            self.prompt = str(messages[0].get("content") or "")
            return SimpleNamespace(content="summary")

    llm = CaptureLLM()
    compactor = LLMSummaryCompactor(
        summary_llm=llm,
        prompt_builder=lambda _messages: "HOST POLICY\n{conversation}",
    )
    history = [user_msg(f"old-{index}") for index in range(8)]

    await compactor.compact(history, keep_recent=2)

    assert llm.prompt.startswith("HOST POLICY")
    assert "old-0" in llm.prompt


def test_projection_uses_real_to_estimated_ratio_for_the_next_request() -> None:
    gauge = InputTokenGauge()
    gauge.tokens = 207_803
    gauge.estimate = 182_283
    policy = InputTokenThresholdPolicy(gauge, 209_715)

    assert gauge.tokens < 209_715
    assert policy.should_compact(
        turn=40,
        messages=[],
        estimated_tokens=232_231,
    )


def test_gauge_reads_usage_and_the_exact_sent_estimate() -> None:
    gauge = InputTokenGauge()
    asyncio.run(gauge.on_llm_response(TurnContext(
        turn=1,
        max_turns=2,
        task_id="task",
        role_id="role",
        ai_text="",
        thinking="",
        tool_calls=[],
        messages=[],
        usage={"prompt_tokens": 12_000},
        metadata={INPUT_ESTIMATE_KEY: 10_000},
    )))

    assert gauge.tokens == 12_000
    assert gauge.estimate == 10_000
    assert gauge.real_to_estimate_scale() == pytest.approx(1.2)


def test_trigger_ratio_is_an_explicit_validated_policy() -> None:
    assert compaction_trigger_tokens(262_144) == 209_715
    assert compaction_trigger_tokens(262_144, 0.65) == 170_393
    with pytest.raises(ValueError, match="between 0 and 1"):
        compaction_trigger_tokens(262_144, 1.0)


def test_budget_consistency_accepts_a_fitting_configuration() -> None:
    assert check_context_budget(
        max_len=262_144,
        max_input_tokens=229_376,
        max_tokens=32_768,
        reasoning_only_max_tokens=16_384,
        label="test",
    ) == []


def test_budget_consistency_reports_independent_violations() -> None:
    problems = check_context_budget(
        max_len=262_144,
        max_input_tokens=229_376,
        max_tokens=65_536,
        reasoning_only_max_tokens=65_536,
        label="test",
    )

    assert any("exceeds max_len" in problem for problem in problems)
    assert any("reasoning watchdog" in problem for problem in problems)


@pytest.mark.asyncio
async def test_tiered_compactor_accepts_the_shared_spill_store(tmp_path: Path) -> None:
    store = SpillStore(tmp_path, "session", visible_root="/spill")
    history = [
        user_msg("research"),
        assistant_msg("", tool_calls=[_call("web_fetch", {}, "old")]),
        tool_msg("old body " * 1_000, "old"),
        assistant_msg("", tool_calls=[_call("web_fetch", {}, "new")]),
        tool_msg("new body " * 1_000, "new"),
    ]
    compactor = TieredCompactor(
        keep_tool_result=1,
        summary_llm=None,
        relief_target=10**9,
        spill_store=store,
    )

    result = await compactor.compact(history, keep_recent=1)

    assert any(message.get("spill_refs") for message in result)
    assert list(store.directory.glob("*.md"))


def test_tiered_rejects_two_spill_owners(tmp_path: Path) -> None:
    store = SpillStore(tmp_path, "session", visible_root="/spill")
    with pytest.raises(ValueError, match="not both"):
        TieredCompactor(
            keep_tool_result=1,
            summary_llm=None,
            relief_target=1,
            spill=lambda _name, _body: None,
            spill_store=store,
        )


@pytest.mark.asyncio
async def test_unseen_results_stay_verbatim_when_the_store_cannot_name_a_path(
    tmp_path: Path,
) -> None:
    """A spill store with no ``visible_root`` recovers nothing, so it protects nothing.

    ``spill_compacted_body`` writes with ``require_visible=True`` and returns
    ``None`` for every body here, so a candidate that shortened the latest
    tool-call turn would discard results the model has not seen once, with no
    path in the manifest to read them back from.
    """
    store = SpillStore(tmp_path, "session")
    latest = "new body " * 400
    history = [
        user_msg("research"),
        assistant_msg("", tool_calls=[_call("web_fetch", {}, "old")]),
        tool_msg("old body " * 400, "old"),
        assistant_msg("", tool_calls=[_call("web_fetch", {}, "new")]),
        tool_msg(latest, "new"),
    ]
    compactor = TieredCompactor(
        keep_tool_result=1,
        summary_llm=None,
        relief_target=600,
        spill_store=store,
    )

    result = await compactor.compact(history, keep_recent=1)

    bodies = {
        message.get("tool_call_id"): str(message.get("content") or "")
        for message in result
        if message.get("role") == "tool"
    }
    assert bodies["new"] == latest
    assert compactor.last_event is not None
    assert not compactor.last_event.selected.startswith("tool_compression_")


@pytest.mark.asyncio
async def test_unseen_results_may_be_compressed_once_a_path_is_recoverable(
    tmp_path: Path,
) -> None:
    store = SpillStore(tmp_path, "session", visible_root="/spill")
    latest = "new body " * 400
    history = [
        user_msg("research"),
        assistant_msg("", tool_calls=[_call("web_fetch", {}, "old")]),
        tool_msg("old body " * 400, "old"),
        assistant_msg("", tool_calls=[_call("web_fetch", {}, "new")]),
        tool_msg(latest, "new"),
    ]
    compactor = TieredCompactor(
        keep_tool_result=1,
        summary_llm=None,
        relief_target=600,
        spill_store=store,
    )

    result = await compactor.compact(history, keep_recent=1)

    bodies = {
        message.get("tool_call_id"): str(message.get("content") or "")
        for message in result
        if message.get("role") == "tool"
    }
    assert len(bodies["new"]) < len(latest)
    assert any(message.get("spill_refs") for message in result)


def test_budget_consistency_shares_the_trigger_ratio() -> None:
    from agent_core.runtime.loop import budget_consistency
    from agent_core.runtime.loop.tiered_compact import DEFAULT_TRIGGER_RATIO

    assert budget_consistency.COMPACTION_TRIGGER_RATIO == DEFAULT_TRIGGER_RATIO


def test_budget_consistency_is_reachable_from_the_package_api() -> None:
    from agent_core.runtime import loop

    assert loop.check_context_budget(
        max_len=262_144,
        max_input_tokens=229_376,
        max_tokens=32_768,
        label="test",
    ) == []
