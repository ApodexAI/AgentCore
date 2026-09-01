"""Tiered context compaction: protect-set, real-token gauge/policy, escalation."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_core.llm import LLMResponse
from agent_core.loop_types import LoopConfig, TurnContext
from agent_core.messages import system_msg, tool_msg
from agent_core.runtime.loop.agent_loop import run_agent_loop
from agent_core.runtime.loop.compact import (
    OMITTED_TOOL_RESULT_PLACEHOLDER,
    KeepLastNToolResultsCompactor,
    estimate_tokens,
)
from agent_core.runtime.loop.tiered_compact import (
    InputTokenGauge,
    InputTokenThresholdPolicy,
    TieredCompactor,
)


def _ai(tid: str, name: str) -> dict:
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": tid, "function": {"name": name, "arguments": "{}"}}]}


def _msgs() -> list[dict]:
    m = [system_msg("S"), _ai("c1", "collect_reports"), tool_msg("REPORT-" + "x" * 500, "c1")]
    for i in range(4):
        m += [_ai(f"w{i}", "web_search"), tool_msg("SR-" + "y" * 500, f"w{i}")]
    return m


def test_keeplastn_protects_named_tools():
    # keep last 1 tool result, but never blank collect_reports regardless of age.
    k = KeepLastNToolResultsCompactor(
        keep_tool_result=1, protect_tool_names=frozenset({"collect_reports"}))
    out = k.compact(_msgs(), 0)
    bodies = [m["content"] for m in out if m.get("role") == "tool"]
    assert any(b.startswith("REPORT-") for b in bodies)  # protected, kept full
    blanked = sum(1 for b in bodies if b.startswith(OMITTED_TOOL_RESULT_PLACEHOLDER))
    assert blanked == 3  # 3 of 4 web results blanked


def test_keeplastn_default_has_no_protect():
    # Backward compatible: no protect set → old collect_reports IS blanked by age.
    k = KeepLastNToolResultsCompactor(keep_tool_result=1)
    out = k.compact(_msgs(), 0)
    bodies = [m["content"] for m in out if m.get("role") == "tool"]
    assert not any(b.startswith("REPORT-") for b in bodies)


def test_input_token_gauge_and_policy():
    g = InputTokenGauge()
    ctx = TurnContext(turn=1, max_turns=9, task_id="t", role_id="r", ai_text="",
                      thinking="", tool_calls=[], messages=[],
                      usage={"prompt_tokens": 210_000}, metadata={})
    asyncio.run(g.on_llm_response(ctx))
    pol = InputTokenThresholdPolicy(g, int(262_144 * 0.8))
    assert pol.should_compact(1, [], 0) is True          # 210k > 209k
    g.tokens = 100_000
    assert pol.should_compact(1, [], 0) is False


class _FakeLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, messages, **kw):
        self.calls += 1
        class _R:
            content = "SUMMARY"
        return _R()


def test_tiered_escalates_only_when_tier1_insufficient():
    async def run():
        # relief huge → Tier1 alone suffices, Tier2 (LLM summary) must NOT fire.
        t1 = TieredCompactor(keep_tool_result=1, summary_llm=_FakeLLM(), relief_target=10**9)
        o1 = await t1.compact(_msgs(), 6)
        assert not any("SUMMARY" in (m.get("content") or "") for m in o1)
        # relief 0 → Tier1 never enough → escalate to Tier2 summary.
        llm = _FakeLLM()
        t2 = TieredCompactor(keep_tool_result=1, summary_llm=llm, relief_target=0)
        await t2.compact(_msgs(), 6)
        # Escalation means Tier 2 is evaluated. The compactor still returns the
        # smallest candidate, which may be deterministic compression.
        assert llm.calls == 1

    asyncio.run(run())


def test_tiered_calibration_escalates_via_real_token_scale():
    # The relief check is in REAL tokens: with a gauge reporting real >> estimate,
    # a Tier1 result that the RAW estimate would clear still escalates to Tier2.
    async def run():
        tier1_out = KeepLastNToolResultsCompactor(keep_tool_result=1).compact(_msgs(), 6)
        relief = estimate_tokens(tier1_out) + 10  # raw est(out) clears this
        # No gauge → raw estimate path → Tier1 alone, NO summary.
        t_raw = TieredCompactor(keep_tool_result=1, summary_llm=_FakeLLM(), relief_target=relief)
        o_raw = await t_raw.compact(_msgs(), 6)
        assert not any("SUMMARY" in (m.get("content") or "") for m in o_raw)
        # Gauge saw real = 50× the estimate → scale pushes over relief → Tier2 fires.
        g = InputTokenGauge()
        g.estimate = estimate_tokens(_msgs())
        g.tokens = g.estimate * 50
        llm = _FakeLLM()
        t_cal = TieredCompactor(
            keep_tool_result=1, summary_llm=llm, relief_target=relief, gauge=g)
        await t_cal.compact(_msgs(), 6)
        assert llm.calls == 1

    asyncio.run(run())


def _tool(name: str):
    t = MagicMock()
    t.name = name
    t.ainvoke = AsyncMock(return_value="ok")
    return t


def _resp(content: str, tool_calls: list[dict] | None, prompt_tokens: int) -> LLMResponse:
    wire = [{"id": tc["id"], "type": "function",
             "function": {"name": tc["name"], "arguments": "{}"}}
            for tc in (tool_calls or [])]
    return LLMResponse(content=content, tool_calls=wire,
                       usage={"prompt_tokens": prompt_tokens, "completion_tokens": 5})


class _SeqUsageLLM:
    """Fake LLM returning queued responses (last repeats) carrying usage."""
    model = "fake-usage"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._r = responses
        self._i = 0

    async def chat(self, messages, **kw):
        r = self._r[min(self._i, len(self._r) - 1)]
        self._i += 1
        return r


class _SpyCompactor:
    def __init__(self) -> None:
        self.calls = 0

    async def compact(self, messages, keep_recent):
        self.calls += 1
        return messages


@pytest.mark.asyncio
async def test_gauge_feeds_policy_inside_run_agent_loop():
    # End-to-end wiring contract: the critical gauge updates in on_llm_response
    # (loop step 6) BEFORE the policy reads it in the compaction step (step 10),
    # so a real-token trigger actually drives compaction through the live loop.
    gauge = InputTokenGauge()
    spy = _SpyCompactor()
    llm = _SeqUsageLLM([
        _resp("use tool", [{"id": "c1", "name": "web_search"}], prompt_tokens=5000),
        _resp("done", None, prompt_tokens=5000),
    ])
    cfg = LoopConfig(max_turns=5, no_tool_max_retries=1,
                     compactor=spy, compaction_policy=InputTokenThresholdPolicy(gauge, 1000))
    await run_agent_loop(system_prompt="s", user_message="q", llm=llm,
                         tools=[_tool("web_search")], config=cfg, observers=[gauge])
    assert gauge.tokens == 5000  # gauge captured the real prompt_tokens
    # Only the threshold policy can fire → compaction running proves the gauge
    # was fed (step 6) before the policy read it (step 10) within the live loop.
    assert spy.calls >= 1
