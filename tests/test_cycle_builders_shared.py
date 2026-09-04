"""Tests for SessionBackedWriter / SessionBackedAuditor.

Mocked-AgentBus tests covering:

- writer creates exactly ONE session reused across rounds (FR9)
- auditor creates a FRESH session per round (FR9)
- ConcludePhaseObserver auto-injected into auditor (FR7)
- llm_timeout propagated to create_session (FR8)
- tools_override defaults to [] for auditor; explicit list becomes whitelist
- llm_override propagated for both writer and auditor
- audit JSON parsing (fenced + bare + parse-failure fallback)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from agent_core.components.agent_bus.models import (
    CollectResult,
    SubAgentResult,
)
from agent_core.components.cycle.builders import (
    MajorityVoteAuditor,
    ScoreThresholdAuditor,
    SessionBackedAuditor,
    SessionBackedWriter,
    _confidence_from_score,
    _extract_first_number,
    _extract_first_str,
    _parse_audit_report,
    select_best_attempt,
)
from agent_core.components.cycle.types import AuditFinding, AuditReport, WriterOutput
from agent_core.components.observers.conclude_phase_observer import ConcludePhaseObserver


def _ok_result(content: str = "ok", metadata: dict | None = None) -> SubAgentResult:
    return SubAgentResult(
        question="x",
        role_id="any",
        final_content=content,
        success=True,
        metadata=metadata or {},
    )


def _make_bus(*, content: str = "ok", session_id: str = "task-1::w") -> Any:
    bus = AsyncMock()
    bus.create_session = AsyncMock(return_value=session_id)
    bus.submit_task_to_session = AsyncMock(return_value="job-1")
    bus.collect = AsyncMock(
        return_value=CollectResult(completed=[_ok_result(content)]),
    )
    return bus


# ── SessionBackedWriter ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_writer_creates_one_session_reused_across_rounds(
    tmp_path: Path,
) -> None:
    bus = _make_bus()
    writer = SessionBackedWriter(
        bus=bus, task_id="t1", role_id="paper_writer", name="paper_w",
        system_prompt="you write papers", initial_prompt="write a paper",
        work_dir=tmp_path, output_check=lambda _: [],
        llm_timeout=420,
    )

    await writer.generate(None, 0, "")
    await writer.generate(None, 1, "feedback round 1")
    await writer.generate(None, 2, "feedback round 2")

    # Exactly ONE create_session call across all rounds
    assert bus.create_session.await_count == 1
    # Three submissions (one per round)
    assert bus.submit_task_to_session.await_count == 3


@pytest.mark.asyncio
async def test_writer_round_0_submits_initial_prompt(tmp_path: Path) -> None:
    bus = _make_bus()
    writer = SessionBackedWriter(
        bus=bus, task_id="t", role_id="r", name="w",
        system_prompt="sp", initial_prompt="write X",
        work_dir=tmp_path, output_check=lambda _: [],
    )
    await writer.generate(None, 0, "")
    args, _kwargs = bus.submit_task_to_session.await_args
    submitted_prompt = args[1]
    assert "write X" in submitted_prompt


@pytest.mark.asyncio
async def test_writer_round_n_submits_feedback_plus_revision(
    tmp_path: Path,
) -> None:
    bus = _make_bus()
    writer = SessionBackedWriter(
        bus=bus, task_id="t", role_id="r", name="w",
        system_prompt="sp", initial_prompt="write X",
        work_dir=tmp_path, output_check=lambda _: [],
        revision_instruction="REVISE_NOW",
    )
    await writer.generate(None, 1, "FEEDBACK_HERE")
    args, _ = bus.submit_task_to_session.await_args
    submitted_prompt = args[1]
    assert "FEEDBACK_HERE" in submitted_prompt
    assert "REVISE_NOW" in submitted_prompt


@pytest.mark.asyncio
async def test_writer_propagates_llm_timeout_and_override(
    tmp_path: Path,
) -> None:
    bus = _make_bus()
    fake_llm = object()
    writer = SessionBackedWriter(
        bus=bus, task_id="t", role_id="r", name="w",
        system_prompt="sp", initial_prompt="x",
        work_dir=tmp_path, output_check=lambda _: [],
        llm_timeout=999, llm_override=fake_llm,
    )
    await writer.generate(None, 0, "")
    _, kwargs = bus.create_session.await_args
    assert kwargs["llm_timeout"] == 999
    assert kwargs["llm_override"] is fake_llm


@pytest.mark.asyncio
async def test_writer_returns_writer_output_with_content_and_metadata(
    tmp_path: Path,
) -> None:
    bus = AsyncMock()
    bus.create_session = AsyncMock(return_value="t::w")
    bus.submit_task_to_session = AsyncMock(return_value="j1")
    bus.collect = AsyncMock(
        return_value=CollectResult(completed=[
            _ok_result(
                "draft v1",
                metadata={"files": ["intro.md"], "message_count": 12},
            ),
        ]),
    )
    writer = SessionBackedWriter(
        bus=bus, task_id="t", role_id="r", name="w",
        system_prompt="sp", initial_prompt="x",
        work_dir=tmp_path, output_check=lambda _: [],
    )
    out = await writer.generate(None, 0, "")
    assert out.content == "draft v1"
    assert out.files == ["intro.md"]
    assert out.message_count == 12


# ── SessionBackedAuditor ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_auditor_creates_fresh_session_per_round(
    tmp_path: Path,
) -> None:
    del tmp_path  # not used here but shared fixture style
    # Each call to create_session returns a per-round id
    sid_returns = iter(["t::a_R0", "t::a_R1", "t::a_R2"])
    bus = AsyncMock()
    bus.create_session = AsyncMock(side_effect=lambda **_: next(sid_returns))
    bus.submit_task_to_session = AsyncMock(return_value="job")
    bus.collect = AsyncMock(
        return_value=CollectResult(completed=[
            _ok_result('{"verdict": "iterate", "findings": []}'),
        ]),
    )
    auditor = SessionBackedAuditor(
        bus=bus, task_id="t", role_id="auditor",
        system_prompt="emit JSON",
    )

    for r in range(3):
        await auditor.verify(WriterOutput(content="x"), r)

    assert bus.create_session.await_count == 3
    # Fresh names per round
    names = [
        call.kwargs["name"] for call in bus.create_session.await_args_list
    ]
    assert names == ["auditor_R0", "auditor_R1", "auditor_R2"]


@pytest.mark.asyncio
async def test_auditor_injects_conclude_phase_observer() -> None:
    bus = _make_bus(content='{"verdict": "iterate", "findings": []}')
    auditor = SessionBackedAuditor(
        bus=bus, task_id="t", role_id="r",
        system_prompt="sp",
        conclude_ratio=0.7,
    )
    await auditor.verify(WriterOutput(content="x"), 0)
    _, kwargs = bus.submit_task_to_session.await_args
    obs_list = kwargs["observers"]
    assert len(obs_list) >= 1
    has_conclude = any(
        isinstance(o, ConcludePhaseObserver) for o in obs_list
    )
    assert has_conclude


@pytest.mark.asyncio
async def test_auditor_default_tools_is_empty_list() -> None:
    bus = _make_bus(content='{"verdict": "success", "findings": []}')
    auditor = SessionBackedAuditor(
        bus=bus, task_id="t", role_id="r", system_prompt="sp",
    )
    await auditor.verify(WriterOutput(content="x"), 0)
    _, kwargs = bus.create_session.await_args
    # Default is empty tool set — Codex-style hard restriction
    assert kwargs["tools_override"] == []


@pytest.mark.asyncio
async def test_auditor_explicit_tools_become_whitelist() -> None:
    bus = _make_bus(content='{"verdict": "success", "findings": []}')
    fake_tool = object()
    auditor = SessionBackedAuditor(
        bus=bus, task_id="t", role_id="r", system_prompt="sp",
        tools=[fake_tool],  # type: ignore[list-item]
    )
    await auditor.verify(WriterOutput(content="x"), 0)
    _, kwargs = bus.create_session.await_args
    assert kwargs["tools_override"] == [fake_tool]


@pytest.mark.asyncio
async def test_auditor_propagates_llm_timeout_and_override() -> None:
    bus = _make_bus(content='{"verdict": "success", "findings": []}')
    fake_llm = object()
    auditor = SessionBackedAuditor(
        bus=bus, task_id="t", role_id="r", system_prompt="sp",
        llm_timeout=777, llm_override=fake_llm,
    )
    await auditor.verify(WriterOutput(content="x"), 0)
    _, kwargs = bus.create_session.await_args
    assert kwargs["llm_timeout"] == 777
    assert kwargs["llm_override"] is fake_llm


@pytest.mark.asyncio
async def test_auditor_handles_runtime_failure_as_iterate() -> None:
    """If AgentBus returns no completed/failed results, auditor synthesizes."""
    bus = AsyncMock()
    bus.create_session = AsyncMock(return_value="t::a")
    bus.submit_task_to_session = AsyncMock(return_value="job")
    bus.collect = AsyncMock(return_value=CollectResult())
    auditor = SessionBackedAuditor(
        bus=bus, task_id="t", role_id="r", system_prompt="sp",
    )
    report = await auditor.verify(WriterOutput(content="x"), 0)
    assert report.verdict == "iterate"
    assert any(
        f.category == "auditor_runtime_error" for f in report.findings
    )


# ── _parse_audit_report ─────────────────────────────────────────────────────


def test_parse_fenced_json_block() -> None:
    raw = """Some prose first.

```json
{
  "verdict": "iterate",
  "findings": [
    {"category": "x", "severity": "error", "short_message": "m"}
  ],
  "summary": "needs work",
  "confidence": 0.9
}
```

Trailing prose.
"""
    r = _parse_audit_report(raw)
    assert r.verdict == "iterate"
    assert len(r.findings) == 1
    assert r.findings[0].category == "x"
    assert r.summary == "needs work"
    assert r.raw_text == raw  # preserved for debugging


def test_parse_bare_json_object() -> None:
    raw = 'prefix prose {"verdict": "success", "findings": []} suffix'
    r = _parse_audit_report(raw)
    assert r.verdict == "success"
    assert r.findings == []


def test_parse_invalid_json_synthesizes_parse_failure() -> None:
    r = _parse_audit_report("totally non-JSON output here")
    assert r.verdict == "iterate"
    assert any(
        f.category == "auditor_parse_failure" for f in r.findings
    )
    assert r.metadata.get("reason") == "auditor_parse_failure"


def test_parse_empty_response_synthesizes_parse_failure() -> None:
    r = _parse_audit_report("")
    assert r.verdict == "iterate"
    assert any(
        f.category == "auditor_parse_failure" for f in r.findings
    )


def test_parse_schema_mismatch_synthesizes_schema_error() -> None:
    """JSON parses but is missing the required 'verdict' key."""
    r = _parse_audit_report('{"not_a_verdict": "x"}')
    assert r.verdict == "iterate"
    assert any(
        f.category == "auditor_schema_mismatch" for f in r.findings
    )


# ── ScoreThresholdAuditor ───────────────────────────────────────────────────


def _make_grader_bus(content: str) -> Any:
    bus = AsyncMock()
    bus.create_session = AsyncMock(return_value="t::g")
    bus.submit_task_to_session = AsyncMock(return_value="job-g")
    bus.collect = AsyncMock(
        return_value=CollectResult(completed=[_ok_result(content)]),
    )
    return bus


@pytest.mark.asyncio
async def test_score_threshold_pass_emits_success_verdict() -> None:
    bus = _make_grader_bus(
        '{"score": 7, "feedback": "looks great"}',
    )
    auditor = ScoreThresholdAuditor(
        bus=bus, task_id="t", role_id="grader",
        system_prompt="grade per rubric",
        pass_threshold=6,
    )
    report = await auditor.verify(WriterOutput(content="proof v1"), 0)
    assert report.verdict == "success"
    assert report.metadata["score"] == 7.0
    assert report.metadata["passed"] is True
    assert report.confidence == pytest.approx(1.0)
    assert report.findings[0].category == "grade"
    assert report.findings[0].severity == "info"
    assert "looks great" in report.summary


@pytest.mark.asyncio
async def test_score_threshold_fail_emits_iterate_verdict() -> None:
    bus = _make_grader_bus(
        '{"score": 1, "feedback": "missing case analysis"}',
    )
    auditor = ScoreThresholdAuditor(
        bus=bus, task_id="t", role_id="grader",
        system_prompt="grade per rubric",
        pass_threshold=6,
    )
    report = await auditor.verify(WriterOutput(content="proof"), 0)
    assert report.verdict == "iterate"
    assert report.metadata["score"] == 1.0
    assert report.metadata["passed"] is False
    assert report.findings[0].severity == "warning"
    assert "missing case analysis" in report.summary


@pytest.mark.asyncio
async def test_score_threshold_alternate_keys_points_explanation() -> None:
    """IMO-GVR's grader emits {points, explanation} not {score, feedback}."""
    bus = _make_grader_bus(
        '{"points": 6, "explanation": "minor gap"}',
    )
    auditor = ScoreThresholdAuditor(
        bus=bus, task_id="t", role_id="grader",
        system_prompt="grade",
        pass_threshold=6,
    )
    report = await auditor.verify(WriterOutput(content="x"), 0)
    assert report.verdict == "success"
    assert report.metadata["score"] == 6.0
    assert report.metadata["feedback"] == "minor gap"


@pytest.mark.asyncio
async def test_score_threshold_always_iterate_for_run_k_mode() -> None:
    """pass_verdict='iterate' lets the cycle run all max_rounds even on
    a perfect score — parity with MiroVerifier IMO-GVR."""
    bus = _make_grader_bus('{"score": 7, "feedback": "perfect"}')
    auditor = ScoreThresholdAuditor(
        bus=bus, task_id="t", role_id="grader",
        system_prompt="grade",
        pass_threshold=6,
        pass_verdict="iterate",  # never terminate via verdict
    )
    report = await auditor.verify(WriterOutput(content="x"), 0)
    assert report.verdict == "iterate"
    assert report.metadata["score"] == 7.0
    assert report.metadata["passed"] is True


@pytest.mark.asyncio
async def test_score_threshold_parse_failure_synthesizes_finding() -> None:
    bus = _make_grader_bus("totally non-JSON garbage")
    auditor = ScoreThresholdAuditor(
        bus=bus, task_id="t", role_id="grader",
        system_prompt="grade",
    )
    report = await auditor.verify(WriterOutput(content="x"), 0)
    assert report.verdict == "iterate"
    assert any(
        f.category == "grader_parse_failure" for f in report.findings
    )
    assert report.metadata.get("reason") == "grader_parse_failure"


@pytest.mark.asyncio
async def test_score_threshold_score_missing_synthesizes_finding() -> None:
    """JSON parses but has no recognised score key."""
    bus = _make_grader_bus('{"verdict": "ok"}')
    auditor = ScoreThresholdAuditor(
        bus=bus, task_id="t", role_id="grader",
        system_prompt="grade",
    )
    report = await auditor.verify(WriterOutput(content="x"), 0)
    assert report.verdict == "iterate"
    assert any(
        f.category == "grader_score_missing" for f in report.findings
    )


@pytest.mark.asyncio
async def test_score_threshold_string_score_is_coerced() -> None:
    """Some models emit '"score": "7"' — should still parse."""
    bus = _make_grader_bus('{"score": "7", "feedback": "ok"}')
    auditor = ScoreThresholdAuditor(
        bus=bus, task_id="t", role_id="grader",
        system_prompt="grade",
        pass_threshold=6,
    )
    report = await auditor.verify(WriterOutput(content="x"), 0)
    assert report.verdict == "success"
    assert report.metadata["score"] == 7.0


@pytest.mark.asyncio
async def test_score_threshold_creates_fresh_session_per_round() -> None:
    """Same per-round freshness invariant as SessionBackedAuditor."""
    sid_returns = iter(["t::g_R0", "t::g_R1", "t::g_R2"])
    bus = AsyncMock()
    bus.create_session = AsyncMock(side_effect=lambda **_: next(sid_returns))
    bus.submit_task_to_session = AsyncMock(return_value="job")
    bus.collect = AsyncMock(
        return_value=CollectResult(completed=[
            _ok_result('{"score": 5, "feedback": "fix x"}'),
        ]),
    )
    auditor = ScoreThresholdAuditor(
        bus=bus, task_id="t", role_id="grader",
        system_prompt="grade",
    )
    for r in range(3):
        await auditor.verify(WriterOutput(content="x"), r)
    assert bus.create_session.await_count == 3
    names = [
        call.kwargs["name"] for call in bus.create_session.await_args_list
    ]
    assert names == ["grader_R0", "grader_R1", "grader_R2"]


@pytest.mark.asyncio
async def test_score_threshold_default_tools_is_empty_list() -> None:
    bus = _make_grader_bus('{"score": 7, "feedback": "ok"}')
    auditor = ScoreThresholdAuditor(
        bus=bus, task_id="t", role_id="grader", system_prompt="grade",
    )
    await auditor.verify(WriterOutput(content="x"), 0)
    _, kwargs = bus.create_session.await_args
    assert kwargs["tools_override"] == []


@pytest.mark.asyncio
async def test_score_threshold_injects_conclude_observer() -> None:
    bus = _make_grader_bus('{"score": 7, "feedback": "ok"}')
    auditor = ScoreThresholdAuditor(
        bus=bus, task_id="t", role_id="grader", system_prompt="grade",
        conclude_ratio=0.6,
    )
    await auditor.verify(WriterOutput(content="x"), 0)
    _, kwargs = bus.submit_task_to_session.await_args
    obs_list = kwargs["observers"]
    assert any(isinstance(o, ConcludePhaseObserver) for o in obs_list)


@pytest.mark.asyncio
async def test_score_threshold_handles_runtime_failure_as_fail_verdict() -> None:
    """If AgentBus.collect returns nothing, audit synthesises a runtime
    error report (verdict 'iterate' to keep the cycle alive)."""
    bus = AsyncMock()
    bus.create_session = AsyncMock(return_value="t::g")
    bus.submit_task_to_session = AsyncMock(return_value="job")
    bus.collect = AsyncMock(return_value=CollectResult())
    auditor = ScoreThresholdAuditor(
        bus=bus, task_id="t", role_id="grader", system_prompt="grade",
    )
    report = await auditor.verify(WriterOutput(content="x"), 0)
    assert report.verdict == "iterate"
    assert any(
        f.category == "auditor_runtime_error" for f in report.findings
    )


def test_score_threshold_rejects_invalid_score_range() -> None:
    bus = AsyncMock()
    with pytest.raises(ValueError, match="score_range"):
        ScoreThresholdAuditor(
            bus=bus, task_id="t", role_id="g", system_prompt="x",
            score_range=(7, 0),  # backwards
        )


def test_score_threshold_rejects_empty_score_keys() -> None:
    bus = AsyncMock()
    with pytest.raises(ValueError, match="score_keys"):
        ScoreThresholdAuditor(
            bus=bus, task_id="t", role_id="g", system_prompt="x",
            score_keys=(),
        )


# ── Score / selection helpers ───────────────────────────────────────────────


def test_extract_first_number_prefers_first_key() -> None:
    assert _extract_first_number(
        {"score": 7, "points": 3}, ("score", "points"),
    ) == 7.0
    assert _extract_first_number(
        {"points": 3}, ("score", "points"),
    ) == 3.0


def test_extract_first_number_skips_bool_and_non_numeric() -> None:
    # bool is a subclass of int; ScoreThresholdAuditor must not treat True as 1.
    assert _extract_first_number({"score": True}, ("score",)) is None
    assert _extract_first_number({"score": "abc"}, ("score",)) is None
    assert _extract_first_number({}, ("score",)) is None


def test_extract_first_str_prefers_non_empty() -> None:
    assert _extract_first_str(
        {"feedback": "", "explanation": "real"},
        ("feedback", "explanation"),
    ) == "real"
    assert _extract_first_str({}, ("feedback",)) is None
    # whitespace-only counts as empty
    assert _extract_first_str(
        {"feedback": "   "}, ("feedback",),
    ) is None


def test_confidence_from_score_clamps() -> None:
    assert _confidence_from_score(7, (0, 7)) == 1.0
    assert _confidence_from_score(0, (0, 7)) == 0.0
    assert _confidence_from_score(3.5, (0, 7)) == pytest.approx(0.5)
    # Out-of-range scores clamp.
    assert _confidence_from_score(8, (0, 7)) == 1.0
    assert _confidence_from_score(-1, (0, 7)) == 0.0
    # None range disables.
    assert _confidence_from_score(7, None) == 0.0


def _hist_entry(content: str, score: float | None) -> tuple[WriterOutput, AuditReport]:
    metadata = {"score": score} if score is not None else {}
    return (
        WriterOutput(content=content),
        AuditReport(
            verdict="iterate",
            findings=[
                AuditFinding(
                    category="grade", severity="info",
                    short_message=f"score={score}",
                ),
            ],
            metadata=metadata,
        ),
    )


def test_select_best_attempt_picks_highest_score() -> None:
    history = [
        _hist_entry("v1", 1.0),
        _hist_entry("v2", 7.0),
        _hist_entry("v3", 6.0),
    ]
    idx, w, a = select_best_attempt(history)
    assert idx == 1
    assert w.content == "v2"
    assert a.metadata["score"] == 7.0


def test_select_best_attempt_ties_go_to_latest() -> None:
    """IMO-GVR rule: when scores tie, the *latest* attempt wins
    (revisions are at least as good as earlier ones)."""
    history = [
        _hist_entry("v1", 7.0),
        _hist_entry("v2", 7.0),
        _hist_entry("v3", 7.0),
    ]
    idx, w, _a = select_best_attempt(history)
    assert idx == 2
    assert w.content == "v3"


def test_select_best_attempt_unscored_audit_treated_as_minimum() -> None:
    """Unscored audits should never beat scored ones."""
    history = [
        _hist_entry("v1", 1.0),
        _hist_entry("v2", None),  # no score in metadata
    ]
    idx, w, _a = select_best_attempt(history)
    assert idx == 0
    assert w.content == "v1"


def test_select_best_attempt_empty_history_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        select_best_attempt([])


# ── MajorityVoteAuditor ────────────────────────────────────────────────────


class _FakeAuditor:
    """Returns scripted AuditReports in order. Each call pops one."""

    role_id = "fake_judge"

    def __init__(self, reports: list) -> None:
        self._reports = list(reports)
        self.calls = 0

    async def verify(self, writer_output, round_num):
        self.calls += 1
        item = self._reports.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _grade_report(score: float, verdict: str = "iterate") -> AuditReport:
    return AuditReport(
        verdict=verdict,
        findings=[
            AuditFinding(
                category="grade", severity="info",
                short_message=f"score={score}",
                metadata={"score": score},
            ),
        ],
        summary=f"feedback for score {score}",
        confidence=score / 7.0,
        metadata={"score": score, "feedback": f"feedback for score {score}"},
    )


@pytest.mark.asyncio
async def test_majority_vote_aggregates_median_score() -> None:
    base = _FakeAuditor([
        _grade_report(7.0),
        _grade_report(6.0),
        _grade_report(7.0),
    ])
    voter = MajorityVoteAuditor(base=base, n_votes=3)
    report = await voter.verify(WriterOutput(content="x"), 0)
    assert report.metadata["score"] == 7.0  # median([7,6,7]) = 7
    assert report.metadata["individual_scores"] == [7.0, 6.0, 7.0]
    assert report.confidence == 1.0
    assert report.metadata["n_votes"] == 3
    assert report.metadata["n_succeeded"] == 3


@pytest.mark.asyncio
async def test_majority_vote_majority_verdict() -> None:
    base = _FakeAuditor([
        _grade_report(7.0, verdict="success"),
        _grade_report(6.0, verdict="success"),
        _grade_report(1.0, verdict="iterate"),
    ])
    voter = MajorityVoteAuditor(base=base, n_votes=3)
    report = await voter.verify(WriterOutput(content="x"), 0)
    assert report.verdict == "success"
    assert report.metadata["individual_verdicts"] == [
        "success", "success", "iterate",
    ]


@pytest.mark.asyncio
async def test_majority_vote_partial_failures_lower_confidence() -> None:
    base = _FakeAuditor([
        _grade_report(7.0),
        RuntimeError("judge 2 boom"),
        _grade_report(6.0),
    ])
    voter = MajorityVoteAuditor(base=base, n_votes=3)
    report = await voter.verify(WriterOutput(content="x"), 0)
    assert report.confidence == pytest.approx(2 / 3)
    assert report.metadata["n_succeeded"] == 2
    assert report.metadata["n_failed"] == 1
    assert report.metadata["score"] == 6.5  # median([7, 6])


@pytest.mark.asyncio
async def test_majority_vote_all_failures_synthesizes_report() -> None:
    base = _FakeAuditor([
        RuntimeError("a"),
        RuntimeError("b"),
        RuntimeError("c"),
    ])
    voter = MajorityVoteAuditor(base=base, n_votes=3)
    report = await voter.verify(WriterOutput(content="x"), 0)
    assert report.verdict == "iterate"
    assert any(
        f.category == "majority_vote_failed" for f in report.findings
    )
    assert report.metadata["n_succeeded"] == 0
    assert report.confidence == 0.0


@pytest.mark.asyncio
async def test_majority_vote_calls_base_n_times_in_parallel() -> None:
    base = _FakeAuditor([
        _grade_report(7.0), _grade_report(7.0), _grade_report(7.0),
        _grade_report(6.0), _grade_report(6.0), _grade_report(6.0),
    ])
    voter = MajorityVoteAuditor(base=base, n_votes=3)
    await voter.verify(WriterOutput(content="x"), 0)
    assert base.calls == 3
    await voter.verify(WriterOutput(content="y"), 1)
    assert base.calls == 6


@pytest.mark.asyncio
async def test_majority_vote_custom_aggregator() -> None:
    """statistics.mean instead of median."""
    import statistics as _s
    base = _FakeAuditor([
        _grade_report(7.0), _grade_report(6.0), _grade_report(2.0),
    ])
    voter = MajorityVoteAuditor(
        base=base, n_votes=3, score_aggregator=_s.mean,
    )
    report = await voter.verify(WriterOutput(content="x"), 0)
    assert report.metadata["score"] == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_majority_vote_inherits_role_id() -> None:
    base = _FakeAuditor([_grade_report(7.0)])
    voter = MajorityVoteAuditor(base=base, n_votes=1)
    assert voter.role_id == "fake_judge"


def test_majority_vote_invalid_n_rejected() -> None:
    base = _FakeAuditor([])
    with pytest.raises(ValueError, match="n_votes"):
        MajorityVoteAuditor(base=base, n_votes=0)
