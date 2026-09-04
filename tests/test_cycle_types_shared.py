"""Tests for agent_core/components/cycle/types.py — round-trip and defaults."""
from __future__ import annotations

from agent_core.components.cycle.types import (
    AuditFinding,
    AuditReport,
    CycleOutput,
    WriterOutput,
)

# ── AuditFinding ────────────────────────────────────────────────────────────


def test_finding_minimal_construction() -> None:
    f = AuditFinding(
        category="missing_section",
        severity="error",
        short_message="Missing intro",
    )
    assert f.detailed_message == ""
    assert f.file is None
    assert f.line is None
    assert f.metadata == {}


def test_finding_round_trip_minimal() -> None:
    f = AuditFinding(
        category="x", severity="warning", short_message="msg",
    )
    restored = AuditFinding.from_dict(f.to_dict())
    assert restored == f


def test_finding_round_trip_full() -> None:
    f = AuditFinding(
        category="citation_invalid",
        severity="error",
        short_message="cite missing",
        detailed_message="full detail",
        file="paper.md",
        line=42,
        snippet="...as Smith et al. show...",
        suggested_action="add citation [SMITH2024]",
        target_role="method_writer",
        metadata={"source_id": "abc-123"},
    )
    restored = AuditFinding.from_dict(f.to_dict())
    assert restored == f


# ── AuditReport ─────────────────────────────────────────────────────────────


def test_report_default_construction() -> None:
    r = AuditReport(verdict="success")
    assert r.findings == []
    assert r.summary == ""
    assert r.confidence == 0.0
    assert r.raw_text == ""


def test_report_round_trip_with_findings() -> None:
    r = AuditReport(
        verdict="iterate",
        findings=[
            AuditFinding(
                category="a", severity="error", short_message="m1",
            ),
            AuditFinding(
                category="b", severity="warning", short_message="m2",
                file="f.py", line=10,
            ),
        ],
        summary="Two issues found in this round.",
        confidence=0.85,
        raw_text="raw auditor output",
        metadata={"audit_round": 2},
    )
    restored = AuditReport.from_dict(r.to_dict())
    assert restored == r


def test_report_json_round_trip() -> None:
    r = AuditReport(
        verdict="success",
        findings=[
            AuditFinding(
                category="info_only",
                severity="info",
                short_message="LGTM",
            ),
        ],
        summary="No blockers.",
        confidence=0.95,
    )
    text = r.to_json()
    restored = AuditReport.from_json(text)
    assert restored == r


def test_report_from_dict_tolerates_missing_optional_fields() -> None:
    """Older serialized forms missing newer fields should still load."""
    minimal = {"verdict": "iterate", "findings": []}
    r = AuditReport.from_dict(minimal)
    assert r.verdict == "iterate"
    assert r.findings == []
    assert r.summary == ""
    assert r.confidence == 0.0


# ── WriterOutput ────────────────────────────────────────────────────────────


def test_writer_output_round_trip_drops_loop_result() -> None:
    """loop_result is intentionally not serialized."""
    wo = WriterOutput(
        content="draft text",
        files=["sections/intro.md", "sections/method.md"],
        message_count=42,
        metadata={"turns_used": 12},
        loop_result={"unserializable": object()},
    )
    restored = WriterOutput.from_dict(wo.to_dict())
    assert restored.content == wo.content
    assert restored.files == wo.files
    assert restored.message_count == wo.message_count
    assert restored.metadata == wo.metadata
    assert restored.loop_result is None  # dropped on round-trip


# ── CycleOutput ─────────────────────────────────────────────────────────────


def test_cycle_output_empty_history() -> None:
    co = CycleOutput(
        success=False,
        rounds_used=0,
        final_writer_output=None,
        final_audit=None,
        reason="never_started",
    )
    restored = CycleOutput.from_dict(co.to_dict())
    assert restored == co


def test_cycle_output_full_round_trip() -> None:
    wo1 = WriterOutput(content="draft v1", files=["a.md"], message_count=10)
    a1 = AuditReport(
        verdict="iterate",
        findings=[
            AuditFinding(
                category="missing_section",
                severity="error",
                short_message="add intro",
            ),
        ],
        summary="Need an intro.",
    )
    wo2 = WriterOutput(content="draft v2", files=["a.md"], message_count=20)
    a2 = AuditReport(verdict="success", summary="LGTM.", confidence=0.9)
    co = CycleOutput(
        success=True,
        rounds_used=2,
        final_writer_output=wo2,
        final_audit=a2,
        history=[(wo1, a1), (wo2, a2)],
        reason="verdict=success",
    )
    text = co.to_json()
    restored = CycleOutput.from_json(text)
    assert restored.success == co.success
    assert restored.rounds_used == co.rounds_used
    assert restored.final_audit == a2
    assert len(restored.history) == 2
    assert restored.history[0][1] == a1
    assert restored.history[1][1] == a2
    assert restored.reason == "verdict=success"
