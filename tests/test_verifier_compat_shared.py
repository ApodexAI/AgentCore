"""Round-trip Verdict ↔ AuditReport via _compat helpers."""

from __future__ import annotations

import inspect

import pytest

from agent_core.components.cycle.types import (
    AuditFinding,
    AuditReport,
    WriterOutput,
)
from agent_core.components.verifier import (
    Finding,
    Verdict,
    VerifierContext,
    cycle_auditor_from_verifier,
)
from agent_core.components.verifier._compat import (
    audit_report_from_verdict,
    verdict_from_audit_report,
)


def test_audit_report_to_verdict_passed():
    report = AuditReport(
        verdict="success",
        findings=[
            AuditFinding(
                category="missing_section",
                severity="warning",
                short_message="missing intro",
                file="paper.md",
            )
        ],
        summary="Looks fine overall.",
        confidence=0.92,
        raw_text="<raw>",
        metadata={"reviewer": "auditor-3"},
    )
    v = verdict_from_audit_report(report)
    assert v.passed is True
    assert v.score == 0.92
    assert v.reasoning == "Looks fine overall."
    assert len(v.findings) == 1
    assert v.findings[0].severity == "warning"
    assert v.findings[0].message == "missing intro"
    assert v.findings[0].location == "paper.md"
    assert v.metadata["_cycle_verdict"] == "success"
    assert v.metadata["_cycle_raw_text"] == "<raw>"
    assert v.metadata["reviewer"] == "auditor-3"


def test_audit_report_to_verdict_failed():
    report = AuditReport(verdict="iterate", confidence=0.4)
    v = verdict_from_audit_report(report)
    assert v.passed is False


def test_verdict_to_audit_report_roundtrip_preserves_label():
    original = AuditReport(
        verdict="abandon",
        findings=[
            AuditFinding(
                category="parse_failure",
                severity="error",
                short_message="bad json",
            )
        ],
        summary="Can't proceed.",
        confidence=0.1,
        raw_text="<raw>",
        metadata={"k": "v"},
    )
    bridged = audit_report_from_verdict(verdict_from_audit_report(original))
    assert bridged.verdict == "abandon"
    assert bridged.summary == "Can't proceed."
    assert bridged.confidence == 0.1
    assert bridged.raw_text == "<raw>"
    assert bridged.metadata["k"] == "v"
    assert "_cycle_verdict" not in bridged.metadata
    assert len(bridged.findings) == 1
    assert bridged.findings[0].severity == "error"


def test_verdict_to_audit_report_uses_default_fail_label():
    v = Verdict(passed=False, reasoning="failure")
    report = audit_report_from_verdict(v, default_fail_label="iterate")
    assert report.verdict == "iterate"


def test_verdict_to_audit_report_passed_uses_success_label():
    v = Verdict(passed=True)
    report = audit_report_from_verdict(v)
    assert report.verdict == "success"


def test_finding_with_unknown_severity_coerces_to_info():
    v = Verdict(
        passed=False,
        findings=[Finding(severity="critical", message="boom")],
    )
    report = audit_report_from_verdict(v)
    assert report.findings[0].severity == "info"
    assert report.findings[0].category == "critical"


# ── cycle_auditor_from_verifier adapter ────────────────────────────────


class _RecordingVerifier:
    role_id = "recording"

    def __init__(self, verdict: Verdict) -> None:
        self._verdict = verdict
        self.last_subject: object | None = None
        self.last_ctx: VerifierContext | None = None

    async def verify(self, subject, ctx):
        self.last_subject = subject
        self.last_ctx = ctx
        return self._verdict


@pytest.mark.asyncio
async def test_adapter_threads_writer_output_and_round_to_verifier():
    inner = _RecordingVerifier(Verdict(passed=True, score=0.8))
    auditor = cycle_auditor_from_verifier(inner)
    writer_out = WriterOutput(content="hello")
    report = await auditor.verify(writer_out, round_num=2)

    assert isinstance(report, AuditReport)
    assert inner.last_subject is writer_out
    assert inner.last_ctx is not None
    assert inner.last_ctx.is_runtime is True
    assert inner.last_ctx.metadata["round_num"] == 2


@pytest.mark.asyncio
async def test_adapter_converts_passed_verdict_to_success():
    inner = _RecordingVerifier(
        Verdict(passed=True, score=0.9, reasoning="all good"),
    )
    auditor = cycle_auditor_from_verifier(inner)
    report = await auditor.verify(WriterOutput(content="x"), round_num=0)
    assert report.verdict == "success"
    assert report.confidence == 0.9
    assert report.summary == "all good"


@pytest.mark.asyncio
async def test_adapter_converts_failed_verdict_to_iterate():
    inner = _RecordingVerifier(
        Verdict(
            passed=False,
            findings=[Finding(severity="error", message="boom")],
        ),
    )
    auditor = cycle_auditor_from_verifier(inner)
    report = await auditor.verify(WriterOutput(content="x"), round_num=0)
    assert report.verdict == "iterate"
    assert len(report.findings) == 1
    assert report.findings[0].severity == "error"


@pytest.mark.asyncio
async def test_adapter_custom_ctx_factory():
    inner = _RecordingVerifier(Verdict(passed=True))

    def factory(round_num: int) -> VerifierContext:
        return VerifierContext(
            is_runtime=False,
            metadata={"round_num": round_num, "tag": "eval"},
        )

    auditor = cycle_auditor_from_verifier(inner, ctx_factory=factory)
    await auditor.verify(WriterOutput(content="x"), round_num=5)

    assert inner.last_ctx is not None
    assert inner.last_ctx.is_runtime is False
    assert inner.last_ctx.metadata["tag"] == "eval"


def test_adapter_satisfies_cycle_auditor_shape():
    inner = _RecordingVerifier(Verdict(passed=True))
    auditor = cycle_auditor_from_verifier(inner)
    assert hasattr(auditor, "role_id") and isinstance(auditor.role_id, str)
    assert inspect.iscoroutinefunction(auditor.verify)
    assert auditor.role_id == "recording"


def test_adapter_role_id_override():
    inner = _RecordingVerifier(Verdict(passed=True))
    auditor = cycle_auditor_from_verifier(inner, role_id="override")
    assert auditor.role_id == "override"
