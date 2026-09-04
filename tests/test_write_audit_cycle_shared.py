"""Tests for WriteAuditCycle — six §7.1 cases against fake Writer/Auditor."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_core.components.cycle.types import (
    AuditFinding,
    AuditReport,
    WriterOutput,
)
from agent_core.components.cycle.write_audit_cycle import WriteAuditCycle


class _ScriptedWriter:
    """Returns scripted WriterOutputs in order. Re-uses last on overflow."""

    role_id = "scripted_writer"

    def __init__(self, outputs: list[WriterOutput]) -> None:
        self.outputs = outputs
        self.calls: list[tuple[AuditReport | None, int, str]] = []

    async def generate(
        self,
        prev_audit: AuditReport | None,
        round_num: int,
        feedback_md: str,
    ) -> WriterOutput:
        self.calls.append((prev_audit, round_num, feedback_md))
        idx = min(round_num, len(self.outputs) - 1)
        return self.outputs[idx]


class _RaisingWriter:
    role_id = "raising_writer"

    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, *args, **kwargs):
        self.calls += 1
        raise RuntimeError("kaboom")


class _ScriptedAuditor:
    """Returns scripted AuditReports in order. Re-uses last on overflow."""

    role_id = "scripted_auditor"

    def __init__(self, reports: list[AuditReport]) -> None:
        self.reports = reports
        self.calls: list[tuple[WriterOutput, int]] = []

    async def verify(
        self, writer_output: WriterOutput, round_num: int,
    ) -> AuditReport:
        self.calls.append((writer_output, round_num))
        idx = min(round_num, len(self.reports) - 1)
        return self.reports[idx]


def _no_missing(_wd: Path) -> list[str]:
    return []


def _always_missing(missing: list[str]):
    def _check(_wd: Path) -> list[str]:
        return list(missing)
    return _check


# ── Case 1: terminal-success on round 0 ─────────────────────────────────────


@pytest.mark.asyncio
async def test_success_on_round_0(tmp_path: Path) -> None:
    writer = _ScriptedWriter([WriterOutput(content="draft v1")])
    auditor = _ScriptedAuditor([
        AuditReport(verdict="success", summary="LGTM"),
    ])
    cycle = WriteAuditCycle(
        writer=writer, auditor=auditor,
        work_dir=tmp_path, output_check=_no_missing,
    )
    out = await cycle.run()
    assert out.success is True
    assert out.rounds_used == 1
    assert out.reason == "verdict=success"
    assert len(out.history) == 1
    assert len(writer.calls) == 1
    # Round 0 has no prior feedback
    assert writer.calls[0] == (None, 0, "")
    assert (tmp_path / "audit_round_0.json").exists()


# ── Case 2: iterate then success on a later round ───────────────────────────


@pytest.mark.asyncio
async def test_iterate_then_success(tmp_path: Path) -> None:
    writer = _ScriptedWriter([
        WriterOutput(content="v1"),
        WriterOutput(content="v2"),
    ])
    auditor = _ScriptedAuditor([
        AuditReport(
            verdict="iterate",
            findings=[
                AuditFinding(
                    category="missing_section",
                    severity="error",
                    short_message="add intro",
                ),
            ],
            summary="Need intro.",
        ),
        AuditReport(verdict="success", summary="LGTM"),
    ])
    cycle = WriteAuditCycle(
        writer=writer, auditor=auditor,
        work_dir=tmp_path, output_check=_no_missing,
    )
    out = await cycle.run()
    assert out.success is True
    assert out.rounds_used == 2
    assert len(out.history) == 2
    # Round 1's feedback_md must be non-empty (contains the iterate audit's findings)
    _, _, fb = writer.calls[1]
    assert "add intro" in fb
    assert "missing_section" in fb
    # Both round files persisted
    assert (tmp_path / "audit_round_0.json").exists()
    assert (tmp_path / "audit_round_1.json").exists()


# ── Case 3: max_rounds exhausted ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_max_rounds_exhausted(tmp_path: Path) -> None:
    writer = _ScriptedWriter([WriterOutput(content="x")])
    auditor = _ScriptedAuditor([AuditReport(verdict="iterate")])
    cycle = WriteAuditCycle(
        writer=writer, auditor=auditor,
        work_dir=tmp_path, output_check=_no_missing,
        max_rounds=3,
    )
    out = await cycle.run()
    assert out.success is False
    assert out.rounds_used == 3
    assert out.reason == "max_rounds_exhausted"
    assert len(out.history) == 3
    assert writer.calls[0][2] == ""           # round 0 has no feedback
    assert writer.calls[1][2] != ""           # round 1 sees prior audit
    assert writer.calls[2][2] != ""


# ── Case 4: output_check failure short-circuits auditor ─────────────────────


@pytest.mark.asyncio
async def test_output_check_failure_short_circuits_auditor(
    tmp_path: Path,
) -> None:
    writer = _ScriptedWriter([WriterOutput(content="empty draft")])
    auditor = _ScriptedAuditor([
        AuditReport(verdict="success", summary="should not be called"),
    ])
    cycle = WriteAuditCycle(
        writer=writer, auditor=auditor,
        work_dir=tmp_path,
        output_check=_always_missing(["sections/intro.md", "data/refs.bib"]),
        max_rounds=2,
    )
    out = await cycle.run()
    # Auditor was never invoked
    assert auditor.calls == []
    # Synthetic audit names every missing artifact
    assert out.final_audit is not None
    cats = {f.category for f in out.final_audit.findings}
    assert cats == {"output_missing"}
    files = {f.file for f in out.final_audit.findings}
    assert files == {"sections/intro.md", "data/refs.bib"}
    # max_rounds_exhausted because the writer keeps failing output_check
    assert out.reason == "max_rounds_exhausted"
    # Persistence still fires for synthetic audits
    persisted = json.loads((tmp_path / "audit_round_0.json").read_text())
    assert persisted["metadata"]["synthetic"] is True
    assert persisted["metadata"]["reason"] == "output_missing"


# ── Case 5: writer exception captured as writer_exception finding ───────────


@pytest.mark.asyncio
async def test_writer_exception_captured_and_continues(
    tmp_path: Path,
) -> None:
    writer = _RaisingWriter()
    auditor = _ScriptedAuditor([AuditReport(verdict="success")])
    cycle = WriteAuditCycle(
        writer=writer, auditor=auditor,
        work_dir=tmp_path, output_check=_no_missing,
        max_rounds=2,
    )
    out = await cycle.run()
    # The cycle did NOT raise
    assert writer.calls == 2
    # Auditor never invoked because output_check / writer never produced
    assert auditor.calls == []
    # Both rounds have a synthetic writer_exception audit
    assert out.success is False
    assert out.reason == "max_rounds_exhausted"
    for _, audit in out.history:
        assert audit.metadata.get("reason") == "writer_exception"
        assert any(
            f.category == "writer_exception" for f in audit.findings
        )
        # Exception type is surfaced in the short message
        assert any("RuntimeError" in f.short_message for f in audit.findings)
    # Persistence happened for both rounds
    assert (tmp_path / "audit_round_0.json").exists()
    assert (tmp_path / "audit_round_1.json").exists()


# ── Case 6: persistence round-trips to_json / from_json ─────────────────────


@pytest.mark.asyncio
async def test_persisted_audit_round_trips(tmp_path: Path) -> None:
    writer = _ScriptedWriter([WriterOutput(content="v1")])
    auditor = _ScriptedAuditor([
        AuditReport(
            verdict="success",
            findings=[
                AuditFinding(
                    category="info_only",
                    severity="info",
                    short_message="all good",
                    metadata={"score": 0.95},
                ),
            ],
            summary="Looks great.",
            confidence=0.95,
            metadata={"reviewer_id": "auditor_R0"},
        ),
    ])
    cycle = WriteAuditCycle(
        writer=writer, auditor=auditor,
        work_dir=tmp_path, output_check=_no_missing,
    )
    await cycle.run()
    text = (tmp_path / "audit_round_0.json").read_text()
    restored = AuditReport.from_json(text)
    assert restored.verdict == "success"
    assert restored.findings[0].category == "info_only"
    assert restored.findings[0].metadata == {"score": 0.95}
    assert restored.summary == "Looks great."
    assert restored.confidence == 0.95
    assert restored.metadata == {"reviewer_id": "auditor_R0"}


# ── Extra: invariants ───────────────────────────────────────────────────────


def test_max_rounds_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_rounds"):
        WriteAuditCycle(
            writer=_ScriptedWriter([]),
            auditor=_ScriptedAuditor([]),
            work_dir=Path("/tmp"),
            output_check=_no_missing,
            max_rounds=0,
        )


def test_success_must_subset_terminal() -> None:
    with pytest.raises(ValueError, match="success_verdicts"):
        WriteAuditCycle(
            writer=_ScriptedWriter([]),
            auditor=_ScriptedAuditor([]),
            work_dir=Path("/tmp"),
            output_check=_no_missing,
            terminal_verdicts={"abandon"},
            success_verdicts={"success"},  # not in terminal
        )


@pytest.mark.asyncio
async def test_abandon_terminates_with_failure(tmp_path: Path) -> None:
    writer = _ScriptedWriter([WriterOutput(content="x")])
    auditor = _ScriptedAuditor([AuditReport(verdict="abandon")])
    cycle = WriteAuditCycle(
        writer=writer, auditor=auditor,
        work_dir=tmp_path, output_check=_no_missing,
    )
    out = await cycle.run()
    assert out.success is False
    assert out.reason == "verdict=abandon"
    assert out.rounds_used == 1


@pytest.mark.asyncio
async def test_custom_terminal_verdicts(tmp_path: Path) -> None:
    """Caller can use domain-specific verdict vocabulary."""
    writer = _ScriptedWriter([WriterOutput(content="x")])
    auditor = _ScriptedAuditor([AuditReport(verdict="merge")])
    cycle = WriteAuditCycle(
        writer=writer, auditor=auditor,
        work_dir=tmp_path, output_check=_no_missing,
        terminal_verdicts={"merge", "discard"},
        success_verdicts={"merge"},
    )
    out = await cycle.run()
    assert out.success is True
    assert out.reason == "verdict=merge"


# ── Day 5: cycle accepts unified Verifier via adapter ───────────────────────


@pytest.mark.asyncio
async def test_cycle_runs_with_adapter_wrapped_verifier(tmp_path: Path) -> None:
    """End-to-end: components.verifier.Verifier → cycle via adapter."""
    from agent_core.components.verifier import (
        Verdict,
        cycle_auditor_from_verifier,
    )

    class _UnifiedVerifier:
        role_id = "unified"

        def __init__(self, verdicts: list[Verdict]) -> None:
            self.verdicts = verdicts
            self.calls: list[int] = []

        async def verify(self, subject, ctx) -> Verdict:
            self.calls.append(ctx.metadata["round_num"])
            idx = min(len(self.calls) - 1, len(self.verdicts) - 1)
            return self.verdicts[idx]

    writer = _ScriptedWriter([
        WriterOutput(content="draft v1"),
        WriterOutput(content="draft v2"),
    ])
    inner = _UnifiedVerifier([
        Verdict(passed=False, score=0.3, reasoning="needs work"),
        Verdict(passed=True, score=0.95, reasoning="ship it"),
    ])
    auditor = cycle_auditor_from_verifier(inner)
    cycle = WriteAuditCycle(
        writer=writer, auditor=auditor,
        work_dir=tmp_path, output_check=_no_missing,
        max_rounds=3,
    )
    out = await cycle.run()

    assert out.success is True
    assert out.reason == "verdict=success"
    assert out.rounds_used == 2
    assert inner.calls == [0, 1]
    assert out.final_audit is not None
    assert out.final_audit.confidence == 0.95
    assert out.final_audit.summary == "ship it"
