"""Tests for DefaultFeedbackRenderer."""
from __future__ import annotations

import json
import re

from agent_core.components.cycle.default_renderer import DefaultFeedbackRenderer
from agent_core.components.cycle.types import AuditFinding, AuditReport


def test_multi_finding_emits_table_and_json_block() -> None:
    audit = AuditReport(
        verdict="iterate",
        findings=[
            AuditFinding(
                category="missing_section",
                severity="error",
                short_message="add intro",
                suggested_action="write 200-word introduction",
            ),
            AuditFinding(
                category="citation_invalid",
                severity="warning",
                short_message="cite missing",
                file="paper.md", line=42,
                suggested_action="add citation",
            ),
        ],
        summary="Two issues blocking acceptance.",
    )
    out = DefaultFeedbackRenderer().render(audit)
    # Table header is present
    assert "| # | Severity | Category" in out
    # Both findings show up as rows
    assert "missing_section" in out
    assert "citation_invalid" in out
    assert "paper.md:42" in out
    # JSON block is present and valid
    m = re.search(r"```json\n(.+?)\n```", out, re.DOTALL)
    assert m, "expected fenced json block"
    parsed = json.loads(m.group(1))
    assert isinstance(parsed, list)
    assert len(parsed) == 2
    assert parsed[0]["category"] == "missing_section"
    # Free-form summary is rendered verbatim
    assert "Two issues blocking acceptance." in out


def test_zero_findings_non_terminal_emits_reattempt_message() -> None:
    audit = AuditReport(verdict="iterate", findings=[], summary="")
    out = DefaultFeedbackRenderer().render(audit)
    assert "re-attempt" in out.lower() or "re-attempt" in out
    assert "iterate" in out
    # No empty string — the whole point of FR6
    assert out.strip() != ""
    # Must NOT include a generic placeholder
    assert "please address some issues" not in out.lower()


def test_zero_findings_success_verdict_skips_reattempt() -> None:
    """Success verdict + no findings is a clean pass — no re-attempt nudge."""
    audit = AuditReport(verdict="success", findings=[], summary="LGTM.")
    out = DefaultFeedbackRenderer().render(audit)
    assert "re-attempt" not in out.lower()
    assert "LGTM." in out


def test_single_finding_table_well_formed() -> None:
    audit = AuditReport(
        verdict="iterate",
        findings=[
            AuditFinding(
                category="x", severity="error", short_message="m",
            ),
        ],
    )
    out = DefaultFeedbackRenderer().render(audit)
    lines = out.splitlines()
    # Header + alignment separator + one data row, in order
    header_idx = next(
        (i for i, line in enumerate(lines) if line.startswith("| # |")),
        -1,
    )
    assert header_idx >= 0, f"missing header in:\n{out}"
    assert lines[header_idx + 1].startswith("|---")
    assert lines[header_idx + 2].startswith("| 1 |")
    assert "| error |" in lines[header_idx + 2]
    assert "| x |" in lines[header_idx + 2]


def test_pipe_in_message_escaped() -> None:
    audit = AuditReport(
        verdict="iterate",
        findings=[
            AuditFinding(
                category="x", severity="error",
                short_message="contains | pipe",
            ),
        ],
    )
    out = DefaultFeedbackRenderer().render(audit)
    # Table row must escape the pipe so the row stays well-formed
    assert "contains \\| pipe" in out


def test_custom_success_verdicts() -> None:
    """Caller can override what counts as a 'success' verdict."""
    audit = AuditReport(verdict="merge", findings=[], summary="")
    out = DefaultFeedbackRenderer(
        success_verdicts={"merge", "ship"},
    ).render(audit)
    # 'merge' is success here → no re-attempt message
    assert "re-attempt" not in out.lower()


def test_summary_only_no_findings_terminal() -> None:
    audit = AuditReport(
        verdict="success", findings=[],
        summary="Looks great.", confidence=0.92,
    )
    out = DefaultFeedbackRenderer().render(audit)
    assert "Looks great." in out
    # Confidence surfaced when non-zero
    assert "0.92" in out
