"""Bridge between cycle-domain ``AuditReport`` and unified ``Verdict``.

The cycle GVR engine retains its richer ``AuditReport`` shape (verdict
label, structured findings with audit-domain fields, summary, raw text).
Verifier composers use the unified ``Verdict`` shape. These helpers
convert between the two at the boundary — workflow authors only call
them when stitching a verifier composer into a cycle engine.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast, get_args

from agent_core.components.cycle.protocols import CycleAuditor
from agent_core.components.cycle.types import Severity, WriterOutput
from agent_core.components.verifier.protocols import (
    Finding,
    Verdict,
    Verifier,
    VerifierContext,
)

if TYPE_CHECKING:
    from agent_core.components.cycle.types import AuditReport

# Must match ``cycle/write_audit_cycle.py`` ``_DEFAULT_SUCCESS`` so that a
# round-tripped Verdict whose ``passed=True`` becomes a verdict label the
# cycle treats as terminal.
_TERMINAL_OK_LABELS: frozenset[str] = frozenset({"success"})
# Non-terminal label so a failed Verdict re-enters the cycle loop instead
# of being treated as terminal-abandon.
_DEFAULT_FAIL_LABEL = "iterate"
_VALID_SEVERITIES: frozenset[str] = frozenset(get_args(Severity))


def verdict_from_audit_report(report: AuditReport) -> Verdict:
    """Convert a cycle ``AuditReport`` into a unified ``Verdict``."""
    return Verdict(
        passed=report.verdict in _TERMINAL_OK_LABELS,
        score=report.confidence,
        findings=[
            Finding(
                severity=str(f.severity),
                message=f.short_message,
                location=f.file,
            )
            for f in report.findings
        ],
        reasoning=report.summary,
        metadata={
            **report.metadata,
            "_cycle_verdict": report.verdict,
            "_cycle_raw_text": report.raw_text,
        },
    )


def audit_report_from_verdict(
    verdict: Verdict,
    *,
    default_fail_label: str = _DEFAULT_FAIL_LABEL,
) -> AuditReport:
    """Convert a ``Verdict`` back into a cycle ``AuditReport``."""
    from agent_core.components.cycle.types import AuditFinding, AuditReport

    label = verdict.metadata.get("_cycle_verdict")
    if label is None:
        label = "success" if verdict.passed else default_fail_label
    raw = verdict.metadata.get("_cycle_raw_text", "")
    leftover_metadata = {
        k: v
        for k, v in verdict.metadata.items()
        if k not in {"_cycle_verdict", "_cycle_raw_text"}
    }
    return AuditReport(
        verdict=label,
        findings=[
            AuditFinding(
                category=str(f.severity),
                severity=_coerce_severity(f.severity),
                short_message=f.message,
                file=f.location,
            )
            for f in verdict.findings
        ],
        summary=verdict.reasoning,
        confidence=verdict.score or 0.0,
        raw_text=raw,
        metadata=leftover_metadata,
    )


def _coerce_severity(value: str) -> Severity:
    if value in _VALID_SEVERITIES:
        return cast(Severity, value)
    return "info"


def _default_ctx_factory(round_num: int) -> VerifierContext:
    return VerifierContext(
        is_runtime=True,
        metadata={"round_num": round_num},
    )


class _CycleAuditorAdapter:
    """Wrap a unified ``Verifier`` to satisfy the cycle ``CycleAuditor``.

    Translates the cycle's ``(WriterOutput, round_num) → AuditReport``
    call shape into ``(subject, ctx) → Verdict`` and converts the
    returned :class:`Verdict` back via :func:`audit_report_from_verdict`.
    """

    def __init__(
        self,
        verifier: Verifier,
        ctx_factory: Callable[[int], VerifierContext],
        role_id: str,
    ) -> None:
        self._inner = verifier
        self._ctx_factory = ctx_factory
        self.role_id = role_id

    async def verify(
        self,
        writer_output: WriterOutput,
        round_num: int,
    ) -> AuditReport:
        ctx = self._ctx_factory(round_num)
        verdict = await self._inner.verify(writer_output, ctx)
        return audit_report_from_verdict(verdict)


def cycle_auditor_from_verifier(
    verifier: Verifier,
    *,
    ctx_factory: Callable[[int], VerifierContext] | None = None,
    role_id: str | None = None,
) -> CycleAuditor:
    """Adapt a unified :class:`Verifier` to the cycle ``CycleAuditor`` shape.

    The returned object structurally satisfies
    :class:`agent_core.components.cycle.protocols.CycleAuditor` and can
    be passed directly as ``WriteAuditCycle(auditor=...)``.

    ``ctx_factory`` builds the :class:`VerifierContext` for each round
    (defaults to ``is_runtime=True`` with ``round_num`` in metadata).
    """
    return _CycleAuditorAdapter(
        verifier,
        ctx_factory or _default_ctx_factory,
        role_id or getattr(verifier, "role_id", "verifier"),
    )
