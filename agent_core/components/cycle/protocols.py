"""Caller-side protocols for the WriteAuditCycle.

The cycle's generator extension lives in
``agent_core.components.verifier.Generator``. The verifier extension
surface is the cycle-domain ``CycleAuditor`` Protocol defined here —
distinct from the unified ``components.verifier.Verifier`` Protocol
because cycle auditors operate on ``WriterOutput`` + ``round_num``
and return ``AuditReport`` rather than the unified
``(subject, ctx) → Verdict`` shape. To plug a unified ``Verifier``
into a cycle, wrap it via
:func:`agent_core.components.verifier.cycle_auditor_from_verifier`.

The framework provides default session-backed implementations
(``SessionBackedWriter`` / ``SessionBackedAuditor`` /
``DefaultFeedbackRenderer``). Sophisticated callers can implement
``CycleAuditor`` directly — for example, an "auditor team" that fans
out to N sub-verifiers and aggregates their verdicts into one
:class:`AuditReport`.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent_core.components.cycle.types import AuditReport, WriterOutput


class CycleAuditor(Protocol):
    """Cycle-domain verifier contract used by ``WriteAuditCycle``.

    Distinct from the unified ``components.verifier.Verifier`` Protocol:
    cycle auditors receive the writer's output + round number and emit
    an ``AuditReport`` (with verdict label, structured findings, raw
    text). Use :func:`cycle_auditor_from_verifier` to bridge from a
    unified ``Verifier``.
    """

    role_id: str

    async def verify(
        self,
        writer_output: WriterOutput,
        round_num: int,
    ) -> AuditReport: ...


@runtime_checkable
class FeedbackRenderer(Protocol):
    """Turns an AuditReport into the prompt segment for the next generator round.

    The default renderer (:class:`DefaultFeedbackRenderer`) emits a
    Markdown summary table + JSON detail block + the verifier's
    free-form ``summary`` paragraph above. Callers can supply their own
    renderer when they need a different format.
    """

    def render(self, audit: AuditReport) -> str:
        """Render the audit as a feedback string.

        Implementations must not return empty strings or generic
        placeholders like ``"please address some issues"`` — when the
        audit has zero findings but a non-terminal verdict, return an
        explicit ``"no specific findings, full re-attempt required"``
        or equivalent (FR6).
        """
        ...
