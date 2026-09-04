"""Tests for cycle protocols + verifier protocol contract within cycle scope."""
from __future__ import annotations

from agent_core.components.cycle.protocols import FeedbackRenderer
from agent_core.components.cycle.types import AuditReport, WriterOutput
from agent_core.components.verifier import Generator, Verifier


class _FakeGenerator:
    role_id = "fake_writer"

    async def generate(
        self,
        prev_audit: AuditReport | None,
        round_num: int,
        feedback_md: str,
    ) -> WriterOutput:
        return WriterOutput(content=f"round {round_num}")


class _FakeVerifier:
    role_id = "fake_auditor"

    async def verify(
        self,
        writer_output: WriterOutput,
        round_num: int,
    ) -> AuditReport:
        return AuditReport(verdict="success")


class _FakeRenderer:
    def render(self, audit: AuditReport) -> str:
        return f"verdict={audit.verdict}"


class _MissingMethod:
    role_id = "broken"


def test_generator_protocol_passes_isinstance() -> None:
    assert isinstance(_FakeGenerator(), Generator)


def test_verifier_protocol_passes_isinstance() -> None:
    assert isinstance(_FakeVerifier(), Verifier)


def test_feedback_renderer_protocol_passes_isinstance() -> None:
    assert isinstance(_FakeRenderer(), FeedbackRenderer)


def test_missing_method_fails_isinstance() -> None:
    """runtime_checkable protocols verify method presence."""
    assert not isinstance(_MissingMethod(), Generator)
    assert not isinstance(_MissingMethod(), Verifier)


def test_renderer_does_not_require_role_id() -> None:
    """FeedbackRenderer is method-only — no role_id attribute."""

    class _PureRenderer:
        def render(self, audit: AuditReport) -> str:
            return "x"

    assert isinstance(_PureRenderer(), FeedbackRenderer)
