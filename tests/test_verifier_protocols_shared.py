"""Verifier protocol contracts + ground-truth oracle isolation."""

from __future__ import annotations

import pytest

from agent_core.components.verifier import (
    Finding,
    GroundTruth,
    Verdict,
    Verifier,
    VerifierContext,
)


def test_runtime_strips_oracle_fields():
    gt = GroundTruth(
        rubric="grade as IMO proof",
        _reference="the official answer",
        _formal_spec="lean code",
        _test_cases=[1, 2, 3],
    )
    ctx = VerifierContext(is_runtime=True, _ground_truth=gt)
    runtime_view = ctx.ground_truth
    assert runtime_view is not None
    assert runtime_view.rubric == "grade as IMO proof"
    assert runtime_view._reference is None
    assert runtime_view._formal_spec is None
    assert runtime_view._test_cases is None


def test_eval_keeps_oracle_fields():
    gt = GroundTruth(
        rubric="grade as IMO proof",
        _reference="the official answer",
        _formal_spec="lean code",
    )
    ctx = VerifierContext(is_runtime=False, _ground_truth=gt)
    eval_view = ctx.ground_truth
    assert eval_view is not None
    assert eval_view._reference == "the official answer"
    assert eval_view._formal_spec == "lean code"


def test_no_ground_truth_returns_none():
    ctx = VerifierContext(is_runtime=True, _ground_truth=None)
    assert ctx.ground_truth is None


def test_runtime_metadata_isolation():
    gt = GroundTruth(rubric="r", metadata={"k": "v"})
    ctx = VerifierContext(is_runtime=True, _ground_truth=gt)
    runtime_view = ctx.ground_truth
    runtime_view.metadata["mutate"] = "after"
    assert "mutate" not in gt.metadata


def test_verdict_defaults():
    v = Verdict()
    assert v.score is None
    assert v.passed is False
    assert v.findings == []
    assert v.sub_verdicts == []
    assert v.metadata == {}


def test_finding_required_fields():
    f = Finding(severity="error", message="boom")
    assert f.severity == "error"
    assert f.message == "boom"
    assert f.location is None


class _StubVerifier:
    role_id = "stub"

    async def verify(self, subject, ctx):
        return Verdict(passed=True)


@pytest.mark.asyncio
async def test_verifier_protocol_runtime_check():
    v = _StubVerifier()
    assert isinstance(v, Verifier)
    out = await v.verify("x", VerifierContext(is_runtime=True))
    assert out.passed is True
