"""Composer semantics + nesting."""

from __future__ import annotations

import pytest

from agent_core.components.verifier import (
    Cascade,
    ConsensusVerifier,
    Ensemble,
    Fallback,
    Parallel,
    Pipeline,
    Verdict,
    VerifierContext,
)


class _Stub:
    """Returns a fixed verdict; records call count."""

    def __init__(
        self,
        *,
        passed: bool = True,
        score: float | None = None,
        role_id: str = "stub",
    ) -> None:
        self._passed = passed
        self._score = score
        self.role_id = role_id
        self.calls = 0

    async def verify(self, subject, ctx):
        self.calls += 1
        return Verdict(
            passed=self._passed,
            score=self._score,
            reasoning=f"{self.role_id} returned passed={self._passed}",
        )


@pytest.fixture
def ctx():
    return VerifierContext(is_runtime=True)


# ── Pipeline ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pipeline_all_pass(ctx):
    a, b, c = _Stub(passed=True), _Stub(passed=True), _Stub(passed=True)
    p = Pipeline(a, b, c)
    out = await p.verify("x", ctx)
    assert out.passed is True
    assert len(out.sub_verdicts) == 3
    assert a.calls == b.calls == c.calls == 1


@pytest.mark.asyncio
async def test_pipeline_short_circuits_on_first_fail(ctx):
    a = _Stub(passed=True, role_id="a")
    b = _Stub(passed=False, role_id="b")
    c = _Stub(passed=True, role_id="c")
    p = Pipeline(a, b, c)
    out = await p.verify("x", ctx)
    assert out.passed is False
    assert len(out.sub_verdicts) == 2
    assert c.calls == 0
    assert "short-circuited at b" in out.reasoning


def test_pipeline_rejects_empty():
    with pytest.raises(ValueError):
        Pipeline()


# ── Ensemble ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ensemble_majority_pass(ctx):
    e = Ensemble(
        _Stub(passed=True),
        _Stub(passed=True),
        _Stub(passed=False),
        aggregator="majority",
    )
    out = await e.verify("x", ctx)
    assert out.passed is True
    assert len(out.sub_verdicts) == 3


@pytest.mark.asyncio
async def test_ensemble_majority_fail(ctx):
    e = Ensemble(
        _Stub(passed=False),
        _Stub(passed=False),
        _Stub(passed=True),
        aggregator="majority",
    )
    out = await e.verify("x", ctx)
    assert out.passed is False


@pytest.mark.asyncio
async def test_ensemble_score_aggregation(ctx):
    e = Ensemble(
        _Stub(passed=True, score=0.6),
        _Stub(passed=True, score=0.8),
        _Stub(passed=True, score=1.0),
        aggregator="median",
    )
    out = await e.verify("x", ctx)
    assert out.score == 0.8


@pytest.mark.asyncio
async def test_ensemble_score_min(ctx):
    e = Ensemble(
        _Stub(passed=True, score=0.5),
        _Stub(passed=True, score=0.9),
        aggregator="min",
    )
    out = await e.verify("x", ctx)
    assert out.score == 0.5


def test_ensemble_rejects_unknown_aggregator():
    with pytest.raises(ValueError):
        Ensemble(_Stub(), aggregator="totally-bogus")


# ── Fallback ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fallback_primary_passes(ctx):
    primary = _Stub(passed=True, role_id="primary")
    backup = _Stub(passed=True, role_id="backup")
    f = Fallback(primary, backup)
    out = await f.verify("x", ctx)
    assert out.passed is True
    assert backup.calls == 0


@pytest.mark.asyncio
async def test_fallback_primary_fails_backup_recovers(ctx):
    primary = _Stub(passed=False, role_id="p")
    backup_a = _Stub(passed=False, role_id="ba")
    backup_b = _Stub(passed=True, role_id="bb")
    f = Fallback(primary, backup_a, backup_b)
    out = await f.verify("x", ctx)
    assert out.passed is True
    assert "bb" in out.reasoning
    assert primary.calls == backup_a.calls == backup_b.calls == 1


@pytest.mark.asyncio
async def test_fallback_all_fail(ctx):
    f = Fallback(
        _Stub(passed=False),
        _Stub(passed=False),
        _Stub(passed=False),
    )
    out = await f.verify("x", ctx)
    assert out.passed is False


# ── Cascade ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_cascade_cheap_confident_skips_expensive(ctx):
    cheap = _Stub(passed=True, score=0.95, role_id="cheap")
    expensive = _Stub(passed=True, score=0.99, role_id="exp")
    c = Cascade(cheap, expensive, confidence_threshold=0.9)
    out = await c.verify("x", ctx)
    assert out.passed is True
    assert expensive.calls == 0
    assert "confident" in out.reasoning


@pytest.mark.asyncio
async def test_cascade_low_confidence_escalates(ctx):
    cheap = _Stub(passed=True, score=0.5, role_id="cheap")
    expensive = _Stub(passed=False, score=0.4, role_id="exp")
    c = Cascade(cheap, expensive, confidence_threshold=0.9)
    out = await c.verify("x", ctx)
    assert out.passed is False
    assert expensive.calls == 1
    assert len(out.sub_verdicts) == 2


@pytest.mark.asyncio
async def test_cascade_no_cheap_score_escalates(ctx):
    cheap = _Stub(passed=True, score=None, role_id="cheap")
    expensive = _Stub(passed=True, score=0.7, role_id="exp")
    c = Cascade(cheap, expensive, confidence_threshold=0.9)
    await c.verify("x", ctx)
    assert expensive.calls == 1


def test_cascade_threshold_validation():
    with pytest.raises(ValueError):
        Cascade(_Stub(), _Stub(), confidence_threshold=1.5)


# ── Parallel ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_parallel_collects_all(ctx):
    stubs = [_Stub(passed=True, role_id=f"s{i}") for i in range(5)]
    p = Parallel(*stubs)
    out = await p.verify("x", ctx)
    assert out.passed is True
    assert len(out.sub_verdicts) == 5


@pytest.mark.asyncio
async def test_parallel_fails_if_any_fails(ctx):
    p = Parallel(
        _Stub(passed=True),
        _Stub(passed=True),
        _Stub(passed=False),
    )
    out = await p.verify("x", ctx)
    assert out.passed is False
    assert len(out.sub_verdicts) == 3


# ── Nesting ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pipeline_of_ensemble(ctx):
    """Pipeline can short-circuit on a composer's verdict."""
    inner_ok = Ensemble(_Stub(passed=True), _Stub(passed=True))
    inner_fail = Ensemble(_Stub(passed=False), _Stub(passed=False))
    final = _Stub(passed=True)
    p = Pipeline(inner_ok, inner_fail, final)
    out = await p.verify("x", ctx)
    assert out.passed is False
    assert len(out.sub_verdicts) == 2
    assert final.calls == 0


@pytest.mark.asyncio
async def test_fallback_of_parallel_recovers(ctx):
    failing = Parallel(_Stub(passed=False), _Stub(passed=True))
    succeeding = Parallel(_Stub(passed=True), _Stub(passed=True))
    f = Fallback(failing, succeeding)
    out = await f.verify("x", ctx)
    assert out.passed is True
    assert len(out.sub_verdicts) == 2


# ── ConsensusVerifier ──────────────────────────────────────────────────


class _AnswerStub:
    """Verifier that surfaces a fixed answer in metadata."""

    def __init__(
        self,
        *,
        answer: str | None,
        passed: bool = True,
        role_id: str = "answer-stub",
    ) -> None:
        self._answer = answer
        self._passed = passed
        self.role_id = role_id

    async def verify(self, subject, ctx):
        meta: dict[str, str] = {}
        if self._answer is not None:
            meta["answer"] = self._answer
        return Verdict(passed=self._passed, metadata=meta)


class _RaisingStub:
    role_id = "raising"

    async def verify(self, subject, ctx):
        raise RuntimeError("simulated grader failure")


@pytest.mark.asyncio
async def test_consensus_strict_majority(ctx):
    c = ConsensusVerifier(
        _AnswerStub(answer="42"),
        _AnswerStub(answer="42"),
        _AnswerStub(answer="7"),
    )
    out = await c.verify("subject", ctx)
    assert out.passed is True
    assert out.metadata["consensus_answer"] == "42"
    assert out.metadata["n_succeeded"] == 3
    assert out.metadata["agreement"] == pytest.approx(2 / 3)


@pytest.mark.asyncio
async def test_consensus_no_majority_two_two_split(ctx):
    c = ConsensusVerifier(
        _AnswerStub(answer="a"),
        _AnswerStub(answer="a"),
        _AnswerStub(answer="b"),
        _AnswerStub(answer="b"),
    )
    out = await c.verify("subject", ctx)
    assert out.passed is False  # 2-2 split is not strict majority
    # Counter.most_common breaks ties first-seen
    assert out.metadata["consensus_answer"] == "a"
    assert out.metadata["agreement"] == 0.5


@pytest.mark.asyncio
async def test_consensus_resilient_to_single_failure(ctx):
    c = ConsensusVerifier(
        _AnswerStub(answer="42"),
        _AnswerStub(answer="42"),
        _RaisingStub(),
    )
    out = await c.verify("subject", ctx)
    assert out.passed is True  # 2/2 valid agree → strict majority
    assert out.metadata["n_total"] == 3
    assert out.metadata["n_succeeded"] == 2
    assert out.metadata["consensus_answer"] == "42"
    assert any(
        f.severity == "warning" and "sub-verifiers failed" in f.message
        for f in out.findings
    )


@pytest.mark.asyncio
async def test_consensus_all_failed(ctx):
    c = ConsensusVerifier(_RaisingStub(), _RaisingStub())
    out = await c.verify("subject", ctx)
    assert out.passed is False
    assert out.metadata["n_succeeded"] == 0
    assert out.sub_verdicts == []


@pytest.mark.asyncio
async def test_consensus_missing_answer_key(ctx):
    c = ConsensusVerifier(
        _AnswerStub(answer=None, passed=True),
        _AnswerStub(answer=None, passed=True),
    )
    out = await c.verify("subject", ctx)
    assert out.passed is False
    assert "consensus_answer" not in out.metadata
    assert any(
        "no sub-verdict carried metadata" in f.message for f in out.findings
    )


@pytest.mark.asyncio
async def test_consensus_custom_answer_key(ctx):
    class _ExtractedAnswerStub:
        role_id = "extracted"

        def __init__(self, value: str) -> None:
            self._v = value

        async def verify(self, subject, ctx):
            return Verdict(
                passed=True, metadata={"extracted_answer": self._v}
            )

    c = ConsensusVerifier(
        _ExtractedAnswerStub("π/4"),
        _ExtractedAnswerStub("π/4"),
        _ExtractedAnswerStub("0"),
        answer_key="extracted_answer",
    )
    out = await c.verify("subject", ctx)
    assert out.passed is True
    assert out.metadata["consensus_answer"] == "π/4"


def test_consensus_rejects_empty():
    with pytest.raises(ValueError):
        ConsensusVerifier()
