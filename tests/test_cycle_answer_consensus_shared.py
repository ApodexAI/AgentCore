"""TDD-first tests for ``select_by_answer_consensus``.

Mirrors MiroVerifier@fac1b9e:agents/imo_gvr.py:_select answer mode:
attempts grouped by answer-equivalence buckets; pick the bucket with
highest summed score (tie → larger bucket); within winning bucket,
``select_best_attempt`` rule.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest

from agent_core.components.cycle import AuditReport, WriterOutput
from agent_core.components.cycle.builders import select_by_answer_consensus


def _attempt(
    *,
    content: str = "writer output",
    score: float | None,
    extracted: str,
    extra: dict | None = None,
) -> tuple[WriterOutput, AuditReport]:
    metadata = {"score": score, "extracted_answer": extracted}
    if extra:
        metadata.update(extra)
    audit = AuditReport(
        verdict="iterate",
        findings=[],
        summary="",
        confidence=1.0,
        metadata=metadata,
    )
    return (WriterOutput(content=content), audit)


# --- equivalence helpers used in tests ---------------------------------

async def _exact(a: str, b: str) -> bool:
    return a.strip() == b.strip()


def _make_equivalence_set(*groups: list[str]) -> Callable[[str, str], Awaitable[bool]]:
    """Return an equivalence function that maps each input to its group."""
    bucket_of: dict[str, int] = {}
    for i, group in enumerate(groups):
        for s in group:
            bucket_of[s] = i

    async def eq(a: str, b: str) -> bool:
        return bucket_of.get(a, -1) == bucket_of.get(b, -2)

    return eq


# --- tests --------------------------------------------------------------


class TestSimpleCases:
    @pytest.mark.asyncio
    async def test_empty_history_raises(self) -> None:
        with pytest.raises(ValueError):
            await select_by_answer_consensus([], equivalence=_exact)

    @pytest.mark.asyncio
    async def test_single_attempt_returns_it(self) -> None:
        history = [_attempt(score=5, extracted="42")]
        idx, _w, a = await select_by_answer_consensus(history, equivalence=_exact)
        assert idx == 0
        assert a.metadata["extracted_answer"] == "42"


class TestBucketing:
    @pytest.mark.asyncio
    async def test_same_answer_bucketed_together(self) -> None:
        history = [
            _attempt(score=3, extracted="42"),
            _attempt(score=5, extracted="42"),
        ]
        idx, _w, a = await select_by_answer_consensus(history, equivalence=_exact)
        # Both in same bucket, best-by-score → idx 1
        assert idx == 1
        assert a.metadata["score"] == 5

    @pytest.mark.asyncio
    async def test_different_answers_separate_buckets_higher_summed_score_wins(
        self,
    ) -> None:
        history = [
            _attempt(score=7, extracted="x+1"),
            _attempt(score=3, extracted="y-1"),
            _attempt(score=4, extracted="y-1"),
        ]
        # bucket A "x+1": sum=7
        # bucket B "y-1": sum=7
        # tie on sum → larger bucket wins → B (size=2)
        idx, _w, a = await select_by_answer_consensus(history, equivalence=_exact)
        assert a.metadata["extracted_answer"] == "y-1"
        # within bucket B: best-by-score → idx 2 (score 4)
        assert idx == 2

    @pytest.mark.asyncio
    async def test_higher_summed_score_wins_over_larger_bucket(self) -> None:
        history = [
            _attempt(score=7, extracted="x+1"),
            _attempt(score=3, extracted="y-1"),
            _attempt(score=3, extracted="y-1"),
        ]
        # bucket A: sum=7, size=1
        # bucket B: sum=6, size=2
        # higher summed score wins → A
        idx, _w, a = await select_by_answer_consensus(history, equivalence=_exact)
        assert idx == 0
        assert a.metadata["extracted_answer"] == "x+1"

    @pytest.mark.asyncio
    async def test_equivalent_answers_via_callable_bucket_together(self) -> None:
        # "1/2" and "\frac{1}{2}" treated as equivalent by custom callable
        eq = _make_equivalence_set(["1/2", "\\frac{1}{2}"], ["x+1"])
        history = [
            _attempt(score=5, extracted="1/2"),
            _attempt(score=4, extracted="\\frac{1}{2}"),
            _attempt(score=7, extracted="x+1"),
        ]
        # bucket A "1/2 ≡ \frac{1}{2}": sum=9, size=2
        # bucket B "x+1": sum=7, size=1
        # higher sum → A
        idx, _w, a = await select_by_answer_consensus(history, equivalence=eq)
        assert a.metadata["extracted_answer"] == "1/2"  # best-by-score in bucket A
        assert idx == 0


class TestFallback:
    @pytest.mark.asyncio
    async def test_no_extracted_answers_falls_back_to_best_by_score(self) -> None:
        history = [
            _attempt(score=3, extracted=""),
            _attempt(score=7, extracted=""),
            _attempt(score=5, extracted=""),
        ]
        idx, _w, a = await select_by_answer_consensus(history, equivalence=_exact)
        assert idx == 1
        assert a.metadata["score"] == 7

    @pytest.mark.asyncio
    async def test_some_invalid_some_valid_only_valid_bucketed(self) -> None:
        history = [
            _attempt(score=7, extracted=""),
            _attempt(score=3, extracted="x+1"),
        ]
        # only valid attempt is idx=1 → bucket {1}, returns it
        idx, _w, _a = await select_by_answer_consensus(history, equivalence=_exact)
        assert idx == 1


class TestEquivalenceExceptions:
    @pytest.mark.asyncio
    async def test_equivalence_raising_treated_as_not_equivalent(self) -> None:
        async def flaky(a: str, b: str) -> bool:
            raise RuntimeError("boom")

        history = [
            _attempt(score=5, extracted="x"),
            _attempt(score=3, extracted="y"),
        ]
        # equivalence always raises → every attempt becomes its own
        # bucket → highest score bucket wins (idx 0)
        idx, _w, _a = await select_by_answer_consensus(history, equivalence=flaky)
        assert idx == 0


class TestTieBreaking:
    @pytest.mark.asyncio
    async def test_within_bucket_ties_go_to_latest(self) -> None:
        history = [
            _attempt(score=5, extracted="42"),
            _attempt(score=5, extracted="42"),
        ]
        idx, _w, _a = await select_by_answer_consensus(history, equivalence=_exact)
        # Same bucket, same score → latest wins (parity with select_best_attempt)
        assert idx == 1
