"""Composer verifiers — each implements the Verifier protocol so they nest.

Six composition patterns:

- ``Pipeline``           short-circuits on first ``passed=False``
- ``Ensemble``           runs N verifiers in parallel and aggregates scores/passes
- ``Fallback``           primary fail → backups try in order
- ``Cascade``            cheap verifier with high confidence skips the expensive one
- ``Parallel``           runs N verifiers in parallel without aggregation,
                         all results stashed in ``sub_verdicts``
- ``ConsensusVerifier``  cross-trajectory consensus — majority vote on a
                         metadata key (e.g. extracted answer)
"""

from __future__ import annotations

import asyncio
import collections
import statistics
from typing import Any

from agent_core.components.verifier.protocols import (
    Finding,
    Verdict,
    Verifier,
    VerifierContext,
)

_AGGREGATORS = {"majority", "median", "min", "mean", "max"}


class Pipeline:
    """Run verifiers sequentially; return on first ``passed=False``."""

    def __init__(self, *verifiers: Verifier, role_id: str = "pipeline") -> None:
        if not verifiers:
            raise ValueError("Pipeline requires at least one verifier")
        self._verifiers = verifiers
        self.role_id = role_id

    async def verify(self, subject: Any, ctx: VerifierContext) -> Verdict:
        sub: list[Verdict] = []
        for v in self._verifiers:
            r = await v.verify(subject, ctx)
            sub.append(r)
            if not r.passed:
                return Verdict(
                    passed=False,
                    sub_verdicts=sub,
                    reasoning=f"short-circuited at {v.role_id}: {r.reasoning}",
                )
        return Verdict(passed=True, sub_verdicts=sub)


class Ensemble:
    """Run N verifiers in parallel and aggregate."""

    def __init__(
        self,
        *verifiers: Verifier,
        aggregator: str = "majority",
        role_id: str = "ensemble",
    ) -> None:
        if not verifiers:
            raise ValueError("Ensemble requires at least one verifier")
        if aggregator not in _AGGREGATORS:
            raise ValueError(
                f"unknown aggregator {aggregator!r}; expected one of {_AGGREGATORS}"
            )
        self._verifiers = verifiers
        self._aggregator = aggregator
        self.role_id = role_id

    async def verify(self, subject: Any, ctx: VerifierContext) -> Verdict:
        results = await asyncio.gather(
            *(v.verify(subject, ctx) for v in self._verifiers)
        )
        passes = [r.passed for r in results]
        scores = [r.score for r in results if r.score is not None]

        if self._aggregator == "majority":
            passed = passes.count(True) > len(passes) / 2
        else:
            passed = all(passes)

        score: float | None
        if not scores:
            score = None
        elif self._aggregator == "median":
            score = statistics.median(scores)
        elif self._aggregator == "min":
            score = min(scores)
        elif self._aggregator == "max":
            score = max(scores)
        elif self._aggregator == "mean":
            score = statistics.fmean(scores)
        else:
            # ``majority`` aggregates pass/fail; use median for score.
            score = statistics.median(scores)

        return Verdict(
            score=score,
            passed=passed,
            sub_verdicts=list(results),
            reasoning=f"ensemble({self._aggregator}, n={len(results)})",
        )


class Fallback:
    """Try primary; if it fails, try backups in order."""

    def __init__(
        self,
        primary: Verifier,
        *backups: Verifier,
        role_id: str = "fallback",
    ) -> None:
        self._primary = primary
        self._backups = backups
        self.role_id = role_id

    async def verify(self, subject: Any, ctx: VerifierContext) -> Verdict:
        sub: list[Verdict] = []
        first = await self._primary.verify(subject, ctx)
        sub.append(first)
        if first.passed:
            return Verdict(passed=True, sub_verdicts=sub, score=first.score)

        for backup in self._backups:
            r = await backup.verify(subject, ctx)
            sub.append(r)
            if r.passed:
                return Verdict(
                    passed=True,
                    sub_verdicts=sub,
                    score=r.score,
                    reasoning=f"primary failed; recovered via {backup.role_id}",
                )

        return Verdict(
            passed=False,
            sub_verdicts=sub,
            reasoning="primary and all backups failed",
        )


class Cascade:
    """Cheap-then-expensive: skip expensive when cheap is confident."""

    def __init__(
        self,
        cheap: Verifier,
        expensive: Verifier,
        *,
        confidence_threshold: float = 0.9,
        role_id: str = "cascade",
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        self._cheap = cheap
        self._expensive = expensive
        self._threshold = confidence_threshold
        self.role_id = role_id

    async def verify(self, subject: Any, ctx: VerifierContext) -> Verdict:
        cheap_v = await self._cheap.verify(subject, ctx)
        if cheap_v.score is not None and cheap_v.score >= self._threshold:
            return Verdict(
                passed=cheap_v.passed,
                score=cheap_v.score,
                sub_verdicts=[cheap_v],
                reasoning=f"cheap verifier confident (≥{self._threshold})",
            )
        expensive_v = await self._expensive.verify(subject, ctx)
        return Verdict(
            passed=expensive_v.passed,
            score=expensive_v.score,
            sub_verdicts=[cheap_v, expensive_v],
            reasoning=f"escalated to {self._expensive.role_id}",
        )


class Parallel:
    """Run N verifiers in parallel; collect all results without aggregating."""

    def __init__(self, *verifiers: Verifier, role_id: str = "parallel") -> None:
        if not verifiers:
            raise ValueError("Parallel requires at least one verifier")
        self._verifiers = verifiers
        self.role_id = role_id

    async def verify(self, subject: Any, ctx: VerifierContext) -> Verdict:
        results = await asyncio.gather(
            *(v.verify(subject, ctx) for v in self._verifiers)
        )
        return Verdict(
            passed=all(r.passed for r in results),
            sub_verdicts=list(results),
            reasoning=f"parallel(n={len(results)})",
        )


class ConsensusVerifier:
    """Run N verifiers in parallel and aggregate by majority answer.

    Cross-trajectory consensus primitive: each sub-verifier surfaces a
    candidate answer in ``Verdict.metadata[answer_key]``; the wrapper
    counts answers and passes when a strict majority of valid voters
    agree (``top_count * 2 > n_valid``).

    Resilient to sub-verifier exceptions — failures are excluded from
    the vote and reported via ``metadata.n_succeeded`` plus a warning
    Finding. Returns ``passed=False`` if every sub-verifier failed or
    no sub-verdict carried the answer key.

    Note: ``ConsensusVerifier`` and
    :class:`agent_core.components.cycle.builders.MajorityVoteAuditor`
    are **not equivalent** despite both being "majority over N parallel
    runs":

    - ``ConsensusVerifier`` votes on ``Verdict.metadata[answer_key]``
      (extracted candidate answer) — N sub-verifiers run on the **same**
      subject and agree on what the answer is.
    - ``MajorityVoteAuditor`` votes on ``AuditReport.verdict`` label
      (pass/fail/iterate) — N grader calls run on the **same** writer
      output and agree on the verdict.

    Different vote fields, different intents. Both legitimately coexist
    and pick the one that matches your topology. (Earlier docstrings
    suggested one collapses into the other — that was wrong; see
    ``internal-docs/designs/2026-04-30-verifier-first-class-lite.md``
    §0.1.)
    """

    def __init__(
        self,
        *verifiers: Verifier,
        answer_key: str = "answer",
        role_id: str = "consensus",
    ) -> None:
        if not verifiers:
            raise ValueError("ConsensusVerifier requires at least one verifier")
        self._verifiers = verifiers
        self._answer_key = answer_key
        self.role_id = role_id

    async def verify(self, subject: Any, ctx: VerifierContext) -> Verdict:
        n_total = len(self._verifiers)
        results = await asyncio.gather(
            *(v.verify(subject, ctx) for v in self._verifiers),
            return_exceptions=True,
        )
        valid: list[Verdict] = [r for r in results if isinstance(r, Verdict)]
        n_valid = len(valid)

        findings: list[Finding] = []
        if n_valid < n_total:
            findings.append(
                Finding(
                    severity="warning",
                    message=f"{n_total - n_valid}/{n_total} sub-verifiers failed",
                )
            )

        if not valid:
            return Verdict(
                passed=False,
                sub_verdicts=[],
                findings=findings,
                reasoning="all sub-verifiers failed",
                metadata={"n_total": n_total, "n_succeeded": 0},
            )

        answers = [v.metadata.get(self._answer_key) for v in valid]
        non_none = [a for a in answers if a is not None]

        if not non_none:
            findings.append(
                Finding(
                    severity="warning",
                    message=(
                        f"no sub-verdict carried metadata[{self._answer_key!r}]"
                    ),
                )
            )
            return Verdict(
                passed=False,
                sub_verdicts=valid,
                findings=findings,
                reasoning=(
                    f"consensus failed: no answers in "
                    f"metadata[{self._answer_key!r}]"
                ),
                metadata={
                    "n_total": n_total,
                    "n_succeeded": n_valid,
                },
            )

        counter = collections.Counter(non_none)
        consensus_answer, top_count = counter.most_common(1)[0]
        agreement = top_count / n_valid
        passed = top_count * 2 > n_valid

        return Verdict(
            passed=passed,
            sub_verdicts=valid,
            findings=findings,
            reasoning=(
                f"consensus({n_valid}/{n_total} valid, "
                f"agreement={agreement:.0%}): {consensus_answer!r}"
            ),
            metadata={
                "n_total": n_total,
                "n_succeeded": n_valid,
                "consensus_answer": consensus_answer,
                "agreement": agreement,
            },
        )
