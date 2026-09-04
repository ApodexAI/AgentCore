"""Built-in :class:`RoundObserver` implementations.

These are the most common cycle-extension patterns, shipped so callers
don't have to re-derive them. All are pure additions on top of
:class:`agent_core.components.cycle.observers.BaseRoundObserver` — feel free to
read them as templates for your own observers.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from agent_core.components.cycle.builders import select_best_attempt
from agent_core.components.cycle.observers import (
    BaseRoundObserver,
    RoundIntervention,
)
from agent_core.components.cycle.types import AuditReport, WriterOutput

logger = logging.getLogger(__name__)


# ── BestSoFarObserver ───────────────────────────────────────────────────────


# A callback receiving (best_index_zero_based, best_writer_out, best_audit).
# May be sync or async. Errors are caught + logged; the cycle continues.
BestSoFarCallback = Callable[
    [int, WriterOutput, AuditReport],
    Awaitable[None] | None,
]


class BestSoFarObserver(BaseRoundObserver):
    """Invoke ``callback(best_idx, best_writer_out, best_audit)``
    after every audit, with the highest-scoring attempt seen so far.

    Score selection uses :func:`agent_core.components.cycle.select_best_attempt`
    (highest ``audit.metadata['score']``, ties → latest). The
    observer maintains its own append-only history; it does not read
    cycle internals.

    Designed for the "still-running cycle should always have a usable
    best-so-far artefact persisted somewhere" pattern — most directly
    the MiroVerifier IMO-GVR ``_upload_content`` per-attempt upload
    that protects against agent timeouts. Make ``callback`` upload
    the proof to your sandbox / object store / DB.

    Synthetic audits (writer exceptions, output_missing) typically
    have no ``metadata['score']`` and are treated as worse than any
    scored attempt, so they don't poison the best-so-far selection.
    """

    def __init__(self, callback: BestSoFarCallback) -> None:
        self._callback = callback
        self._history: list[tuple[WriterOutput, AuditReport]] = []

    async def on_audit_done(
        self,
        round_num: int,
        writer_out: WriterOutput,
        audit: AuditReport,
    ) -> RoundIntervention | None:
        self._history.append((writer_out, audit))
        try:
            best_idx, best_w, best_a = select_best_attempt(self._history)
        except ValueError:
            return None
        try:
            ret = self._callback(best_idx, best_w, best_a)
            if inspect.isawaitable(ret):
                await ret
        except Exception:
            logger.exception(
                "BestSoFarObserver callback raised on round %d", round_num,
            )
        return None


# ── PlateauAbortObserver ────────────────────────────────────────────────────


class PlateauAbortObserver(BaseRoundObserver):
    """Abort the cycle when scores have plateaued.

    Triggers :class:`RoundIntervention.ABORT` when the best score in
    the most recent ``plateau_rounds`` rounds is no greater than the
    best score in any earlier round. Useful for cost control on
    always-run-k workflows where the LLM has clearly converged and
    further rounds won't help.

    Reads ``audit.metadata['score']``; rounds without a score
    (synthetic audits, grader errors) are skipped — they neither
    trigger plateau nor reset the counter.

    Parameters
    ----------
    plateau_rounds
        Number of consecutive trailing rounds that must show no
        improvement over historical best to trigger abort. Default
        ``3``. Must be ≥ 1.
    min_rounds
        Skip plateau check until at least this many scored rounds
        have happened. Default ``plateau_rounds`` itself — that is,
        the earliest possible abort is after exactly
        ``plateau_rounds`` consecutive scored rounds with no
        improvement.
    """

    def __init__(
        self,
        plateau_rounds: int = 3,
        *,
        min_rounds: int | None = None,
    ) -> None:
        if plateau_rounds < 1:
            raise ValueError(
                f"plateau_rounds must be >= 1, got {plateau_rounds!r}",
            )
        self._plateau = plateau_rounds
        self._min_rounds = (
            min_rounds if min_rounds is not None else plateau_rounds
        )
        self._scores: list[float] = []

    async def on_audit_done(
        self,
        round_num: int,
        writer_out: WriterOutput,
        audit: AuditReport,
    ) -> RoundIntervention | None:
        del round_num, writer_out
        score = audit.metadata.get("score")
        if not isinstance(score, (int, float)):
            return None
        self._scores.append(float(score))

        if len(self._scores) < self._min_rounds:
            return None
        if len(self._scores) < self._plateau + 1:
            # Need at least one earlier round to compare against.
            return None
        recent = self._scores[-self._plateau:]
        earlier = self._scores[: -self._plateau]
        if max(recent) <= max(earlier):
            logger.info(
                "PlateauAbortObserver: aborting; recent_max=%s earlier_max=%s",
                max(recent), max(earlier),
            )
            return RoundIntervention.ABORT
        return None


# ── MetricsObserver (lightweight logging) ───────────────────────────────────


MetricsCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


class MetricsObserver(BaseRoundObserver):
    """Emit a metrics dict per round.

    Each ``on_audit_done`` fires the callback with::

        {
            "round": <int>,
            "duration_seconds": <float>,
            "score": <float | None>,
            "verdict": <str>,
            "writer_chars": <int>,
            "audit_findings": <int>,
        }

    The default callback logs at INFO level. Pass a custom callback
    to push to Prometheus / a metrics client / a JSON line file.
    Sync or async callbacks both work; exceptions are logged + swallowed.
    """

    def __init__(
        self,
        callback: MetricsCallback | None = None,
    ) -> None:
        self._callback = callback
        self._round_starts: dict[int, float] = {}

    async def on_round_start(
        self,
        round_num: int,
        prev_audit: AuditReport | None,
    ) -> None:
        del prev_audit
        loop = asyncio.get_event_loop()
        self._round_starts[round_num] = loop.time()

    async def on_audit_done(
        self,
        round_num: int,
        writer_out: WriterOutput,
        audit: AuditReport,
    ) -> RoundIntervention | None:
        loop = asyncio.get_event_loop()
        duration = loop.time() - self._round_starts.pop(
            round_num, loop.time(),
        )
        score_meta = audit.metadata.get("score")
        metrics = {
            "round": round_num,
            "duration_seconds": duration,
            "score": (
                float(score_meta)
                if isinstance(score_meta, (int, float))
                else None
            ),
            "verdict": audit.verdict,
            "writer_chars": len(writer_out.content),
            "audit_findings": len(audit.findings),
        }
        if self._callback is None:
            logger.info("cycle round metrics: %s", metrics)
        else:
            try:
                ret = self._callback(metrics)
                if inspect.isawaitable(ret):
                    await ret
            except Exception:
                logger.exception(
                    "MetricsObserver callback raised on round %d", round_num,
                )
        return None
