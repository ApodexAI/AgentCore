"""WriteAuditCycle — iterate-until-pass loop over Writer + Auditor.

Pseudocode (FR2)::

    for round_num in 0 .. max_rounds:
        feedback_md = renderer.render(prev_audit) if prev_audit else ""
        try:
            writer_out = await writer.generate(prev_audit, round_num, feedback_md)
        except Exception:
            prev_audit = synth_audit_from_exception(...)
            persist(prev_audit); continue
        missing = output_check(work_dir)
        if missing:
            prev_audit = synth_audit_from_missing(missing)
            persist(prev_audit); continue
        audit = await auditor.verify(writer_out, round_num)
        persist(audit)
        if audit.verdict in terminal_set:
            return CycleOutput(success=verdict_is_success, ...)
        prev_audit = audit
    return CycleOutput(success=False, reason="max_rounds_exhausted", ...)

Invariants:

- Every round persists exactly one ``audit_round_N.json`` to
  ``work_dir`` — including synthetic audits constructed by the cycle
  for output_check failures and writer exceptions (FR4).
- Writer exceptions are **never** propagated to the caller. They become
  ``writer_exception`` findings and the loop continues (FR10). The only
  terminations are: terminal verdict, max_rounds exhaustion, or
  cancellation from outside.
- Auditor is **not invoked** when ``output_check`` reports missing
  artifacts — auditing vacuous output is a known false-positive source
  (FR5). The cycle synthesizes a structured ``output_missing`` audit
  itself.
"""
from __future__ import annotations

import logging
import time
import traceback
from collections import deque
from collections.abc import Callable, Iterable
from pathlib import Path

from agent_core.components.cycle.default_renderer import DefaultFeedbackRenderer
from agent_core.components.cycle.observers import (
    CycleContext,
    RoundIntervention,
    RoundObserver,
)
from agent_core.components.cycle.protocols import (
    CycleAuditor,
    FeedbackRenderer,
)
from agent_core.components.cycle.types import (
    AuditFinding,
    AuditReport,
    CycleOutput,
    WriterOutput,
)
from agent_core.components.verifier.protocols import Generator

logger = logging.getLogger(__name__)


# Default verdict set that *terminates* the loop. ``success`` ends with
# ``CycleOutput.success=True``; ``abandon`` ends with ``False``. Callers
# override via ``terminal_verdicts`` + ``success_verdicts`` constructor
# arguments.
_DEFAULT_TERMINAL: frozenset[str] = frozenset({"success", "abandon"})
_DEFAULT_SUCCESS: frozenset[str] = frozenset({"success"})


class WriteAuditCycle:
    """Generic iterate-until-pass orchestrator.

    Parameters
    ----------
    writer
        Caller-supplied :class:`Writer` implementation.
    auditor
        Caller-supplied :class:`Auditor` implementation.
    work_dir
        Filesystem root for artifacts + per-round audit JSON. Created if
        absent.
    output_check
        ``(work_dir) -> list[str]`` returning the list of *missing*
        required artifacts. Empty list means "all required artifacts
        present, proceed to audit". Required.
    feedback_renderer
        Defaults to :class:`DefaultFeedbackRenderer`.
    max_rounds
        Hard upper bound on writer attempts. Default 10.
    terminal_verdicts
        Set of verdict strings that terminate the loop. Default
        ``{"success", "abandon"}``. Free-form strings — see PRD §10.2.
    success_verdicts
        Subset of ``terminal_verdicts`` that count as ``success=True``.
        Default ``{"success"}``.
    audit_path_template
        Filename pattern for per-round persistence. Default
        ``"audit_round_{round}.json"``.
    max_wall_seconds
        Optional wall-clock budget (seconds). When set, the cycle
        checks remaining time **between rounds** (never mid-round —
        protects partial work). If the next-round estimate
        (``wall_clock_safety_ratio × max(recent durations)``) exceeds
        the remaining budget, the cycle exits cleanly with
        ``reason="wall_clock_exhausted"``. Default ``None`` (no
        budget). Mirrors MiroVerifier IMO-GVR's deadline behaviour.
    wall_clock_safety_ratio
        Multiplier on the maximum of recent round durations when
        estimating whether the next round will fit in the remaining
        budget. Default ``1.1``. Must be ≥ 1.0.
    observers
        Optional list of :class:`RoundObserver` instances to receive
        per-round hooks (start/end of cycle, after writer, after
        audit). See :mod:`agent_core.components.cycle.observers`. Observer
        exceptions are caught + logged so a buggy observer can never
        kill the cycle. Observers may return
        :attr:`RoundIntervention.ABORT` from ``on_writer_done`` /
        ``on_audit_done`` to terminate the cycle early with
        ``reason="observer_aborted"``.
    """

    def __init__(
        self,
        *,
        writer: Generator,
        auditor: CycleAuditor,
        work_dir: Path | str,
        output_check: Callable[[Path], list[str]],
        feedback_renderer: FeedbackRenderer | None = None,
        max_rounds: int = 10,
        terminal_verdicts: Iterable[str] = _DEFAULT_TERMINAL,
        success_verdicts: Iterable[str] = _DEFAULT_SUCCESS,
        audit_path_template: str = "audit_round_{round}.json",
        max_wall_seconds: float | None = None,
        wall_clock_safety_ratio: float = 1.1,
        observers: list[RoundObserver] | None = None,
    ) -> None:
        if max_rounds < 1:
            raise ValueError(
                f"max_rounds must be >= 1, got {max_rounds!r}",
            )
        if max_wall_seconds is not None and max_wall_seconds <= 0:
            raise ValueError(
                f"max_wall_seconds must be > 0 when set, "
                f"got {max_wall_seconds!r}",
            )
        if wall_clock_safety_ratio < 1.0:
            raise ValueError(
                f"wall_clock_safety_ratio must be >= 1.0, "
                f"got {wall_clock_safety_ratio!r}",
            )
        self.writer = writer
        self.auditor = auditor
        self.work_dir = Path(work_dir)
        self.output_check = output_check
        self.renderer = feedback_renderer or DefaultFeedbackRenderer(
            success_verdicts=success_verdicts,
        )
        self.max_rounds = max_rounds
        self.terminal_verdicts = frozenset(terminal_verdicts)
        self.success_verdicts = frozenset(success_verdicts)
        self.audit_path_template = audit_path_template
        self.max_wall_seconds = max_wall_seconds
        self.wall_clock_safety_ratio = wall_clock_safety_ratio
        self._observers: list[RoundObserver] = list(observers or [])
        if not self.success_verdicts.issubset(self.terminal_verdicts):
            raise ValueError(
                "success_verdicts must be a subset of terminal_verdicts; "
                f"got success={set(self.success_verdicts)!r} "
                f"terminal={set(self.terminal_verdicts)!r}",
            )

    async def run(self) -> CycleOutput:
        """Run the cycle until terminal verdict, max_rounds exhaustion,
        wall-clock budget exhaustion, or observer-driven abort.

        Never raises — writer exceptions are captured as
        ``writer_exception`` findings (FR10) and observer exceptions
        are caught + logged.
        """
        self.work_dir.mkdir(parents=True, exist_ok=True)
        prev_audit: AuditReport | None = None
        history: list[tuple[WriterOutput, AuditReport]] = []
        last_writer_output: WriterOutput | None = None

        cycle_start = time.monotonic()
        round_durations: deque[float] = deque(maxlen=3)

        ctx = CycleContext(
            work_dir=self.work_dir,
            max_rounds=self.max_rounds,
            max_wall_seconds=self.max_wall_seconds,
            history_so_far=history,  # live view
        )
        await self._fire("on_cycle_start", ctx)

        for round_num in range(self.max_rounds):
            # Wall-clock check (between rounds; never mid-round).
            if (
                self.max_wall_seconds is not None
                and round_durations
            ):
                elapsed = time.monotonic() - cycle_start
                remaining = self.max_wall_seconds - elapsed
                est_next = (
                    max(round_durations) * self.wall_clock_safety_ratio
                )
                if remaining < est_next:
                    logger.info(
                        "Wall-clock budget exhausted: remaining=%.2fs "
                        "est_next=%.2fs (rounds_used=%d)",
                        remaining, est_next, round_num,
                    )
                    return await self._finalize(
                        success=False,
                        rounds_used=round_num,
                        final_writer_output=last_writer_output,
                        final_audit=prev_audit,
                        history=history,
                        reason="wall_clock_exhausted",
                    )

            round_t0 = time.monotonic()
            await self._fire("on_round_start", round_num, prev_audit)

            feedback_md = (
                self.renderer.render(prev_audit) if prev_audit else ""
            )

            try:
                writer_out = await self.writer.generate(
                    prev_audit, round_num, feedback_md,
                )
            except Exception as exc:
                logger.warning(
                    "Writer raised on round %d: %s", round_num, exc,
                )
                synth = _synth_writer_exception_audit(exc)
                self._persist(synth, round_num)
                empty_out = WriterOutput(content="")
                history.append((empty_out, synth))
                round_durations.append(time.monotonic() - round_t0)
                if (
                    await self._fire_intervention(
                        "on_audit_done", round_num, empty_out, synth,
                    )
                    == RoundIntervention.ABORT
                ):
                    return await self._finalize(
                        success=False,
                        rounds_used=round_num + 1,
                        final_writer_output=empty_out,
                        final_audit=synth,
                        history=history,
                        reason="observer_aborted",
                    )
                prev_audit = synth
                last_writer_output = empty_out
                continue

            if (
                await self._fire_intervention(
                    "on_writer_done", round_num, writer_out,
                )
                == RoundIntervention.ABORT
            ):
                return await self._finalize(
                    success=False,
                    rounds_used=round_num + 1,
                    final_writer_output=writer_out,
                    final_audit=prev_audit,
                    history=history,
                    reason="observer_aborted",
                )

            missing = self.output_check(self.work_dir)
            if missing:
                logger.info(
                    "Round %d: output_check missing=%s — auditor skipped",
                    round_num, missing,
                )
                synth = _synth_output_missing_audit(missing)
                self._persist(synth, round_num)
                history.append((writer_out, synth))
                round_durations.append(time.monotonic() - round_t0)
                if (
                    await self._fire_intervention(
                        "on_audit_done", round_num, writer_out, synth,
                    )
                    == RoundIntervention.ABORT
                ):
                    return await self._finalize(
                        success=False,
                        rounds_used=round_num + 1,
                        final_writer_output=writer_out,
                        final_audit=synth,
                        history=history,
                        reason="observer_aborted",
                    )
                prev_audit = synth
                last_writer_output = writer_out
                continue

            audit = await self.auditor.verify(writer_out, round_num)
            self._persist(audit, round_num)
            history.append((writer_out, audit))
            round_durations.append(time.monotonic() - round_t0)
            last_writer_output = writer_out

            if (
                await self._fire_intervention(
                    "on_audit_done", round_num, writer_out, audit,
                )
                == RoundIntervention.ABORT
            ):
                return await self._finalize(
                    success=False,
                    rounds_used=round_num + 1,
                    final_writer_output=writer_out,
                    final_audit=audit,
                    history=history,
                    reason="observer_aborted",
                )

            if audit.verdict in self.terminal_verdicts:
                success = audit.verdict in self.success_verdicts
                return await self._finalize(
                    success=success,
                    rounds_used=round_num + 1,
                    final_writer_output=writer_out,
                    final_audit=audit,
                    history=history,
                    reason=f"verdict={audit.verdict}",
                )

            prev_audit = audit

        # Loop fell through — max_rounds exhausted with no terminal
        # verdict. last_writer_output / prev_audit reflect the final
        # round's outcome.
        return await self._finalize(
            success=False,
            rounds_used=self.max_rounds,
            final_writer_output=last_writer_output,
            final_audit=prev_audit,
            history=history,
            reason="max_rounds_exhausted",
        )

    async def _finalize(
        self,
        *,
        success: bool,
        rounds_used: int,
        final_writer_output: WriterOutput | None,
        final_audit: AuditReport | None,
        history: list[tuple[WriterOutput, AuditReport]],
        reason: str,
    ) -> CycleOutput:
        """Build the CycleOutput, fire on_cycle_end, return."""
        output = CycleOutput(
            success=success,
            rounds_used=rounds_used,
            final_writer_output=final_writer_output,
            final_audit=final_audit,
            history=history,
            reason=reason,
        )
        await self._fire("on_cycle_end", output)
        return output

    async def _fire(self, hook_name: str, *args: object) -> None:
        """Fire a no-return hook on every observer; isolate errors."""
        for obs in self._observers:
            try:
                await getattr(obs, hook_name)(*args)
            except Exception:
                logger.exception(
                    "Observer %r raised in %s", obs, hook_name,
                )

    async def _fire_intervention(
        self, hook_name: str, *args: object,
    ) -> RoundIntervention:
        """Fire an intervention hook; ``ABORT`` from any observer wins.

        Errors from individual observers are isolated; the hook is
        still called on every other observer (so abort decisions are
        not silently masked by an unrelated bug).
        """
        result = RoundIntervention.CONTINUE
        for obs in self._observers:
            try:
                ret = await getattr(obs, hook_name)(*args)
                if ret == RoundIntervention.ABORT:
                    result = RoundIntervention.ABORT
            except Exception:
                logger.exception(
                    "Observer %r raised in %s", obs, hook_name,
                )
        return result

    def _persist(self, audit: AuditReport, round_num: int) -> None:
        """Write ``audit_round_{round}.json`` under work_dir.

        Failures are logged but never raised — losing a trail entry is
        annoying but must not abort the cycle.
        """
        path = self.work_dir / self.audit_path_template.format(
            round=round_num,
        )
        try:
            path.write_text(audit.to_json(), encoding="utf-8")
        except Exception:
            logger.exception(
                "Failed to persist audit trail to %s", path,
            )


# ── Synthetic audits constructed by the cycle ───────────────────────────────


def _synth_output_missing_audit(missing: list[str]) -> AuditReport:
    """Build a synthetic AuditReport for a failed output_check (FR5)."""
    findings = [
        AuditFinding(
            category="output_missing",
            severity="error",
            short_message=f"Required artifact missing: {path}",
            detailed_message=(
                f"The cycle's output_check reported {path!r} as missing "
                f"after the writer round. Auditor was not invoked. "
                f"Re-attempt the round and ensure this artifact is "
                f"produced."
            ),
            file=path,
            suggested_action=f"Produce the artifact at {path!r}.",
        )
        for path in missing
    ]
    return AuditReport(
        verdict="iterate",
        findings=findings,
        summary=(
            f"Required output artifacts are missing "
            f"({len(missing)}). The auditor was not invoked because "
            f"auditing missing artifacts produces unreliable verdicts."
        ),
        confidence=1.0,
        metadata={"synthetic": True, "reason": "output_missing"},
    )


def _synth_writer_exception_audit(exc: Exception) -> AuditReport:
    """Build a synthetic AuditReport for a writer exception (FR10)."""
    tb = traceback.format_exc()
    return AuditReport(
        verdict="iterate",
        findings=[
            AuditFinding(
                category="writer_exception",
                severity="error",
                short_message=(
                    f"{type(exc).__name__}: {str(exc)[:200]}"
                ),
                detailed_message=tb,
                suggested_action=(
                    "Inspect the writer-side error and re-attempt "
                    "the round. The cycle will continue regardless."
                ),
            ),
        ],
        summary=(
            f"The writer raised {type(exc).__name__} on this round. "
            f"The cycle continues to the next round."
        ),
        confidence=1.0,
        metadata={"synthetic": True, "reason": "writer_exception"},
    )
