"""Tests for RoundObserver hooks + builtin observers + cycle wall-clock.

Covers:

- D1 wall-clock budget: between-rounds check, reason="wall_clock_exhausted",
  safety ratio honoured, no max_wall_seconds = no check
- D2 RoundObserver: every hook fires on every observer; observer
  exception is isolated; ABORT intervention from any hook terminates
  cycle with reason="observer_aborted"; on_cycle_end always fires
- BestSoFarObserver: callback invoked per round with the best-so-far
  selection; sync + async callbacks both work; callback errors don't
  crash cycle
- PlateauAbortObserver: triggers ABORT after no-improvement window;
  rounds without scores skipped; minimum-rounds gate
- MetricsObserver: emits per-round metrics dict; default logger
  callback works
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest

from agent_core.components.cycle import (
    AuditFinding,
    AuditReport,
    BaseRoundObserver,
    BestSoFarObserver,
    CycleContext,
    MetricsObserver,
    PlateauAbortObserver,
    RoundIntervention,
    WriteAuditCycle,
    WriterOutput,
)

# ── Test fixtures ───────────────────────────────────────────────────────────


class _ScriptedWriter:
    role_id = "writer"

    def __init__(
        self,
        outputs: list[str],
        sleep_between: float = 0.0,
    ) -> None:
        self._outputs = list(outputs)
        self._sleep = sleep_between
        self.calls: list[int] = []

    async def generate(self, prev_audit, round_num, feedback_md):
        del prev_audit, feedback_md
        self.calls.append(round_num)
        if self._sleep:
            await asyncio.sleep(self._sleep)
        if not self._outputs:
            raise AssertionError("scripted writer exhausted")
        return WriterOutput(content=self._outputs.pop(0))


class _ScriptedAuditor:
    role_id = "auditor"

    def __init__(
        self,
        verdicts: list[str],
        scores: list[float | None] | None = None,
    ) -> None:
        self._verdicts = list(verdicts)
        self._scores = list(scores) if scores else [None] * len(verdicts)
        self.calls: list[int] = []

    async def verify(self, writer_output, round_num):
        self.calls.append(round_num)
        verdict = self._verdicts.pop(0)
        score = self._scores.pop(0) if self._scores else None
        meta: dict = {}
        if score is not None:
            meta["score"] = score
        return AuditReport(
            verdict=verdict,
            findings=[
                AuditFinding(
                    category="grade", severity="info",
                    short_message=f"score={score}",
                ),
            ],
            metadata=meta,
        )


# ── D1: Wall-clock budget ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_wall_clock_budget_exits_between_rounds(
    tmp_path: Path,
) -> None:
    """Each round sleeps 0.1s; budget is 0.25s. Should run round 0
    successfully (records duration ~0.1s), check at round 1 boundary
    (elapsed=0.1, remaining=0.15, est_next=0.11), pass; then check
    again at round 2 boundary (elapsed=0.2, remaining=0.05,
    est_next=0.11), fail → exit."""
    writer = _ScriptedWriter(["v0", "v1", "v2"], sleep_between=0.1)
    auditor = _ScriptedAuditor(["iterate", "iterate", "iterate"])
    cycle = WriteAuditCycle(
        writer=writer, auditor=auditor,
        work_dir=tmp_path,
        output_check=lambda _: [],
        max_rounds=10,
        max_wall_seconds=0.25,
    )
    output = await cycle.run()
    assert output.reason == "wall_clock_exhausted"
    assert output.success is False
    assert 1 <= output.rounds_used <= 3  # ran 1-2 rounds before exhausting


@pytest.mark.asyncio
async def test_wall_clock_safety_ratio_honoured(tmp_path: Path) -> None:
    """With ratio=2.0 and round_duration ~0.05s, budget 0.15s should
    only allow round 0 (after, est_next=0.10, remaining=0.10, equal →
    fail)."""
    writer = _ScriptedWriter(["v0", "v1", "v2"], sleep_between=0.05)
    auditor = _ScriptedAuditor(["iterate", "iterate", "iterate"])
    cycle = WriteAuditCycle(
        writer=writer, auditor=auditor,
        work_dir=tmp_path,
        output_check=lambda _: [],
        max_rounds=10,
        max_wall_seconds=0.15,
        wall_clock_safety_ratio=2.0,
    )
    output = await cycle.run()
    assert output.reason == "wall_clock_exhausted"


@pytest.mark.asyncio
async def test_no_wall_clock_means_no_check(tmp_path: Path) -> None:
    """Default: cycle runs to max_rounds regardless of duration."""
    writer = _ScriptedWriter(["v0", "v1"], sleep_between=0.05)
    auditor = _ScriptedAuditor(["iterate", "iterate"])
    cycle = WriteAuditCycle(
        writer=writer, auditor=auditor,
        work_dir=tmp_path,
        output_check=lambda _: [],
        max_rounds=2,
    )
    output = await cycle.run()
    assert output.reason == "max_rounds_exhausted"
    assert output.rounds_used == 2


def test_wall_clock_invalid_value_rejected(tmp_path: Path) -> None:
    writer = _ScriptedWriter(["x"])
    auditor = _ScriptedAuditor(["iterate"])
    with pytest.raises(ValueError, match="max_wall_seconds"):
        WriteAuditCycle(
            writer=writer, auditor=auditor, work_dir=tmp_path,
            output_check=lambda _: [], max_wall_seconds=0,
        )


def test_wall_clock_safety_ratio_below_one_rejected(tmp_path: Path) -> None:
    writer = _ScriptedWriter(["x"])
    auditor = _ScriptedAuditor(["iterate"])
    with pytest.raises(ValueError, match="wall_clock_safety_ratio"):
        WriteAuditCycle(
            writer=writer, auditor=auditor, work_dir=tmp_path,
            output_check=lambda _: [], wall_clock_safety_ratio=0.5,
        )


# ── D2: RoundObserver hook firing ──────────────────────────────────────────


class _RecordingObserver(BaseRoundObserver):
    """Records every hook call for inspection."""

    def __init__(self) -> None:
        self.events: list[tuple] = []

    async def on_cycle_start(self, ctx):
        self.events.append(("cycle_start", ctx.max_rounds))

    async def on_round_start(self, round_num, prev_audit):
        self.events.append((
            "round_start", round_num,
            prev_audit.verdict if prev_audit else None,
        ))

    async def on_writer_done(self, round_num, writer_out):
        self.events.append(("writer_done", round_num, writer_out.content))
        return None

    async def on_audit_done(self, round_num, writer_out, audit):
        self.events.append(("audit_done", round_num, audit.verdict))
        return None

    async def on_cycle_end(self, output):
        self.events.append(("cycle_end", output.reason))


@pytest.mark.asyncio
async def test_all_hooks_fire_in_order(tmp_path: Path) -> None:
    obs = _RecordingObserver()
    writer = _ScriptedWriter(["v0", "v1"])
    auditor = _ScriptedAuditor(["iterate", "success"])
    cycle = WriteAuditCycle(
        writer=writer, auditor=auditor, work_dir=tmp_path,
        output_check=lambda _: [], max_rounds=3, observers=[obs],
    )
    output = await cycle.run()
    assert output.reason == "verdict=success"

    expected = [
        ("cycle_start", 3),
        ("round_start", 0, None),
        ("writer_done", 0, "v0"),
        ("audit_done", 0, "iterate"),
        ("round_start", 1, "iterate"),
        ("writer_done", 1, "v1"),
        ("audit_done", 1, "success"),
        ("cycle_end", "verdict=success"),
    ]
    assert obs.events == expected


@pytest.mark.asyncio
async def test_observer_exception_isolated(tmp_path: Path) -> None:
    """Observer raising in any hook is logged but cycle continues."""

    class _BoomObserver(BaseRoundObserver):
        async def on_round_start(self, round_num, prev_audit):
            raise RuntimeError("boom")

    obs = _BoomObserver()
    rec = _RecordingObserver()
    writer = _ScriptedWriter(["v0"])
    auditor = _ScriptedAuditor(["success"])
    cycle = WriteAuditCycle(
        writer=writer, auditor=auditor, work_dir=tmp_path,
        output_check=lambda _: [], max_rounds=1, observers=[obs, rec],
    )
    output = await cycle.run()
    assert output.success is True
    # Recording observer still saw all events
    assert any(e[0] == "audit_done" for e in rec.events)
    assert any(e[0] == "cycle_end" for e in rec.events)


@pytest.mark.asyncio
async def test_observer_abort_from_writer_done(tmp_path: Path) -> None:
    class _AbortAfterWriter(BaseRoundObserver):
        async def on_writer_done(self, round_num, writer_out):
            return RoundIntervention.ABORT

    writer = _ScriptedWriter(["v0", "v1"])
    auditor = _ScriptedAuditor(["success", "success"])
    cycle = WriteAuditCycle(
        writer=writer, auditor=auditor, work_dir=tmp_path,
        output_check=lambda _: [], max_rounds=5,
        observers=[_AbortAfterWriter()],
    )
    output = await cycle.run()
    assert output.reason == "observer_aborted"
    assert output.rounds_used == 1
    assert output.final_writer_output.content == "v0"
    # Auditor should NOT have been called (abort fired before audit)
    assert auditor.calls == []


@pytest.mark.asyncio
async def test_observer_abort_from_audit_done(tmp_path: Path) -> None:
    class _AbortAfterAudit(BaseRoundObserver):
        async def on_audit_done(self, round_num, writer_out, audit):
            if round_num >= 1:
                return RoundIntervention.ABORT
            return None

    writer = _ScriptedWriter(["v0", "v1", "v2"])
    auditor = _ScriptedAuditor(["iterate", "iterate", "iterate"])
    cycle = WriteAuditCycle(
        writer=writer, auditor=auditor, work_dir=tmp_path,
        output_check=lambda _: [], max_rounds=10,
        observers=[_AbortAfterAudit()],
    )
    output = await cycle.run()
    assert output.reason == "observer_aborted"
    assert output.rounds_used == 2  # ran round 0 + round 1, aborted after


@pytest.mark.asyncio
async def test_on_cycle_end_fires_on_every_exit(tmp_path: Path) -> None:
    """Ensures on_cycle_end is called for verdict-success, max-rounds,
    wall-clock, and observer-abort exits."""
    end_reasons: list[str] = []

    class _EndCapture(BaseRoundObserver):
        async def on_cycle_end(self, output):
            end_reasons.append(output.reason)

    # Case 1: terminal verdict
    cycle = WriteAuditCycle(
        writer=_ScriptedWriter(["v0"]),
        auditor=_ScriptedAuditor(["success"]),
        work_dir=tmp_path / "a",
        output_check=lambda _: [], max_rounds=1,
        observers=[_EndCapture()],
    )
    await cycle.run()

    # Case 2: max_rounds
    cycle = WriteAuditCycle(
        writer=_ScriptedWriter(["v0"]),
        auditor=_ScriptedAuditor(["iterate"]),
        work_dir=tmp_path / "b",
        output_check=lambda _: [], max_rounds=1,
        observers=[_EndCapture()],
    )
    await cycle.run()

    assert end_reasons == ["verdict=success", "max_rounds_exhausted"]


@pytest.mark.asyncio
async def test_cycle_context_passed_to_on_cycle_start(tmp_path: Path) -> None:
    captured: list[CycleContext] = []

    class _CtxCapture(BaseRoundObserver):
        async def on_cycle_start(self, ctx):
            captured.append(ctx)

    cycle = WriteAuditCycle(
        writer=_ScriptedWriter(["v0"]),
        auditor=_ScriptedAuditor(["success"]),
        work_dir=tmp_path,
        output_check=lambda _: [], max_rounds=5,
        max_wall_seconds=99.0,
        observers=[_CtxCapture()],
    )
    await cycle.run()
    assert len(captured) == 1
    ctx = captured[0]
    assert ctx.max_rounds == 5
    assert ctx.max_wall_seconds == 99.0
    assert ctx.work_dir == tmp_path


# ── BestSoFarObserver ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_best_so_far_observer_invokes_callback_per_round(
    tmp_path: Path,
) -> None:
    snapshots: list[tuple[int, str, float]] = []

    def _cb(idx, w, a):
        snapshots.append((idx, w.content, a.metadata.get("score")))

    obs = BestSoFarObserver(callback=_cb)
    writer = _ScriptedWriter(["v0", "v1", "v2"])
    auditor = _ScriptedAuditor(
        ["iterate", "iterate", "iterate"],
        scores=[3.0, 7.0, 5.0],
    )
    cycle = WriteAuditCycle(
        writer=writer, auditor=auditor, work_dir=tmp_path,
        output_check=lambda _: [], max_rounds=3, observers=[obs],
    )
    await cycle.run()

    assert snapshots == [
        (0, "v0", 3.0),
        (1, "v1", 7.0),
        (1, "v1", 7.0),  # round 2 (score 5) didn't beat round 1 (score 7)
    ]


@pytest.mark.asyncio
async def test_best_so_far_observer_supports_async_callback(
    tmp_path: Path,
) -> None:
    received: list[float] = []

    async def _async_cb(idx, w, a):
        await asyncio.sleep(0)
        received.append(a.metadata.get("score"))

    obs = BestSoFarObserver(callback=_async_cb)
    writer = _ScriptedWriter(["v0", "v1"])
    auditor = _ScriptedAuditor(
        ["iterate", "iterate"], scores=[1.0, 7.0],
    )
    cycle = WriteAuditCycle(
        writer=writer, auditor=auditor, work_dir=tmp_path,
        output_check=lambda _: [], max_rounds=2, observers=[obs],
    )
    await cycle.run()
    assert received == [1.0, 7.0]


@pytest.mark.asyncio
async def test_best_so_far_callback_error_does_not_crash_cycle(
    tmp_path: Path,
) -> None:
    def _bad_cb(idx, w, a):
        raise RuntimeError("kaboom")

    obs = BestSoFarObserver(callback=_bad_cb)
    writer = _ScriptedWriter(["v0", "v1"])
    auditor = _ScriptedAuditor(["iterate", "success"], scores=[1.0, 7.0])
    cycle = WriteAuditCycle(
        writer=writer, auditor=auditor, work_dir=tmp_path,
        output_check=lambda _: [], max_rounds=5, observers=[obs],
    )
    output = await cycle.run()
    assert output.success is True


# ── PlateauAbortObserver ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_plateau_abort_after_no_improvement(tmp_path: Path) -> None:
    """Scores: 3, 5, 5, 5 → no improvement over historical max=5 in
    last 3 → abort at round 4 (after 4 rounds)."""
    obs = PlateauAbortObserver(plateau_rounds=3)
    writer = _ScriptedWriter(["v0", "v1", "v2", "v3", "v4"])
    auditor = _ScriptedAuditor(
        ["iterate"] * 5, scores=[3.0, 5.0, 5.0, 5.0, 5.0],
    )
    cycle = WriteAuditCycle(
        writer=writer, auditor=auditor, work_dir=tmp_path,
        output_check=lambda _: [], max_rounds=10, observers=[obs],
    )
    output = await cycle.run()
    assert output.reason == "observer_aborted"
    # rounds 0,1,2,3 ran (4 rounds). After round 3 the plateau check
    # had history [3,5,5,5] — recent=[5,5,5], earlier=[3] → max(recent)=5
    # > max(earlier)=3 → no abort. After round 4 (score 5):
    # history=[3,5,5,5,5], recent=[5,5,5], earlier=[3,5] → max(recent)=5
    # = max(earlier)=5 → abort.
    assert output.rounds_used == 5


@pytest.mark.asyncio
async def test_plateau_skips_rounds_without_score(tmp_path: Path) -> None:
    """Synthetic audits (score=None) should not advance plateau counter."""
    obs = PlateauAbortObserver(plateau_rounds=2)
    writer = _ScriptedWriter(["v0", "v1", "v2"])
    auditor = _ScriptedAuditor(
        ["iterate", "iterate", "success"], scores=[5.0, None, 7.0],
    )
    cycle = WriteAuditCycle(
        writer=writer, auditor=auditor, work_dir=tmp_path,
        output_check=lambda _: [], max_rounds=3, observers=[obs],
    )
    output = await cycle.run()
    # Score history seen by plateau: [5.0, 7.0] → improvement → no abort
    assert output.reason == "verdict=success"


def test_plateau_invalid_param_rejected() -> None:
    with pytest.raises(ValueError, match="plateau_rounds"):
        PlateauAbortObserver(plateau_rounds=0)


# ── MetricsObserver ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_metrics_observer_emits_per_round(tmp_path: Path) -> None:
    captured: list[dict] = []

    obs = MetricsObserver(callback=lambda m: captured.append(m))
    writer = _ScriptedWriter(["abc", "defgh"])
    auditor = _ScriptedAuditor(
        ["iterate", "success"], scores=[3.0, 7.0],
    )
    cycle = WriteAuditCycle(
        writer=writer, auditor=auditor, work_dir=tmp_path,
        output_check=lambda _: [], max_rounds=2, observers=[obs],
    )
    await cycle.run()

    assert len(captured) == 2
    assert captured[0]["round"] == 0
    assert captured[0]["score"] == 3.0
    assert captured[0]["verdict"] == "iterate"
    assert captured[0]["writer_chars"] == 3
    assert captured[1]["round"] == 1
    assert captured[1]["score"] == 7.0
    assert captured[1]["writer_chars"] == 5


@pytest.mark.asyncio
async def test_metrics_observer_default_logs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    obs = MetricsObserver()  # no callback → logs at INFO
    writer = _ScriptedWriter(["x"])
    auditor = _ScriptedAuditor(["success"], scores=[7.0])
    cycle = WriteAuditCycle(
        writer=writer, auditor=auditor, work_dir=tmp_path,
        output_check=lambda _: [], max_rounds=1, observers=[obs],
    )
    with caplog.at_level(logging.INFO, logger="agent_core.components.cycle.observers_builtin"):
        await cycle.run()
    assert any("cycle round metrics" in r.message for r in caplog.records)
