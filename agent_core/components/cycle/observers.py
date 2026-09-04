"""RoundObserver — pluggable hooks for :class:`WriteAuditCycle`.

Mirrors :class:`agent_core.loop_types.LoopObserver` but at cycle
granularity (rounds, not LLM turns). Lets callers inject behaviour
between rounds without subclassing the cycle:

- best-so-far snapshotting / external upload
- mid-cycle metrics emission
- score-plateau early termination
- ATIF / external trace export
- domain-specific intervention (e.g. abort if 3 rounds saw the same
  exception)

The cycle never trusts an observer to do the right thing — every
hook call is wrapped in try/except, exceptions are logged and
swallowed so observer bugs cannot kill a long-running cycle.

Hook ordering per round::

    on_round_start(round_num, prev_audit)
        ↓
    writer.generate()           # may raise → synth audit; on_writer_done skipped
        ↓
    on_writer_done(round_num, writer_out)        → may abort
        ↓
    output_check + auditor.verify()   (or synth audit)
        ↓
    on_audit_done(round_num, writer_out, audit)  → may abort

Outside the round loop:

    on_cycle_start(ctx)   — once, before round 0
    on_cycle_end(output)  — once, on every exit path (including
                            wall-clock / observer-abort / max-rounds)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

from agent_core.components.cycle.types import AuditReport, CycleOutput, WriterOutput


class RoundIntervention(StrEnum):
    """Verdict an observer can return from a per-round hook.

    ``CONTINUE`` (the default for any hook returning ``None``) lets
    the cycle proceed normally. ``ABORT`` causes the cycle to
    terminate immediately with ``CycleOutput.success=False`` and
    ``reason="observer_aborted"``. The current round's writer output
    + audit (if any) are preserved in ``final_writer_output`` /
    ``final_audit`` and the partial history is returned as usual.
    """

    CONTINUE = "continue"
    ABORT = "abort"


@dataclass
class CycleContext:
    """Read-only handle to cycle state, passed to ``on_cycle_start``.

    Observers should treat ``history_so_far`` as a *live view*: the
    cycle keeps appending to the same list as rounds complete, so
    later hooks can inspect prior rounds without the cycle re-passing
    them. Observers should not mutate the list.
    """

    work_dir: Path
    max_rounds: int
    max_wall_seconds: float | None
    history_so_far: list[tuple[WriterOutput, AuditReport]] = field(
        default_factory=list[tuple[WriterOutput, AuditReport]],
    )


@runtime_checkable
class RoundObserver(Protocol):
    """Pluggable per-round hook for :class:`WriteAuditCycle`.

    All methods are async. All methods are optional in spirit — the
    cycle calls every hook on every observer, so implementations
    that don't care about a particular event simply ``return None``
    or ``pass``. (Use :class:`BaseRoundObserver` for a default-noop
    base class.)

    Methods returning :class:`RoundIntervention` may abort the cycle
    by returning :attr:`RoundIntervention.ABORT`. Other return values
    (including ``None``) are treated as :attr:`RoundIntervention.CONTINUE`.
    """

    async def on_cycle_start(self, ctx: CycleContext) -> None:
        ...

    async def on_round_start(
        self,
        round_num: int,
        prev_audit: AuditReport | None,
    ) -> None:
        ...

    async def on_writer_done(
        self,
        round_num: int,
        writer_out: WriterOutput,
    ) -> RoundIntervention | None:
        ...

    async def on_audit_done(
        self,
        round_num: int,
        writer_out: WriterOutput,
        audit: AuditReport,
    ) -> RoundIntervention | None:
        ...

    async def on_cycle_end(self, output: CycleOutput) -> None:
        ...


class BaseRoundObserver:
    """Default-noop base class for :class:`RoundObserver` impls.

    Subclass and override only the hooks you care about. Implements
    every Protocol method as ``return None``.
    """

    async def on_cycle_start(self, ctx: CycleContext) -> None:
        return None

    async def on_round_start(
        self,
        round_num: int,
        prev_audit: AuditReport | None,
    ) -> None:
        return None

    async def on_writer_done(
        self,
        round_num: int,
        writer_out: WriterOutput,
    ) -> RoundIntervention | None:
        return None

    async def on_audit_done(
        self,
        round_num: int,
        writer_out: WriterOutput,
        audit: AuditReport,
    ) -> RoundIntervention | None:
        return None

    async def on_cycle_end(self, output: CycleOutput) -> None:
        return None
