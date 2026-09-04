"""End-to-end integration smoke for WriteAuditCycle.

Wires SessionBackedWriter + SessionBackedAuditor + WriteAuditCycle
against a deterministic mocked AgentBus. No real LLM in the loop.

Validates the three §7.2 acceptance scenarios:
1. Round-0 success terminates with success=True
2. Iterate-then-success produces a complete two-round history
3. max_rounds_exhausted produces the expected reason + full history
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from agent_core.components.agent_bus.models import (
    CollectResult,
    SubAgentResult,
)
from agent_core.components.cycle import (
    SessionBackedAuditor,
    SessionBackedWriter,
    WriteAuditCycle,
)


def _ok(content: str) -> SubAgentResult:
    return SubAgentResult(
        question="x", role_id="r", final_content=content, success=True,
    )


def _build_bus(content_script: list[str]) -> Any:
    """Bus that returns scripted final_content per submission, in order.

    Each submit_task_to_session triggers the next collect to return the
    next scripted content. We use a counter via list.pop(0).
    """
    pending = list(content_script)

    async def _collect(_job_ids: list[str], **_kwargs: Any) -> CollectResult:
        if not pending:
            raise AssertionError("collect called more times than scripted")
        return CollectResult(completed=[_ok(pending.pop(0))])

    bus = AsyncMock()
    sid_counter = iter(
        f"task-1::session-{i}" for i in range(100)
    )
    bus.create_session = AsyncMock(side_effect=lambda **_: next(sid_counter))
    bus.submit_task_to_session = AsyncMock(return_value="job-x")
    bus.collect = _collect
    return bus


def _make_writer(bus: Any, tmp: Path) -> SessionBackedWriter:
    return SessionBackedWriter(
        bus=bus, task_id="task-1", role_id="writer", name="writer",
        system_prompt="you write", initial_prompt="write the artifact",
        work_dir=tmp,
        output_check=lambda _: [],  # always pass
    )


def _make_auditor(bus: Any) -> SessionBackedAuditor:
    return SessionBackedAuditor(
        bus=bus, task_id="task-1", role_id="auditor",
        system_prompt="emit AuditReport JSON",
        max_turns=10,
    )


# ── Scenario 1: round-0 success ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_smoke_round_0_success(tmp_path: Path) -> None:
    """Writer produces, output_check passes, auditor verdict=success → done."""
    bus = _build_bus(content_script=[
        # Round 0 — writer
        "draft v1",
        # Round 0 — auditor (returns JSON verdict=success)
        '```json\n{"verdict": "success", "findings": [], "summary": "LGTM"}\n```',
    ])
    writer = _make_writer(bus, tmp_path)
    auditor = _make_auditor(bus)
    cycle = WriteAuditCycle(
        writer=writer, auditor=auditor,
        work_dir=tmp_path, output_check=writer.output_check,
        max_rounds=5,
    )
    out = await cycle.run()
    assert out.success is True
    assert out.rounds_used == 1
    assert out.reason == "verdict=success"
    assert len(out.history) == 1
    # Persistence happened
    assert (tmp_path / "audit_round_0.json").exists()


# ── Scenario 2: iterate-then-success ────────────────────────────────────────


@pytest.mark.asyncio
async def test_smoke_iterate_then_success(tmp_path: Path) -> None:
    bus = _build_bus(content_script=[
        # Round 0 — writer
        "draft v1",
        # Round 0 — auditor: iterate
        (
            '```json\n{"verdict": "iterate", "findings": ['
            '{"category": "missing_section", "severity": "error",'
            ' "short_message": "add intro"}], "summary": "fix intro"}\n```'
        ),
        # Round 1 — writer (revised)
        "draft v2 with intro",
        # Round 1 — auditor: success
        '```json\n{"verdict": "success", "findings": [], "summary": "LGTM"}\n```',
    ])
    writer = _make_writer(bus, tmp_path)
    auditor = _make_auditor(bus)
    cycle = WriteAuditCycle(
        writer=writer, auditor=auditor,
        work_dir=tmp_path, output_check=writer.output_check,
        max_rounds=5,
    )
    out = await cycle.run()
    assert out.success is True
    assert out.rounds_used == 2
    assert len(out.history) == 2
    assert out.history[0][1].verdict == "iterate"
    assert out.history[1][1].verdict == "success"
    # Both round files persisted
    assert (tmp_path / "audit_round_0.json").exists()
    assert (tmp_path / "audit_round_1.json").exists()
    # Writer reused the same persistent session: only ONE create_session
    # call from the writer (auditor creates fresh per round → 2 of those).
    # Total create_session calls = 1 (writer) + 2 (auditor) = 3.
    assert bus.create_session.await_count == 3


# ── Scenario 3: max_rounds exhausted ────────────────────────────────────────


@pytest.mark.asyncio
async def test_smoke_max_rounds_exhausted(tmp_path: Path) -> None:
    bus = _build_bus(content_script=[
        # Round 0
        "draft v1",
        '```json\n{"verdict": "iterate", "findings": [], "summary": "redo"}\n```',
        # Round 1
        "draft v2",
        '```json\n{"verdict": "iterate", "findings": [], "summary": "redo"}\n```',
        # Round 2
        "draft v3",
        '```json\n{"verdict": "iterate", "findings": [], "summary": "redo"}\n```',
    ])
    writer = _make_writer(bus, tmp_path)
    auditor = _make_auditor(bus)
    cycle = WriteAuditCycle(
        writer=writer, auditor=auditor,
        work_dir=tmp_path, output_check=writer.output_check,
        max_rounds=3,
    )
    out = await cycle.run()
    assert out.success is False
    assert out.reason == "max_rounds_exhausted"
    assert out.rounds_used == 3
    assert len(out.history) == 3


# ── Scenario 4: output_check failure short-circuits the auditor ─────────────


@pytest.mark.asyncio
async def test_smoke_output_check_failure_short_circuits_auditor(
    tmp_path: Path,
) -> None:
    """Real wiring: when output_check reports missing, no auditor session is created."""
    # Auditor is never reached, so script only has the writer's outputs.
    bus = _build_bus(content_script=["draft v1", "draft v2"])
    writer = _make_writer(bus, tmp_path)
    auditor = _make_auditor(bus)
    # Output_check reports missing every round
    cycle = WriteAuditCycle(
        writer=writer, auditor=auditor,
        work_dir=tmp_path,
        output_check=lambda _: ["sections/intro.md"],
        max_rounds=2,
    )
    out = await cycle.run()
    assert out.success is False
    assert out.reason == "max_rounds_exhausted"
    # Synthetic audits per round — auditor session never created
    assert bus.create_session.await_count == 1  # only the writer
    # Both rounds' synthetic audits are persisted
    assert (tmp_path / "audit_round_0.json").exists()
    assert (tmp_path / "audit_round_1.json").exists()
    # Findings name the missing artifact
    final_findings = out.final_audit.findings if out.final_audit else []
    assert any(f.file == "sections/intro.md" for f in final_findings)
