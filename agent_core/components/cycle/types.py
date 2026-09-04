"""Structured data model for the WriteAuditCycle.

These types form the cycle's stable contract:

- :class:`AuditFinding` — one structured diagnostic produced by an auditor.
- :class:`AuditReport` — full audit verdict + structured findings + free-form prose.
- :class:`WriterOutput` — what a writer round produced.
- :class:`CycleOutput` — final cycle result with full history.

All four round-trip losslessly through :meth:`to_json` / :meth:`from_json`,
which is how the per-round audit trail is persisted on disk (FR4).

Hybrid structured + free-form output (FR3): :class:`AuditReport` carries
both a list of structured ``findings`` (machine-readable, parsed by the
orchestrator to decide loop continuation) and a free-form ``summary``
paragraph (human/LLM-readable, consumed by the writer to actually
understand the critique). LLMs are asked to emit JSON inside a fenced
block plus prose outside, which is more reliable than encoding prose
inside JSON.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Severity = Literal["error", "warning", "info"]


# ── AuditFinding ────────────────────────────────────────────────────────────


@dataclass
class AuditFinding:
    """One structured diagnostic produced by an auditor.

    ``category`` is free-form but conventionally drawn from a stable
    vocabulary (e.g. ``"missing_section"``, ``"citation_invalid"``,
    ``"writer_exception"``, ``"output_missing"``, ``"auditor_parse_failure"``).
    Free-form is preferred over a closed enum because vocabularies vary
    by domain (paper, code, dataset, plan).

    ``target_role`` is an optional hint about which writer-side role
    should act on this finding when the cycle has multiple writers
    (e.g. method-section writer vs. results-section writer). The cycle
    itself does not route on this field; it is informational for the
    feedback renderer.

    ``metadata`` is an extensible bag for caller-specific payload.
    """

    category: str
    severity: Severity
    short_message: str
    detailed_message: str = ""
    file: str | None = None
    line: int | None = None
    snippet: str = ""
    suggested_action: str = ""
    target_role: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuditFinding:
        return cls(
            category=data["category"],
            severity=data["severity"],
            short_message=data["short_message"],
            detailed_message=data.get("detailed_message", ""),
            file=data.get("file"),
            line=data.get("line"),
            snippet=data.get("snippet", ""),
            suggested_action=data.get("suggested_action", ""),
            target_role=data.get("target_role"),
            metadata=dict(data.get("metadata") or {}),
        )


# ── AuditReport ─────────────────────────────────────────────────────────────


@dataclass
class AuditReport:
    """Full audit verdict for one cycle round.

    ``verdict`` is free-form. The cycle compares it against
    ``terminal_verdicts`` (default ``{"success", "abandon"}``) to decide
    whether to terminate. Common values: ``"success"`` (artifact passes),
    ``"iterate"`` (writer should revise), ``"abandon"`` (unrecoverable).

    ``findings`` is the structured part — what the orchestrator reads.
    ``summary`` is the free-form part — what the writer reads.

    ``confidence`` is the auditor's self-reported confidence in its own
    verdict, in ``[0.0, 1.0]``. Optional but recommended; renderers may
    surface it to downstream readers.

    ``raw_text`` is the auditor's full unparsed output, kept for
    debugging and replay. The default renderer does not include it in
    the next-round prompt — only the structured findings + summary.
    """

    verdict: str
    findings: list[AuditFinding] = field(default_factory=list[AuditFinding])
    summary: str = ""
    confidence: float = 0.0
    raw_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "findings": [f.to_dict() for f in self.findings],
            "summary": self.summary,
            "confidence": self.confidence,
            "raw_text": self.raw_text,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuditReport:
        raw_findings: list[Any] = data.get("findings") or []
        return cls(
            verdict=data["verdict"],
            findings=[AuditFinding.from_dict(f) for f in raw_findings],
            summary=data.get("summary", ""),
            confidence=float(data.get("confidence", 0.0)),
            raw_text=data.get("raw_text", ""),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> AuditReport:
        return cls.from_dict(json.loads(text))


# ── WriterOutput ────────────────────────────────────────────────────────────


@dataclass
class WriterOutput:
    """What a writer round produced.

    ``content`` is the writer's final text (the LLM's last AI message).
    ``files`` lists artifact paths the writer touched, relative to
    ``work_dir`` — populated by the writer implementation, not by the
    cycle. ``message_count`` is the number of messages in the writer's
    session at the end of this round (informational; useful for trimmer
    diagnostics).

    ``loop_result`` is the underlying ``AgentLoopResult`` when the
    writer was run via ``run_agent_loop``. Optional — protocols-based
    writers that don't use the agent loop may leave it ``None``.
    """

    content: str
    files: list[str] = field(default_factory=list[str])
    message_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])
    loop_result: Any = None

    def to_dict(self) -> dict[str, Any]:
        # ``loop_result`` is intentionally dropped — it carries
        # langchain message objects which are not JSON-serialisable.
        # The audit trail is for the AuditReport; writer output is
        # reconstructable from the writer's session messages.
        return {
            "content": self.content,
            "files": list(self.files),
            "message_count": self.message_count,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WriterOutput:
        return cls(
            content=data["content"],
            files=list(data.get("files") or []),
            message_count=int(data.get("message_count", 0)),
            metadata=dict(data.get("metadata") or {}),
            loop_result=None,
        )


# ── CycleOutput ─────────────────────────────────────────────────────────────


@dataclass
class CycleOutput:
    """Final cycle result.

    ``success`` is ``True`` when the final audit's verdict is in the
    cycle's "success" set (typically ``{"success"}``); ``False`` for any
    abandon / max-rounds-exhausted termination.

    ``rounds_used`` counts every writer attempt (including those whose
    output_check failed and the auditor was skipped).

    ``history`` is the complete trail: one
    ``(WriterOutput, AuditReport)`` pair per round, including synthetic
    audits constructed by the cycle when the writer raised or
    output_check failed.

    ``reason`` is a short string explaining termination, e.g.
    ``"verdict=success"``, ``"verdict=abandon"``,
    ``"max_rounds_exhausted"``.
    """

    success: bool
    rounds_used: int
    final_writer_output: WriterOutput | None
    final_audit: AuditReport | None
    history: list[tuple[WriterOutput, AuditReport]] = field(
        default_factory=list[tuple[WriterOutput, AuditReport]],
    )
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "rounds_used": self.rounds_used,
            "final_writer_output": (
                self.final_writer_output.to_dict()
                if self.final_writer_output is not None
                else None
            ),
            "final_audit": (
                self.final_audit.to_dict()
                if self.final_audit is not None
                else None
            ),
            "history": [
                {"writer": w.to_dict(), "audit": a.to_dict()}
                for w, a in self.history
            ],
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CycleOutput:
        fwo_raw: Any = data.get("final_writer_output")
        fa_raw: Any = data.get("final_audit")
        raw_history: list[Any] = data.get("history") or []
        return cls(
            success=bool(data["success"]),
            rounds_used=int(data["rounds_used"]),
            final_writer_output=(
                WriterOutput.from_dict(fwo_raw) if fwo_raw else None
            ),
            final_audit=(
                AuditReport.from_dict(fa_raw) if fa_raw else None
            ),
            history=[
                (
                    WriterOutput.from_dict(item["writer"]),
                    AuditReport.from_dict(item["audit"]),
                )
                for item in raw_history
            ],
            reason=data.get("reason", ""),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> CycleOutput:
        return cls.from_dict(json.loads(text))
