"""Generic ``WorkingMemory`` — structured per-loop recording shared by every workflow.

Records tool calls, key findings, and skill activations during a ReAct
loop. Persists snapshots to ``EventStore`` every N turns for crash
recovery, and provides structured markdown for compaction middleware
to inject as a lossless context summary.

Three responsibilities:

1. **Record** — accumulate tool calls, findings, and skills per turn.
2. **Persist** — serialize to ``EventStore`` every ``persist_interval``
   turns for crash recovery.
3. **Inject** — render structured markdown for auto-compact context
   injection (so a long-running loop doesn't lose its bearings after
   message-history compaction).

Layering
---------------------
This base class is workflow-agnostic. Domain extensions (research-side
``evidence_cards`` / ``assertions_draft``) live as a subclass at
``workflows/default_research/memory.py:ResearchWorkingMemory``. Subclasses
override ``serialize`` / ``from_snapshot`` / ``to_markdown`` to thread
their own fields through the persistence and injection paths.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agent_core.events import EventType

if TYPE_CHECKING:
    from agent_core.protocols import EventReader, EventSink

logger = logging.getLogger(__name__)

# ContextVar so middleware (compaction / token accounting / todo) can
# reach the active WorkingMemory without explicit threading.
current_working_memory: ContextVar[WorkingMemory | None] = ContextVar(
    "current_working_memory", default=None,
)

MAX_KEY_FINDINGS = 20
MAX_TOOL_CALLS_IN_MARKDOWN = 10


@dataclass
class ToolCallRecord:
    """Single tool invocation record."""

    tool_name: str
    tool_args_preview: str  # truncated args for display
    result_preview: str  # truncated result
    turn: int
    duration_ms: int = 0
    evidence_count: int = 0

    def serialize(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "tool_args_preview": self.tool_args_preview,
            "result_preview": self.result_preview,
            "turn": self.turn,
            "duration_ms": self.duration_ms,
            "evidence_count": self.evidence_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ToolCallRecord:
        return cls(
            tool_name=d.get("tool_name", ""),
            tool_args_preview=d.get("tool_args_preview", ""),
            result_preview=d.get("result_preview", ""),
            turn=d.get("turn", 0),
            duration_ms=d.get("duration_ms", 0),
            evidence_count=d.get("evidence_count", 0),
        )


@dataclass
class WorkingMemory:
    """Generic working memory for a single ReAct loop execution.

    Workflow-agnostic. Records tool calls, key findings, and skill
    activations; persists / recovers via ``EventStore`` snapshots.
    Subclasses extend with domain-specific fields (e.g.
    ``ResearchWorkingMemory.evidence_cards``).
    """

    task_id: str = ""
    tool_calls: list[ToolCallRecord] = field(
        default_factory=list[ToolCallRecord],
    )
    key_findings: list[str] = field(default_factory=list[str])
    loaded_skills: list[dict[str, Any]] = field(
        default_factory=list[dict[str, Any]],
    )
    search_count: int = 0
    iteration_count: int = 0
    last_persist_turn: int = 0
    persist_interval: int = 5

    # ── Recording ────────────────────────────────────────────────────

    def record_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        result: str,
        turn: int,
        duration_ms: int = 0,
        evidence_count: int = 0,
    ) -> None:
        self.tool_calls.append(
            ToolCallRecord(
                tool_name=tool_name,
                tool_args_preview=str(tool_args)[:150],
                result_preview=result[:200],
                turn=turn,
                duration_ms=duration_ms,
                evidence_count=evidence_count,
            )
        )
        if tool_name in ("web_search", "web_fetch"):
            self.search_count += 1

    def record_finding(self, text: str) -> None:
        self.key_findings.append(text)
        if len(self.key_findings) > MAX_KEY_FINDINGS:
            self.key_findings = self.key_findings[-MAX_KEY_FINDINGS:]

    def record_skill_loaded(self, skill_id: str, skill_name: str, turn: int) -> None:
        """Record a skill activation for compaction preservation."""
        if not any(s["skill_id"] == skill_id for s in self.loaded_skills):
            self.loaded_skills.append({
                "skill_id": skill_id,
                "skill_name": skill_name,
                "turn": turn,
            })

    # ── Persistence ──────────────────────────────────────────────────

    def should_persist(self) -> bool:
        return (self.iteration_count - self.last_persist_turn) >= self.persist_interval

    def serialize(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "tool_calls": [tc.serialize() for tc in self.tool_calls],
            "key_findings": self.key_findings,
            "loaded_skills": self.loaded_skills,
            "search_count": self.search_count,
            "iteration_count": self.iteration_count,
            "last_persist_turn": self.last_persist_turn,
            "persist_interval": self.persist_interval,
        }

    @classmethod
    def from_snapshot(cls, payload: dict[str, Any]) -> WorkingMemory:
        wm = cls(
            task_id=payload.get("task_id", ""),
            key_findings=payload.get("key_findings", []),
            loaded_skills=payload.get("loaded_skills", []),
            search_count=payload.get("search_count", 0),
            iteration_count=payload.get("iteration_count", 0),
            last_persist_turn=payload.get("last_persist_turn", 0),
            persist_interval=payload.get("persist_interval", 5),
        )
        for tc_dict in payload.get("tool_calls", []):
            wm.tool_calls.append(ToolCallRecord.from_dict(tc_dict))
        return wm

    async def persist(self, event_store: EventSink) -> None:
        """Persist current state to EventStore as a snapshot event."""
        await event_store.append(
            task_id=self.task_id,
            event_type=EventType.WORKING_MEMORY_SNAPSHOT,
            payload=self.serialize(),
            agent_role="system",
        )
        self.last_persist_turn = self.iteration_count
        logger.info(
            "WorkingMemory persisted for task %s at turn %d (%d tools)",
            self.task_id, self.iteration_count, len(self.tool_calls),
        )

    @classmethod
    async def recover(
        cls, event_store: EventReader, task_id: str,
    ) -> WorkingMemory | None:
        """Try to recover from the latest EventStore snapshot.

        Returns ``None`` if no snapshot exists. Subclasses inherit this
        unchanged — ``cls.from_snapshot`` dispatches to the subclass
        implementation, so a ``ResearchWorkingMemory.recover(...)`` call
        rebuilds research-specific fields automatically.
        """
        events = await event_store.get_events(
            task_id=task_id,
            event_type=EventType.WORKING_MEMORY_SNAPSHOT,
        )
        if not events:
            return None
        latest = events[-1]
        payload: dict[str, Any] = (
            latest.payload if hasattr(latest, "payload") else {}
        )
        wm = cls.from_snapshot(payload)
        logger.info(
            "WorkingMemory recovered for task %s from turn %d",
            task_id, wm.iteration_count,
        )
        return wm

    # ── Summaries ────────────────────────────────────────────────────

    def one_line_summary(self) -> str:
        return (
            f"{len(self.tool_calls)} tools, "
            f"{self.search_count} searches, "
            f"turn {self.iteration_count}"
        )

    def to_markdown(self) -> str:
        """Generic structured markdown for compaction context injection.

        Renders findings + active skills + a tail of the tool call log.
        Subclasses override to add domain sections (e.g. Evidence).
        """
        parts: list[str] = []

        if self.key_findings:
            parts.append("## Key Findings")
            for f in self.key_findings:
                parts.append(f"- {f}")

        if self.loaded_skills:
            parts.append("\n## Active Skills")
            for sk in self.loaded_skills:
                parts.append(
                    f"- **{sk['skill_name']}** (id={sk['skill_id']}, "
                    f"loaded at turn {sk['turn']})"
                )

        if self.tool_calls:
            parts.append(f"\n## Tool Call Log ({len(self.tool_calls)} total)")
            for tc in self.tool_calls[-MAX_TOOL_CALLS_IN_MARKDOWN:]:
                parts.append(
                    f"- T{tc.turn}: {tc.tool_name} → {tc.result_preview[:80]}"
                )

        return "\n".join(parts)
