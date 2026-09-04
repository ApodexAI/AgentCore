"""StatusReportMiddleware — phase-level heartbeat for sub-agents.

Issue #24 Phase D: sub-agents report progress at phase boundaries
(not per-LLM-call, to avoid event flooding in react_solve's 50+ turns).

Only fires when execution_context contains parent_agent_id,
indicating this is a sub-agent spawned by AgentBus.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from typing import Any

from agent_core.protocols import ExecutionMiddleware, PhaseContext

logger = logging.getLogger(__name__)

PhaseResultSummarizer = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class StatusReportMiddleware(ExecutionMiddleware):
    """Emits phase-level status reports from sub-agents to their parent.

    Uses AgentComm with QUEUE delivery (parent polls when ready).
    Only active for sub-agent tasks (parent_agent_id in execution_context).
    """

    def __init__(
        self,
        result_summarizer: PhaseResultSummarizer | None = None,
    ) -> None:
        """Create a status reporter.

        ``result_summarizer`` is host-owned: it may add domain-specific fields
        to the message content without teaching AgentCore about a workflow's
        result schema.
        """
        self._result_summarizer = result_summarizer

    async def before_phase(self, ctx: PhaseContext) -> PhaseContext:
        """Record phase start time."""
        ctx.metadata["_status_phase_start"] = time.monotonic()
        return ctx

    async def after_phase(
        self, ctx: PhaseContext, result: dict[str, Any],
    ) -> dict[str, Any]:
        """Send status report to parent agent after phase completion."""
        parent_id = ctx.metadata.get("parent_agent_id")
        if not parent_id:
            # Not a sub-agent — skip
            return result

        try:
            from agent_core.components.agent_bus.agent_comm import (
                AgentComm,
                DeliveryMode,
            )
            from agent_core.models.agent_message import AgentMessage
            from agent_core.runtime.registries import services as registry

            agent_comm = registry.get_optional(AgentComm)
            if agent_comm is None:
                return result

            start = ctx.metadata.get(
                "_status_phase_start", time.monotonic(),
            )
            duration_ms = int((time.monotonic() - start) * 1000)

            details: dict[str, Any] = {}
            if self._result_summarizer is not None:
                try:
                    details.update(self._result_summarizer(result))
                except Exception:
                    logger.debug(
                        "StatusReport result summarizer failed",
                        exc_info=True,
                    )

            report = AgentMessage(
                task_id=ctx.task_id,
                from_agent=ctx.role_id or "unknown",
                to_agent=parent_id,
                message_type="status_report",
                content={
                    **details,
                    "agent_id": ctx.role_id,
                    "task_id": ctx.task_id,
                    "phase": ctx.phase_id,
                    "status": "phase_completed",
                    "duration_ms": duration_ms,
                },
            )
            await agent_comm.send(report, mode=DeliveryMode.QUEUE)
            logger.debug(
                "StatusReport: %s completed phase '%s' → parent %s",
                ctx.role_id, ctx.phase_id, parent_id,
            )
        except Exception as e:
            # Never fail the pipeline for a status report
            logger.debug("StatusReportMiddleware failed: %s", e)

        return result


__all__ = ["PhaseResultSummarizer", "StatusReportMiddleware"]
