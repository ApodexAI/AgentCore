"""Nudges an agent session toward emitting its final structured output.

The well-known failure mode this guards against: an agent (typically an
auditor or report writer) keeps probing tool calls until ``max_turns`` is
hit, then exits with an empty final content because it never bothered to
write the structured output its task requires. This is especially common
when the auditor has a non-trivial tool budget — it explores until the
loop kills it.

The observer fires once when the session crosses ``conclude_ratio`` of
its ``max_turns`` budget (default 0.8, i.e. last 20% of turns). The
injected message instructs the LLM to stop further investigation and
emit its final structured output now. Subsequent turns do not re-fire,
so the message is not repeated.

Independent of any specific cycle / orchestration: any session whose
terminal output is mandatory benefits — auditors, report writers,
summarizers, finalize-then-stop solvers.

Usage::

    observer = ConcludePhaseObserver(conclude_ratio=0.8)
    await bus.submit_task_to_session(
        session_id, prompt, observers=[observer, ...],
    )
"""
from __future__ import annotations

import logging

from agent_core.loop_types import (
    BaseObserver,
    Intervention,
    TurnContext,
)

logger = logging.getLogger(__name__)


_DEFAULT_MESSAGE = (
    "[conclude phase] You are nearing the turn budget for this task. "
    "Stop further investigation and emit your final structured output "
    "now (the JSON block your task requires). Use only the information "
    "you already have. Further tool calls will be cut off."
)


class ConcludePhaseObserver(BaseObserver):
    """Critical observer that nudges the agent to conclude before turn exhaustion.

    Parameters
    ----------
    conclude_ratio
        Fraction of ``max_turns`` at which the nudge fires. Must be in
        ``(0.0, 1.0]``. Default 0.8 means "fire once in the last 20% of
        the turn budget".
    message
        Override the injected message text. Default is a generic
        instruction to emit the final structured output. Override when
        the session's required output format is specific (e.g. "emit your
        AuditReport JSON now").
    """

    critical = True

    def __init__(
        self,
        conclude_ratio: float = 0.8,
        message: str | None = None,
    ) -> None:
        if not 0.0 < conclude_ratio <= 1.0:
            raise ValueError(
                "conclude_ratio must be in (0.0, 1.0], got "
                f"{conclude_ratio!r}",
            )
        self._ratio = conclude_ratio
        self._message = message if message is not None else _DEFAULT_MESSAGE
        self._fired = False

    async def on_turn_end(self, ctx: TurnContext) -> Intervention | None:
        if self._fired:
            return None
        if ctx.max_turns <= 0:
            return None
        progress = ctx.turn / ctx.max_turns
        if progress < self._ratio:
            return None
        self._fired = True
        logger.info(
            "ConcludePhaseObserver firing at turn=%d/%d (progress=%.2f, "
            "ratio=%.2f, role=%s)",
            ctx.turn,
            ctx.max_turns,
            progress,
            self._ratio,
            ctx.role_id,
        )
        return Intervention(inject_messages=[self._message])
