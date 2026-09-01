"""BaseTaskState — universal state fields shared by all pipeline types.

Every pipeline's state TypedDict should include these fields.
Scenario-specific pipelines extend this with their own fields.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class BaseTaskState(TypedDict):
    """Universal state fields for any AgentCore pipeline."""

    # ── Identity ──────────────────────────────────────────────────────
    task_id: str
    language: str  # "auto" | "en" | "zh" — resolved early in pipeline

    # ── Input ───────────────────────────────────────────────────────────
    original_question: str

    # ── Control ─────────────────────────────────────────────────────────
    current_phase: str
    iteration_count: int

    # ── Output (generic) ────────────────────────────────────────────────
    output: dict[str, Any] | None

    # ── Logging & Communication ─────────────────────────────────────────
    errors: Annotated[list[str], operator.add]
    messages: Annotated[list[dict[str, Any]], operator.add]
    agent_messages: Annotated[list[dict[str, Any]], operator.add]
