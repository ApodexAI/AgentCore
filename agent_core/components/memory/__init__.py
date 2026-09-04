"""Generic working memory primitives shared by every workflow.

The base ``WorkingMemory`` class lives here. Research-specific
extensions (``evidence_cards``, ``assertions_draft``,
``record_evidence``) live as a subclass at
``workflows/default_research/memory.py:ResearchWorkingMemory``.
"""

from agent_core.components.memory.working_memory import (
    MAX_KEY_FINDINGS,
    MAX_TOOL_CALLS_IN_MARKDOWN,
    ToolCallRecord,
    WorkingMemory,
    current_working_memory,
)

__all__ = [
    "MAX_KEY_FINDINGS",
    "MAX_TOOL_CALLS_IN_MARKDOWN",
    "ToolCallRecord",
    "WorkingMemory",
    "current_working_memory",
]
