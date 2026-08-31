"""Shared loop foundation primitives."""

from agent_core.runtime.loop.compact import (
    DefaultCompactionPolicy,
    DefaultMessageCompactor,
)
from agent_core.runtime.loop.message_trimmer import (
    MessageTrimmer,
    NullTrimmer,
    TaskBoundaryTrimmer,
)

__all__ = [
    "DefaultCompactionPolicy",
    "DefaultMessageCompactor",
    "MessageTrimmer",
    "NullTrimmer",
    "TaskBoundaryTrimmer",
]
