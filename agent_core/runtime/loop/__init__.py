"""Shared loop foundation primitives."""

from agent_core.runtime.loop.compact import (
    DefaultCompactionPolicy,
    DefaultMessageCompactor,
)
from agent_core.runtime.loop.llm_client import (
    RUNAWAY_STATE_KEY,
    TRUNCATION_CONTINUATION_GUIDANCE,
    LLMCallExhausted,
    LLMReasoningRunaway,
    LLMStreamStalled,
    ThinkTagSplitter,
    bind_max_tokens,
    bind_session_id,
    bind_temperature,
    bind_tools,
    call_llm,
    extract_final_content,
    extract_leaked_reasoning,
    extract_model_name,
    extract_usage,
    is_truncated_with_text,
)
from agent_core.runtime.loop.message_trimmer import (
    MessageTrimmer,
    NullTrimmer,
    TaskBoundaryTrimmer,
)

__all__ = [
    "RUNAWAY_STATE_KEY",
    "TRUNCATION_CONTINUATION_GUIDANCE",
    "DefaultCompactionPolicy",
    "DefaultMessageCompactor",
    "LLMCallExhausted",
    "LLMReasoningRunaway",
    "LLMStreamStalled",
    "MessageTrimmer",
    "NullTrimmer",
    "TaskBoundaryTrimmer",
    "ThinkTagSplitter",
    "bind_max_tokens",
    "bind_session_id",
    "bind_temperature",
    "bind_tools",
    "call_llm",
    "extract_final_content",
    "extract_leaked_reasoning",
    "extract_model_name",
    "extract_usage",
    "is_truncated_with_text",
]
