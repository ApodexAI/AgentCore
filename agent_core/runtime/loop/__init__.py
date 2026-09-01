"""Shared loop foundation primitives."""

from agent_core.runtime.loop.agent_loop import AgentLoopHooks, run_agent_loop
from agent_core.runtime.loop.compact import (
    DefaultCompactionPolicy,
    DefaultMessageCompactor,
)
from agent_core.runtime.loop.compact_llm import LLMSummaryCompactor
from agent_core.runtime.loop.llm_client import (
    RUNAWAY_STATE_KEY,
    TRUNCATION_CONTINUATION_GUIDANCE,
    LLMCallExhausted,
    LLMDeadlineExceeded,
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
from agent_core.runtime.loop.model_profile import (
    DefaultThinkingParser,
    HistoryPolicy,
    ModelProfile,
    NativeMessageNormalizer,
    configure_model_registry,
)
from agent_core.runtime.loop.tiered_compact import (
    InputTokenGauge,
    InputTokenThresholdPolicy,
    TieredCompactor,
    compaction_trigger_tokens,
)
from agent_core.runtime.loop.tool_call_parser import (
    DefaultToolCallParser,
    MultiFormatToolCallParser,
)
from agent_core.runtime.loop.tool_exec import (
    DefaultToolResultPostProcessor,
    ToolExecutionHooks,
    execute_tools,
)

__all__ = [
    "RUNAWAY_STATE_KEY",
    "TRUNCATION_CONTINUATION_GUIDANCE",
    "AgentLoopHooks",
    "DefaultCompactionPolicy",
    "DefaultMessageCompactor",
    "DefaultThinkingParser",
    "DefaultToolCallParser",
    "DefaultToolResultPostProcessor",
    "HistoryPolicy",
    "InputTokenGauge",
    "InputTokenThresholdPolicy",
    "LLMCallExhausted",
    "LLMDeadlineExceeded",
    "LLMReasoningRunaway",
    "LLMStreamStalled",
    "LLMSummaryCompactor",
    "MessageTrimmer",
    "ModelProfile",
    "MultiFormatToolCallParser",
    "NativeMessageNormalizer",
    "NullTrimmer",
    "TaskBoundaryTrimmer",
    "ThinkTagSplitter",
    "TieredCompactor",
    "ToolExecutionHooks",
    "bind_max_tokens",
    "bind_session_id",
    "bind_temperature",
    "bind_tools",
    "call_llm",
    "compaction_trigger_tokens",
    "configure_model_registry",
    "execute_tools",
    "extract_final_content",
    "extract_leaked_reasoning",
    "extract_model_name",
    "extract_usage",
    "is_truncated_with_text",
    "run_agent_loop",
]
