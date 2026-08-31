# pyright: reportPrivateUsage=false, reportUnknownVariableType=false
"""Bind, invoke, and normalize LLM clients used by the agent loop.

Provider adaptation stays here so the loop remains a readable sequence of
turn-level operations.
"""

from __future__ import annotations

from agent_core.errors import (
    LLMCallExhausted as LLMCallExhausted,
)
from agent_core.errors import (
    LLMReasoningRunaway,
    LLMStreamStalled,
)
from agent_core.runtime.loop._bind import (
    _ensure_bound as _ensure_bound,
)
from agent_core.runtime.loop._bind import (
    bind_max_tokens,
    bind_session_id,
    bind_temperature,
    bind_tools,
)
from agent_core.runtime.loop._call import call_llm
from agent_core.runtime.loop._response import (
    extract_final_content,
    extract_leaked_reasoning,
    extract_model_name,
    extract_usage,
)
from agent_core.runtime.loop._runaway import (
    RUNAWAY_STATE_KEY,
    TRUNCATION_CONTINUATION_GUIDANCE,
    is_truncated_with_text,
)
from agent_core.runtime.loop._streaming import ThinkTagSplitter
from agent_core.tokens import (
    estimate_message_tokens,
    estimate_text_tokens,
)

__all__ = [
    "RUNAWAY_STATE_KEY",
    "TRUNCATION_CONTINUATION_GUIDANCE",
    "LLMCallExhausted",
    "LLMReasoningRunaway",
    "LLMStreamStalled",
    "ThinkTagSplitter",
    "bind_max_tokens",
    "bind_session_id",
    "bind_temperature",
    "bind_tools",
    "call_llm",
    "estimate_message_tokens",
    "estimate_text_tokens",
    "extract_final_content",
    "extract_leaked_reasoning",
    "extract_model_name",
    "extract_usage",
    "is_truncated_with_text",
]
