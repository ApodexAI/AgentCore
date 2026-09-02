"""Provider transports and provider-neutral client wrappers."""

from agent_core.providers.anthropic import AnthropicClient
from agent_core.providers.aux_builder import AuxLLMFactory
from agent_core.providers.fallback import (
    CooldownFallbackLLM,
    FallbackEntry,
    LLMFallbackChain,
    legacy_retryable,
)
from agent_core.providers.finish_reason import (
    FINISH_REASON_LENGTH,
    TRUNCATION_MARKERS,
    normalize_finish_reason,
    responses_finish_reason,
)
from agent_core.providers.nonblocking_stream import NonBlockingStream, nonblocking_stderr
from agent_core.providers.openai_chat import (
    OpenAIClient,
    SessionQueryResolver,
    SessionScopeResolver,
    configure_session_query_resolver,
    configure_session_scope_resolver,
)
from agent_core.providers.openai_responses import OpenAIResponsesClient
from agent_core.providers.prompt_cache import (
    AnthropicPromptCacheAdapter,
    maybe_wrap_for_prompt_cache,
)
from agent_core.providers.protocol_client import (
    build_protocol_client,
    protocol_of,
    provider_label,
    thinking_format_for_protocol,
)
from agent_core.providers.summary import (
    EXTRACT_INFO_PROMPT,
    SummaryCandidate,
    SummaryLLMEngine,
    build_summary_payload,
    default_summary_retryable,
    describe_summary_candidates,
    normalize_summary_endpoint,
    truncate_summary_fallback,
)

__all__ = [
    "EXTRACT_INFO_PROMPT",
    "FINISH_REASON_LENGTH",
    "TRUNCATION_MARKERS",
    "AnthropicClient",
    "AnthropicPromptCacheAdapter",
    "AuxLLMFactory",
    "CooldownFallbackLLM",
    "FallbackEntry",
    "LLMFallbackChain",
    "NonBlockingStream",
    "OpenAIClient",
    "OpenAIResponsesClient",
    "SessionQueryResolver",
    "SessionScopeResolver",
    "SummaryCandidate",
    "SummaryLLMEngine",
    "build_protocol_client",
    "build_summary_payload",
    "configure_session_query_resolver",
    "configure_session_scope_resolver",
    "default_summary_retryable",
    "describe_summary_candidates",
    "legacy_retryable",
    "maybe_wrap_for_prompt_cache",
    "nonblocking_stderr",
    "normalize_finish_reason",
    "normalize_summary_endpoint",
    "protocol_of",
    "provider_label",
    "responses_finish_reason",
    "thinking_format_for_protocol",
    "truncate_summary_fallback",
]
