"""Provider transports and provider-neutral client wrappers."""

from agent_core.providers.anthropic import AnthropicClient
from agent_core.providers.fallback import FallbackEntry, LLMFallbackChain
from agent_core.providers.nonblocking_stream import NonBlockingStream, nonblocking_stderr
from agent_core.providers.openai_chat import (
    OpenAIClient,
    SessionQueryResolver,
    configure_session_query_resolver,
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

__all__ = [
    "AnthropicClient",
    "AnthropicPromptCacheAdapter",
    "FallbackEntry",
    "LLMFallbackChain",
    "NonBlockingStream",
    "OpenAIClient",
    "OpenAIResponsesClient",
    "SessionQueryResolver",
    "build_protocol_client",
    "configure_session_query_resolver",
    "maybe_wrap_for_prompt_cache",
    "nonblocking_stderr",
    "protocol_of",
    "provider_label",
    "thinking_format_for_protocol",
]
