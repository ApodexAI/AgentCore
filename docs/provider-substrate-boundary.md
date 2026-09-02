# Provider substrate boundary

AgentCore owns the provider-neutral physical transports and wrappers used by
all host products:

- OpenAI-compatible Chat Completions, including streamed usage, malformed tool
  name repair, reasoning-effort fallback, and request-override handling;
- OpenAI Responses with encrypted reasoning replay;
- Anthropic Messages and Bedrock, including signed thinking replay;
- fallback chains, prompt-cache decoration, protocol selection, and
  non-blocking diagnostic streams.

Hosts continue to own provider catalogs, credentials, endpoint selection,
deployment-specific headers, billing meters, traces, and UI-facing provider
metadata. Session affinity is supplied to `OpenAIClient` through a
`SessionQueryResolver`; AgentCore never interprets a host header by itself.

The SDK response boundary is intentionally dynamic. Third-party OpenAI and
Anthropic response classes vary by SDK and compatible gateway, so those files
use basic Pyright checking locally while the public constructors, AgentCore
messages, and the rest of the runtime remain strict.

Task pause checks follow the same rule: AgentCore owns safe polling semantics,
while a host injects the task-status loader and its missing-task exception.
