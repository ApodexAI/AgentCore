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
deployment-specific headers, billing policy/sinks, traces, and UI-facing provider
metadata. Session affinity is supplied to `OpenAIClient` through a
`SessionQueryResolver`; AgentCore never interprets a host header by itself.

AgentCore additionally owns the product-neutral mechanics for profile-defined
auxiliary clients, raw-HTTP summary execution, cooldown fallback, and task-local
usage accumulation. Hosts inject provider-type lookup, concrete constructors,
session headers, decorators, candidate configuration, and billing/trace sinks.
The shared meter records quantities only; it does not assign prices or decide
which events are billable.

The SDK response boundary is intentionally dynamic. Third-party OpenAI and
Anthropic response classes vary by SDK and compatible gateway, so those files
use basic Pyright checking locally while the public constructors, AgentCore
messages, and the rest of the runtime remain strict.

Task pause checks follow the same rule: AgentCore owns safe polling semantics,
while a host injects the task-status loader and its missing-task exception.
