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

## Tool-call middleware contract

`GuardrailsMiddleware` and `ToolCallRepairMiddleware` are `before_tool_call`
middleware. AgentCore does not dispatch tools, so enforcement is the host's:

- After running the middleware chain, check `ctx.is_blocked`. When set, do not
  execute the tool — return `ctx.block_reason` to the model as the tool result.
  The reserved metadata keys are `protocols.BLOCKED_KEY` and
  `protocols.BLOCK_REASON_KEY`; set them through `ctx.block(reason)`.
- Call `GuardrailsMiddleware.cleanup_task(task_id)` when a task ends. The
  middleware keeps per-task fingerprint, search-count, and loop-hint state that
  is only released there.
- `ToolCallRepairMiddleware` strips surrounding whitespace from string
  arguments, except keys in `LITERAL_CONTENT_KEYS` (file contents, str-replace
  needles). Hosts whose tools carry literal text under other names must extend
  that set, or exact-match edits will be corrupted.
- Passing `key_aliases={}` or `type_coercions={}` disables the default table;
  omit the argument to inherit it.

## Completion-stop normalization

`agent_core.runtime.loop._runaway` detects a reply the output cap cut off by
matching `LLMResponse.finish_reason == "length"` exactly, and deliberately has
no token-count fallback once visible text is present — an explicit
`finish_reason` is the only evidence that can carry that case. Each transport
names it differently, so every client routes its stop signal through
`providers.finish_reason.normalize_finish_reason`:

| Transport | Raw signal | Normalized |
|---|---|---|
| OpenAI Chat | `finish_reason="length"` | `length` |
| Anthropic Messages | `stop_reason="max_tokens"` | `length` |
| OpenAI Responses | `status="incomplete"` + `incomplete_details.reason="max_output_tokens"` | `length` |

Only truncation markers are rewritten. `tool_use`, `end_turn`, and `stop` pass
through unchanged because hosts read them directly. A new transport that skips
this normalization silently disables truncation recovery for its protocol.
