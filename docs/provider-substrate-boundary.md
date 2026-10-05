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

## Current Claude models and refusal telemetry

Native profiles for `claude-fable-5-1` and `claude-opus-5-5` use
`thinking_type: adaptive` and default to `thinking_display: summarized`.
These models also think adaptively when `thinking` is omitted; the direct
client forwards `effort` in that case. A direct `AnthropicClient` built with
`effort` but no `thinking` now sends `output_config.effort`; models without
effort support reject it, so leave `effort` empty for them. Manual budgets and disabled thinking
are unsupported by these models. Sampling parameters and forced tool choice
are omitted. Model selection and output-token budgets remain host-owned.

An empty thinking string can still carry a signature. The adapter preserves
thinking, redacted thinking, and intervening tool calls in provider order;
otherwise later signatures can become invalid when replayed. Canonical tool
calls remain authoritative after runtime filtering or repair.

These models bind thinking signatures to the preceding system prompt, tools,
and conversation. If a 400 reports an invalid thinking signature (for example
a signature bound to a different conversation after client-side compaction),
the adapter retries once without historical thinking or redacted-thinking blocks.
It logs the recovery and reports `thinking_history_reset` in response metadata
(carried by `StreamDelta.thinking_history_reset` when streaming). After success,
the agent loop removes invalid historical thinking before storing the new
response. Direct client consumers must apply that reset to their own history;
the client does not mutate caller-owned messages. A consumer that ignores it
keeps replaying the stale signatures, so every later request pays a rejected
call plus the retry. Other 400s and a failed retry propagate. Hosts should keep conversation prefixes stable to preserve reasoning.

See Anthropic's [Fable 5.1 migration guide](https://platform.claude.com/docs/en/models/fable-5-1/migration-guide),
[Opus 5.5 migration guide](https://platform.claude.com/docs/en/models/opus-5-5/migration-guide),
and [preserved-thinking contract](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking).

Refusals can contain partial text; they are successful HTTP responses, not
transport exceptions. Both ordinary and streamed responses retain the raw
`stop_reason` and structured `stop_details` in `LLMResponse.response_metadata`.
Unknown detail fields/categories pass through, including with older SDKs.
Observers receive `TurnContext.finish_reason` and `TurnContext.stop_details`;
the trajectory observer writes them to JSON snapshots and JSONL events.
This telemetry does not change loop termination or choose a fallback model.
