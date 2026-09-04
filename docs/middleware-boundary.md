# Middleware boundary

AgentCore owns two composable interception layers and the portable middlewares
that sit in them.

**Phase/tool middleware** — `agent_core.components.middleware.base.MiddlewareChain`
runs the `ExecutionMiddleware` contract declared in `agent_core.protocols`
(before/after phase, before/after tool, on_error). After-hooks run in reverse
registration order, so a chain nests rather than merely sequences: the first
middleware registered is the outermost layer. Hosts resolve the chain through the
structural `PhaseMiddlewareChain` protocol.

**LLM middleware** — `agent_core.components.middleware.llm.base.LLMMiddlewareChain`
wraps an `LLMClient` through `LLMProxy`. Same nesting rule.

Portable middlewares shipped here:

| Module | What it does |
|---|---|
| `llm.retry` | Exponential back-off with jitter for transient failures, clamped by `backoff_max`. Returns `True` from `on_llm_error` so the proxy re-issues. |
| `llm.tracing` | Duration, message counts and usage per call into an injected trace sink. Surfaces a fallback chain's `fallback_used` / `model_actually_used` markers. |
| `llm.token_accounting` | Per-task token totals, cost recording, SSE emission, budget charging. |
| `llm.loop_detection` | Fingerprints recent tool calls and injects a strategy-switch hint on repeats, with state isolated by task/session, role and phase. |
| `llm.output_repair` | Rewrites malformed reasoning markup, preserving every other `LLMResponse` field. |
| `llm.compaction` | Caller-invoked rolling-summary helper for a history that outgrew its budget. |
| `llm.api_key_rotation` | Mid-stream key rotation. Scaffolded and deliberately inert: `_rotate_client_credentials` raises `NotImplementedError`. |
| `rate_limit` | Concurrent-safe token-bucket RPM/TPM limiter that estimates before the call and corrects from actual usage after. |
| `tool_audit` | Pattern-based defense-in-depth classification of `bash` / `web_fetch` arguments; can veto a call and accepts a host-owned bash classifier. |
| `status_report` | Sub-agent phase heartbeat to a parent over the agent bus; host result fields come from an optional summarizer callback. |
| `todo` | Injects compact task progress through the active working memory's polymorphic `one_line_summary()`. |

## What the product owns

**The composition root.** Nothing here decides which middlewares run, in what
order, or with what parameters. A host builds its chains and registers them.
AgentCore ships no default chain, because ordering is a product decision with
observable consequences — `output_repair` last means it is innermost in the
reverse-order after-pass, which is what lets it see the response every other
layer already annotated.

**Durable cost persistence.** `TokenAccountingMiddleware` takes two Protocols
from `agent_core.protocols`, and the split between them is the boundary:

- `CostSink.record` runs per LLM call, synchronously, on the hot path. Its
  `get_summary(task_id)` supplies the final mapping when persistence is enabled;
  it may still live entirely in memory.
- `CostPersister.persist` runs once at completion and is the only path that
  reaches a database. The schema, the column names and the transaction boundary
  are host concerns; `summary` is forwarded from the host's own `get_summary`
  without core inspecting its shape, so a host can evolve it without a core
  release.

Both default to `None`, which makes `persist_cost` a no-op — the right behavior
for a stateless SDK path, not an error. Supplying a persister with a sink that
does not satisfy the full `CostSink` contract fails at construction time rather
than silently dropping final persistence. A raising persister is swallowed at
debug level: accounting is observability, and a database that is down must not
fail the task whose cost it describes.

**Trace and event sinks.** `llm.tracing` takes its sink duck-typed; a host that
registers nothing gets a no-op rather than an error.

**Phase-level infra middlewares that need host services.** Anything that
resolves a task-context store, an event store or a process manager stays in the
product. That is why the product's own `builtins` module is not here: it reaches
for a `TaskContextStore` through the registry and hardcodes a domain vocabulary
when summarising a phase result.

The same rule applies to result and progress vocabularies. Core's status reporter
accepts a host-owned `result_summarizer`, and Todo calls the working-memory
object's `one_line_summary()` method. A research host may report evidence and
assertions through those seams without AgentCore naming either field.

**Shell enforcement.** The built-in ToolAudit patterns catch common catastrophic
forms and now normalize recursive/force `rm` flags, but regexes cannot interpret
the full shell language or know a host's writable roots. Treat them as
defense-in-depth. Hosts that expose a shell should inject their authoritative
classifier and enforce sandbox/filesystem policy at the tool boundary as well.

## Compaction prompts

`llm.compaction` uses `agent_core.runtime.loop.summary_prompt`, which offers
`RESEARCH_COMPACTION_PROMPT`, `HANDOFF_COMPACTION_PROMPT` and a
`compaction_prompt()` selector. A host with a tool-category callback gets
conversation-aware selection; a host with neither gets the research prompt, which
is what `COMPACTION_PROMPT` aliases. Products should not carry private forks of
this prompt — a fork silently stops receiving improvements while still looking
like the shared one.

## Known rough edge, carried across unchanged

`TokenAccountingMiddleware._record_usage_aggregator` calls `record_llm_call`
through a three-level cascading `TypeError` fallback across three different kwarg
signatures. It is tolerant of aggregator builds that predate `provider` / `scene`
/ `cache_write_tokens`. Collapsing it is a behavior change and needs its own
review; it is documented here so nobody mistakes it for an accident.
