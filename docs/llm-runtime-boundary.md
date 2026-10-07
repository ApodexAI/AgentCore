# LLM runtime boundary

This extraction moves one complete physical-call layer into AgentCore:

- client binding and per-call overrides;
- response content, usage, and model-name normalization;
- streaming assembly, first-chunk/stall watchdogs, and reasoning guards;
- retry, backoff, provider-fallback classification, and runaway recovery;
- the public `llm_client` facade.

The source was converged from ApodexHarness and FrontierAgentInternal as one
batch. Small pre-existing differences were resolved as compatible supersets:

- `bind_max_tokens` and `ThinkTagSplitter` remain public;
- response blocks accept both `text` and legacy `content` fields;
- HTTP status extraction is shared through the public retry classifier;
- the portable `AGENT_CORE_` environment prefix wins, followed by the two
  legacy product prefixes.

AgentCore does not import product execution context or provider adapters.
Products supply their remaining decisions through three callbacks:

- `call_llm(..., wall_deadline_remaining=...)` reads the active execution
  scope's remaining wall budget;
- `call_llm(..., chain_fallback_active=...)` reports whether another provider
  chain leg exists;
- `bind_session_id(..., sticky_session_enabled=...)` applies the product's
  session-affinity kill switch.

Products keep thin wrappers that inject these callbacks and re-export the
shared API. Provider clients consume `current_thinking_retry_override()` to
translate semantic retry intent into provider-specific request fields.

Model profiles, provider client construction, tool parsing/execution, and the
agent loop remain product-owned in this phase.

## The usage contract

`extract_usage` returns a `UsageMetadata` (`agent_core.loop_types`). Every
non-`None` return carries all nine normalized keys — `provider`, `model`,
`prompt_tokens`, `completion_tokens`, `cache_read_tokens`,
`cache_write_tokens`, `cached_tokens`, `cache_creation_tokens`,
`reasoning_tokens` — zero-filled when the provider reported nothing, whichever
of the three response shapes it parsed. That is a runtime invariant, not just a
declaration: a TypedDict validates nothing at import time, so
`tests/test_llm_runtime_extract_usage.py` pins each branch's key set against
`UsageMetadata.__required_keys__`. Consumers can index those nine directly.

The `usage` fields on `TurnContext` and `LLMAttemptContext` are deliberately
*wider* — `Mapping[str, Any] | None`, not the TypedDict. Products, not
AgentCore, construct those contexts, and the shapes they hand over are ones a
TypedDict rejects outright:

- partial literals in test doubles — `usage={"prompt_tokens": 182_000}`;
- alias-only shapes — `usage={"input_tokens": …, "output_tokens": …}`, which
  ApodexHarness' budget observer reads;
- `dict(event["usage"])` re-wraps out of a `dict[str, Any]` attempt event,
  which the ApodexHarness loop stamps `provider` / `model` onto before
  constructing `LLMAttemptContext`.

Neither `dict[str, int]` nor `dict[str, Any]` is assignable to a TypedDict, in
strict *or* standard mode, so narrowing these fields would break the product
loops the normalized contract exists to serve. `Mapping` rather than
`dict[str, Any]` because it is covariant in its value type: it accepts all of
the above *and* a `UsageMetadata`, which a `dict[str, Any]` field would reject.
A consumer wanting the precise shape annotates its own parameter
`UsageMetadata`; the boundary does not force that on producers. Hosts stamping
extra keys is likewise their business — a TypedDict cannot express "open" on
Python 3.12 (PEP 728 lands later). `UsageMetadataExtras` declares the aliases
hosts are known to add, for anyone who wants to describe such a mapping as a
`UsageMetadata`.

`tests/test_loop_types.py` asserts these fields stay an open mapping, so a
later well-meant tightening fails loudly.

## Who decides whether a turn streams

Streaming is opt-in and, by default, the RUN decides: it is selected when the
reasoning-only watchdog is configured (`reasoning_only_timeout_s` /
`reasoning_only_max_tokens` — the guard reads the stream) or when an observer
declares `wants_llm_delta = True`. Implementing `on_llm_delta` without that
attribute gets silence.

`LoopConfig.stream_llm_tokens` overrides that decision: `True` streams, `False`
does not, `None` (the default) keeps the automatic choice. An explicit value
exists because the TRANSPORT can be the reason rather than any observer — a
gateway that abandons a non-streaming request while waiting for its response
headers leaves no other option, and for a non-streaming Anthropic request those
headers arrive only once generation is complete. Measured on llm-hub's "Claude
Code" channel, 2026-10-05: an identical `claude-opus-5-5` request at
`effort=max` returned HTTP 502 `upstream_unreachable — first byte timeout` after
76s non-streaming, and HTTP 200 streamed, first byte at 3.9s and the generation
taking 242s. Concurrency was ruled out (five light requests in flight all
succeeded; five heavy ones failed at the same 76s mark).

`LoopConfig.stream_transport` is the explicit transport selector for new
callers. It takes precedence over the older `stream_llm_tokens` field, and a
streamed call can be drained into a final `LLMResponse` even when no delta
observer is registered. At the lower-level call boundary,
`call_llm(..., stream=True)` selects the same transport independently of
`on_delta`. Leaving it unset retains the existing automatic choice and older
call sites continue to work. The new `LoopConfig` field is keyword-only so
existing positional construction retains its argument order.

After a stream ends, native tool arguments are checked according to
`LoopConfig.tool_argument_validation`:

- `"structural"` (default): the arguments must decode to a JSON object that
  carries every top-level `required` property. A blank argument string counts
  as `{}`, which is how Anthropic streaming and several OpenAI-compatible
  servers encode a zero-argument call. Property types are left to the tool,
  so tools that coerce `"5"` to `5` behave as before.
- `"strict"`: the structural checks plus full JSON Schema validation.
- `"off"`: no new checks; only the legacy empty-required-arguments retry.

A tool schema that is not itself valid JSON Schema never fails a call:
validation is skipped for that tool with a warning.

An invalid streamed call triggers at most one streaming retry within the
original attempt's time budget. The discarded request keeps its own usage and
attempt record. Calls that matched the previous empty-required-arguments
condition keep the `stream_empty_tool_arguments` attempt reason (and
`stream_empty_args_replay` for a failed retry) and the `stream_empty_args_*`
response metadata keys; other invalid calls use `stream_invalid_tool_call`.
The retry's `recovery_action` is `retry_streaming`.

A retry that remains invalid is returned with `invalid_tool_calls` diagnostics
(raw arguments included) and the attempt is `accepted_degraded`. At execution
time, only provider-native calls are checked: an invalid one produces an
error tool result (`error_kind="invalid_arguments"`) without invoking the
tool, unless an observer rewrote its arguments. Text-mode calls are never
blocked by this check. A named native call without an id is given a generated
id before the assistant turn reaches history, so the tool reply always
matches. The stream assembler continues to discard nameless slots.

In automatic mode the choice is also gated by protocol:
`UNVERIFIED_STREAM_PROTOCOLS` (`responses`, `bedrock`) stays non-streaming
because no test here proves their streamed turn replays like its non-streaming
twin. `anthropic` is NOT in that set: its `stream` rebuilds the provider's
verbatim block list — including a thinking block's trailing `signature_delta`
and a `redacted_thinking` payload that arrives with no deltas — and
`test_anthropic_latest_models.py` exercises the signature round-trip, the
prefix-bound retry and the reset commit parametrized over streaming. An
explicit `stream_llm_tokens` ignores the gate; the host then owns that fidelity.

When automatic mode suppresses deltas something asked for, the loop logs a
warning. It used to be silent, which meant a profile configuring the
reasoning-only watchdog on a gated protocol got no watchdog and no sign of it.

## Completion classification and recovery

The same raw-response classifier is used for chat and assembled streams, before
history normalization or accepted-attempt events:

| Signal | Handling |
|---|---|
| Visible content, executable call, reasoning or opaque signed blocks | Preserve the response; existing tool/truncation/runaway policies apply. |
| Provider-reported usage, including reported zeros | Preserve it. Usage establishes a real response, not a useful answer. |
| `refusal`, `content_filter`, native refusal text/blocks or structured refusal details | Preserve the provider evidence and stop with `refusal` or `content_filter`, even under a nudge policy. Tools from that turn are not executed; recorded ids get synthetic replies for safe replay. |
| No content, calls, reasoning, reported usage or explicit rejection | Raise `LLMEmptyCompletion` and recover inside `call_llm`. |

`LLMResponse.usage_source` and `StreamDelta.usage_source` are optional provenance
channels. Adapters stamp `provider` for reported usage. A wrapper that invents
counts should stamp `estimated`, or retain an `estimated: true` marker on its
usage map. Estimates are retained as estimated in attempt accounting but cannot
turn a transport blank into a valid response. Unmarked legacy usage is treated
conservatively as reported; there is no reliable way to reconstruct provenance
once a wrapper discards it. The classifier also reads legacy `usage_metadata`
and raw `response_metadata.token_usage` / `usage` channels.

OpenAI-compatible gateways may send `refusal=""` as a placeholder beside text,
tool calls or reasoning (`reasoning_content` / `reasoning`). That placeholder is
not a decline when the turn carries output. Streaming inference waits for normal
EOF, including streams without `finish_reason`; errors and consumer cancellation
never turn a pending marker into a refusal. Protocol terminator validation runs
before this inference, so a truncated transport cannot become a completed decline.
Non-empty refusal text and explicit
provider rejection reasons retain their usual semantics.

### One recovery owner in both transports

`empty_completion_max_retries` defaults to two same-leg resamples. This allowance
is independent of `max_llm_retries`, which bounds generic failures and the
existing reasoning recovery. Resamples use the same retry backoff configuration,
conversation and bindings. A fresh logical call gets a fresh empty allowance;
resamples stay inside the current call and never spend additional loop turns.
They do not restart `logical_call_timeout_s`: admission, requests, backoff and
fallback all share the original deadline, with the run wall deadline taking
precedence when earlier.

Empty requests produce exactly one finished attempt event, with reason
`empty_completion` and outcome `discarded` when recovery continues, or `failed`
when it stops. They never enter assistant history or response observers. The
loop translates terminal empty exhaustion to `stopped_by="empty_completion"`;
deadline exhaustion keeps its own error/deadline reason. The opportunistic chat
replay used to repair streamed tool arguments also gets a separate attempt id
before its request starts. A blank/failed replay is recorded as failed while the
original streamed response remains deliverable; each request retains its own
latency and accounting, rather than charging replay time to the original.

### Fallback routing and wrappers

`LLMFallbackChain` allocates a private cursor for each logical call. After the
serving leg spends its empty allowance, it advances only if that leg's triggers
match: `any_error` or the optional `empty_completion` trigger. An explicit empty
trigger tuple is a barrier. Each subsequent leg gets the same allowance, bounded
by the finite chain and the shared deadline. An ordinary HTTP failure that
already selected a later leg pins empty resamples to that actual leg.

Cursors survive tool, temperature, token and session bindings, nested provider
stamping chains and middleware proxies. Cached clients are never moved to another
leg, so concurrent calls and the next turn start independently. Direct native
chain callers detect blanks inside the chain and apply their configured triggers.
Candidate-empty deltas (including nameless tool argument fragments) are buffered
until a real signal commits that leg, so discarded fragments cannot corrupt the
next leg's tool calls. Transport heartbeats still pass through immediately.
`CooldownFallbackLLM` similarly detects them before its existing retry/degrade
policy and preserves its cooldown and tracing events.

Transparent product wrappers opt into delayed-assembly recovery by explicitly
implementing `for_logical_call()` and `advance_empty_completion(error)`. The
first returns a client preserving the wrapper around fresh inner routing state;
the second returns whether the current call advanced. `LLMProxy` implements both
and keeps its middleware, role and shared atomic counter. A `__getattr__`
forwarder alone is not opt-in, because invoking the inner preparation method
could silently bypass the wrapper. Wrappers without these hooks still receive
empty classification and same-key recovery; products using an external chain
can inject `chain_fallback_active` to receive the terminal error for routing.


### Streamed middleware consumers

`LLMProxy.stream` supplies `after_llm` with terminal usage/provenance, model,
provider and rejection metadata, plus assembled **named native tool calls**.
The proxy and loop use the same indexed tool accumulator: names/arguments are
concatenated in provider order, nameless slots are dropped, and each retry
starts with fresh state. Heartbeats are never model output. The middleware
response remains a passive view; signed-block replay stays owned by the loop.

Token accounting charges only reported usage. `usage_source="estimated"` and
`estimated: true` maps remain visible in tracing/attempt diagnostics, but never
enter `CostSink`, task budget charges, billing events or `UsageAggregator`.
Unmarked legacy usage retains its reported interpretation. Reported cache reads
and writes use `cache_read_tokens` / `cache_write_tokens` first, including real
zeros, with legacy read/creation aliases as fallbacks. Cache-only reported calls
still reach cache-aware usage aggregation; the existing four-argument `CostSink`
contract and prompt/completion budget units remain unchanged.

Rate correction uses reported totals (or reported prompt/completion counts),
including zero. Missing, invalid or estimated usage leaves the admission
reservation in place. Corrections use the **capped reservation actually taken**,
not the original prompt estimate, and records are isolated by limiter instance
so multiple quota layers cannot overwrite one another's state. Request quota is
reserved independently of whether token usage is later available.

`LoopDetectionMiddleware` now receives native streamed tool calls, so completed
repeated calls can trigger the existing strategy-switch hint. Failed or
consumer-closed stream proposals do not enter loop history; authentic usage
from those requests is still eligible for accounting. Streaming cost/token
reports therefore increase from the previously omitted values. Consumers should
recheck budgets and alerts calibrated against the old undercount.
