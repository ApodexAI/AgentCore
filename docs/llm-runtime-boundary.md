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
