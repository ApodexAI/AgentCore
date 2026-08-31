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
