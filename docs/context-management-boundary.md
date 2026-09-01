# Context-management boundary

AgentCore owns the portable lifecycle of prompt context:

- token estimation and calibration against provider-reported input tokens;
- projection of the next request before it crosses the context ceiling;
- deterministic tool-result compression and LLM-backed summaries;
- tier selection, bounded retry/fallback, pairing invariants, and events;
- content-addressed spill persistence, bounded previews, recovery manifests,
  aggregate result budgets, session isolation, and cleanup.

Products configure policy rather than copying the implementation. They supply:

- the summary LLM and optional prompt selector;
- the context window, trigger ratio, relief target, and protected tool names;
- the physical spill root and the path visible inside their sandbox;
- the session identifier and lifecycle point at which cleanup runs;
- sandbox read-only mounts and write/path authorization.

`SpillStore` deliberately does not inspect a product execution scope or tool
registry. A product constructs one store per conversation and may pass it
directly to `TieredCompactor`. Tool adapters can also call `overflow()` and
`enforce_aggregate_budget()` on the same instance, so individual overflow,
per-turn aggregate trimming, and later compaction all point to one recovery
store.

The physical store should be outside every agent-writable workspace. Products
that expose it inside a sandbox must mount it read-only. A backend that cannot
name host paths passes no `visible_root`; AgentCore may retain a host-side copy
for diagnostics, but generated previews do not advertise an unusable path.

Checkpoint persistence and durable user memory are not context compaction.
They remain product concerns; AgentCore only preserves spill references in
messages so a product checkpoint can round-trip them.
