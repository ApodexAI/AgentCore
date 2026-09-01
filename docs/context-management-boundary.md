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
for diagnostics, but generated previews do not advertise an unusable path. Such
a store recovers nothing the model can read, so tiered compaction treats it as
if no spill were configured: results from the latest tool-call turn — which the
model has not seen when compaction runs — are kept verbatim rather than
shortened against a path nobody can open.

AgentCore also owns a separate run-local durable-journal primitive: append-only
entries, scope isolation, attachment integrity, note events, and deterministic
projections. This makes runtime history semantics identical across products;
it does not choose where a product stores a user's sessions or how long they
live. See `durable-context-boundary.md`.

Products still own user/session retention, UI/display history, checkpoint-to-
session association, physical context roots, sandbox access, TTL, and deletion
policy. AgentCore preserves spill and blob references so those product layers
can round-trip them without copying persistence mechanics.
