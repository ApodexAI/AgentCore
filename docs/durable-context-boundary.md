# Durable-context boundary

AgentCore owns the product-neutral, run-local persistence contract:

- append-only journal entries ordered within an explicit scope;
- SQLite durability, transactions, and scope isolation;
- content-addressed file blobs with atomic writes and integrity checks;
- reference-aware cleanup, deterministic projections, and event-sourced notes;
- async protocols that allow another journal/blob backend without changing
  runtime consumers.

A scope is an explicit `(session_id, run_id, agent_id)` value. AgentCore never
derives it from environment variables, a UI session, a sandbox, or a product
checkpoint. A product composition root supplies the scope and physical root,
then may associate journal sequence numbers with its own resume checkpoints.

Products own:

- the physical root and its lifecycle;
- session/run/agent scope allocation;
- resume-checkpoint association;
- sandbox mounts and access authorization;
- UI/display-history projections;
- user-level retention, TTL, export, and deletion timing.

This split means both products use the same durable history semantics without
turning AgentCore into a user database or deployment-policy layer.
