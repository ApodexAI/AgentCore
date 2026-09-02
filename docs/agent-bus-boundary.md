# Agent-bus and scheduling boundary

AgentCore owns the product-neutral multi-agent scheduling state machine:

- typed inter-agent messages and cursored durable delivery;
- spawn depth, concurrency, token, and wall-time guards;
- parallel jobs and reusable serial sessions;
- cancellation, queue draining, result recovery, and idempotent fan-in;
- completion classification and report formatting;
- role-scoped tools/LLMs and fail-closed tool permissions;
- DAG lifecycle scheduling through structural backend protocols.

The implementation accepts product decisions through constructor inputs,
runtime specs, callbacks, and protocols. In particular, products inject the
event store, process manager, pause-check factory, per-role LLM factory,
workflow defaults, observers, result adapters, and execution context setup.

Products continue to own tool assembly, database schemas and persistence,
workflow-specific evidence harvesting/completion policy, UI rendering,
sandbox allocation, endpoint configuration, and session-retention policy.
Neither product package may be imported by AgentCore.

The two legacy wall-time environment prefixes remain accepted after the
portable `AGENT_CORE_TASK_WALL_TIME_S` name so deployments can migrate without
a flag day. Product workflow defaults are registered explicitly rather than
discovered by importing a `workflows` package from the shared kernel.
