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

## What a host owes the event store

Two distinct obligations sit behind one word ("event store"), and conflating
them is the easiest way to wire this package up wrong:

`protocols.EventSink` — telemetry.
: Flat `append(task_id, event_type, payload, agent_role)`. This is what
  `AgentBus(event_sink=...)` takes, and what the session lifecycle events
  (`session_task_submitted` / `session_task_completed`) and the submit/abort
  trace events go through. A host that implements only this is a complete,
  valid telemetry sink.

`agent_comm._AgentCommEventStore` — durable messaging.
: `append(event: KernelEvent) -> KernelEvent` plus `protocols.EventReader`.
  `AgentComm` needs the whole event because an inter-agent message carries
  `from_agent` / `to_agent` / `message_type` / `correlation_id` as
  first-class fields that the reader queries on — the flat `EventSink`
  signature cannot express them. An `EventSink` is therefore **not** an
  AgentComm store, and passing one produces a garbage record rather than a
  message.

The messaging store additionally owes a **total-order ordinal** on every
retained event: `KernelEvent.seq`, a monotonically increasing integer, or a
decimal `KernelEvent.id` for stores whose integer primary key already serves as
the id. `EventReader`'s `after_id` cursor is expressed in that ordinal, and it —
not `timestamp` — defines the order. `EventId` is deliberately an opaque
string, so `types.new_event_id()` (uuid4 hex: neither numeric nor monotonic)
cannot order a cursor; a store that stamps only that fails
`AgentComm.consume` with `EventStoreContractError`, and `AgentBus` logs the
lost durable-recovery capability at ERROR while still degrading to the
in-memory path. `tests/test_agent_comm_store_contract.py` is the executable
form of this section.

## Wall-time keys are independent, not alternatives

`resolve_research_wall` reads `research_wall_time_s` (a research-only budget,
not a ceiling — the reporter runs outside it by design) and `wall_deadline_s` /
`AGENT_CORE_TASK_WALL_TIME_S` (total-task ceilings, reported as
`ResearchWall.hard_total_s`) independently, and selects on **value validity**
rather than key presence. A profile may carry both; the ceiling stays enforced
either way, and an empty or non-numeric `research_wall_time_s` does not discard
the ceiling beside it.
