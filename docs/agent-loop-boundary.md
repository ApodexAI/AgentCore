# Agent-loop boundary

AgentCore owns one complete logical-turn engine:

- thinking extraction and history normalization;
- native, JSON, MCP, Qwen, Seed and wrapped function-call parsing;
- parallel tool dispatch, interruption and cancellation hygiene;
- observer lifecycle, compaction, rollback, continuation and finalization;
- the composition of the shared physical LLM-call runtime across turns.

The products keep policy and state that cannot be portable. `AgentLoopHooks`
injects session affinity, wall-deadline lookup, provider-chain state, execution
scope storage, cancellation cleanup and spill-reference detection. Session
affinity has a single owner per run: `bind_session` fully replaces the built-in
binding, and therefore also replaces `sticky_session_enabled`, which is ignored
(with a warning) when both are supplied.
`ToolExecutionHooks` injects effective timeout calculation, wall-clamped waits,
per-call ContextVars and metering, result spilling/truncation and aggregate
result budgeting.

Model endpoint registries are also product data. A product compatibility facade
calls `configure_model_registry()` with its packaged YAML path; AgentCore has no
knowledge of either product package.

Mixed native function-call batches have one explicit parser decision. By
default, unknown companions are dropped when a known call is executable, while
unknown-only batches are retained for corrective tool errors. A host needing
the previous keep-all behavior constructs the parser with
`keep_unknown_native_companions=True`. In either mode, the loop answers any
dropped assistant `tool_call_id`, preventing an orphaned call from making the
next provider request invalid.

That invariant also covers observer interventions that keep the assistant
message but bypass execution -- `skip_tool_execution`, and
`continue_to_next_turn` without `pop_last_message`. Synthetic answers are
inserted directly beneath the assistant message, so an injected user message
never separates a call from its reply. Host hooks are advisory: a raising
`resolve_timeout`, `on_call`, `transform_batch` or result formatter is logged
and falls back to core behavior rather than failing the batch, and a fan-in
interrupt waiter that returns `False` or raises leaves the tool running instead
of discarding its collected work.
