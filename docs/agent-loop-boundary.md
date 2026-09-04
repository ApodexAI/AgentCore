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

## Repeated-invocation metadata has no core-side consumer, by construction

`ToolResult.repeat_count`, `ToolResult.repeat_recovery_id`, `ToolResult.result_id`
and `ToolResult.error_kind` are
populated from `ToolExecutionHooks.result_metadata` and then read by **nothing
inside AgentCore**. That is deliberate — AgentCore owns the *facts* about a tool
result, while what to say to the model about them is prompt policy, which varies
per product and per profile — but it is load-bearing enough to state here rather
than leave to be rediscovered.

The reason it needs stating: a product that today appends its own note in its own
loop copy loses that note the moment it adopts `run_agent_loop`, and loses it
*silently*. Nothing raises, no test fails, the field is still there and still
correct — the sentence the model used to read is simply gone. Verify against your
own loop before switching, because a canary merge cannot see this class of gap.

`scripts/check_unconsumed_fields.py` enforces this in CI: a field on a watched
model with no attribute read anywhere in `agent_core/` must be named in one of
these boundary documents. The rule is not "document your fields" — it is that
deciding *not* to consume a field is a boundary decision, and an undocumented one
is indistinguishable from an oversight.

Two ways to close the gap were considered.

**A — AgentCore words the note.** Uniform across products, one place to fix the
wording. Costs a metadata contract change: the note cannot be worded correctly
from `repeat_count` alone. Whether a call *counts as* a repeat is per-tool policy
(a stateless retrieval query with the same arguments is the same query even when
the provider reshuffled its snippets; a `bash` re-read over mutated sandbox state
is a legitimate second call), while whether the body came back byte-identical is
a separate, always-observed fact. Collapsing the two makes the note assert
"identical output" for a body that differs — a falsehood the model can check
against its own history. So A requires an `identical_body`-equivalent alongside
`repeat_count`, and AgentCore would then own wording that products may need to
diverge on.

**B — the product words the note.** No contract change to the shape of the loop:
`AgentLoopHooks.render_tool_result` already receives the `ToolResult` and returns
the body that becomes the history message, which is exactly the right seam. This
keeps prompt wording with the layer that owns prompts. The one gap is the same
`identical_body` fact: `result_metadata` maps only the four known keys onto
`ToolResult` and drops everything else, so a product cannot currently carry that
bit from where it is observed to where it words the note without keeping its own
side table keyed by `tool_call_id`.

**B is the chosen route.** `ToolResult.host_metadata` carries whatever
`result_metadata` returned, verbatim, through to `render_tool_result`, so
`identical_body` and anything else a product observes reaches the place where the
product words its note. Two consequences worth stating:

- The pass-through is verbatim rather than "the keys AgentCore did not
  recognise". A residue rule would silently change what a product sees the day
  AgentCore adopts a new reserved key — the same quiet breakage this whole
  section is about. Reserved keys therefore appear both as typed fields and in
  `host_metadata`.
- AgentCore still words nothing about repeats. A product moving off its own loop
  copy must port its note into `render_tool_result`; nothing here will fail if it
  forgets, which is why this paragraph exists.
