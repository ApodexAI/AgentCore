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

## Trajectory snapshots and events

`TrajectoryFileObserver` writes two complementary formats. The JSON envelope's
`messages` follows the loop's current history, including engine-generated tool
replies, post-processed result bodies, injections, rollbacks and compaction.
It preserves the normalizer's reasoning policy and native content blocks rather
than reconstructing assistant messages from the executable subset of calls.
Diagnostic fields (`usage`, `thinking_blocks`, `finish_reason`, `stop_details`)
remain annotations on the corresponding assistant message. After compaction
or a thinking reset substitutes a shallow copy, diagnostics move to the copy
only when it shares mutable content/tool-call containers or blocks with exactly
one vanished assistant and no other copy claims that assistant. Scalar object
identity is not evidence: separate JSON responses can share cached strings.
Ambiguous, deep-copied and plain-text-only copies without shared mutable
structure keep their diagnostics in JSONL when enabled. Source deduplication
uses entry identity and never compares message dictionaries by value, so cyclic
host metadata can reach the rendering fallback safely.
Messages without diagnostics, including resumed assistants, also count as
possible sources. `thinking_blocks` is
omitted while the same blocks are still native assistant content. Values that
are not JSON (datetimes, bytes, SDK objects) are written in a lossy string form
rather than failing the snapshot. Images are redacted
and tool text obeys `SWARM_TRAJECTORY_BODY_MAX_CHARS` (default 16384; 0 disables
the cap), so the snapshot is a trace representation rather than a byte-exact
provider payload. Transient per-request addenda and provider middleware edits
are outside this durable-history snapshot.

JSON snapshots are coalesced during a run (LLM input and response, each tool
result and turn end count toward `SWARM_TRAJECTORY_COALESCE_N`) and atomically
replaced. Routine snapshots compare the spool prefix by reference, which
catches replaced messages and values without serializing history. Edits nested
inside a kept container are found by comparing redacted rendered lines, a
full pass that runs after a compactor run, a rollback or context-limit notice,
and on completion or cancellation. Completion and cancellation force a snapshot of the latest
history, and a cancellation that arrives after completion leaves the terminal
snapshot untouched. Hosts that drive the hooks without loop history still get
the assistant and tool entries built from the hook arguments. JSONL is the event
log: it retains original tool outputs before post-processing, and events whose
messages are later pruned from history. Trajectory hooks are awaited so an
immediate synthetic result cannot overtake its assistant event. Consumers that
need the execution events should read JSONL; consumers that need the current
conversation should read JSON and handle native block-list content.

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

## Image attachments: the core decides visibility, the product supplies pixels

A tool returns images by returning the `agent_core.tool_content.tool_content`
envelope — a plain JSON dict — instead of a string. Sandbox-native tools run in
a child process and their return value crosses `json.dumps(default=str)` on the
way back, so the envelope is a dict by requirement, not by taste; a dataclass
arrives as its repr.

The split of responsibility:

- **The product** decides what is worth attaching, and downscales. Pixel count
  is the cost driver (roughly 1 token per 1024 px against the model this was
  calibrated on — a 1080p screenshot ~2.4K tokens, a 4K one ~8.5K), and
  AgentCore has no imaging dependency with which to resize. The caps in
  `tool_content` (6 MB and 8 images per result) are a backstop against a
  runaway producer, not a resize policy.
- **AgentCore** decides whether an attachment is shown, from
  `ModelProfile.supports_images` and `ModelProfile.protocol`, and how many stay
  in history, from `HistoryPolicy.max_images_in_history`. Products do not
  pre-filter on capability: `attach_images` is called unconditionally and writes
  the withheld note itself. Before that decision, core verifies that the decoded
  bytes have a supported image header matching the declared MIME type and a
  complete lightweight container shape (including PNG chunk CRCs). It also
  derives dimensions from those bytes rather than trusting producer metadata;
  the dimensions drive context-budget accounting and cannot be allowed to
  understate the actual image.

`ToolResult.images` is populated by core (`tool_exec` parses the envelope) and
consumed by core (`agent_loop` builds the message). It is not a host-supplied
field, so unlike `result_id` and the repeated-invocation metadata above it needs
no boundary exemption.

### Why every disappearance is narrated

Both the capability path and the eviction path replace the image with a sentence
saying an image was there and is not visible. The calibration run behind this
feature (MiroHarness
`internal-docs/designs/2026-09-08-native-image-in-tool-result-calibration.md`)
removed the image block from an otherwise working request and changed nothing
else. The model had been told to answer `NO_IMAGE` if it could not see
an image; it instead produced a four-digit code and three shapes, every one of
them invented. A tool result whose text reads as though an image were delivered
will be answered as though one were, so the text has to say otherwise. A product
that adds its own image-bearing path — or its own compactor that touches these
messages — inherits that obligation.

For the same reason `messages.text_of` renders an `image_url` block as a
placeholder rather than as nothing: every flattening caller (Anthropic message
translation, compaction summaries, trajectory lines) is dropping the image at
that call, and a transcript that never mentions the picture is the same trap in
a different place.
