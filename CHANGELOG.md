# Changelog

Notable changes to `apodex-agent-core`, newest first. This file is the contract
between AgentCore and its consumers: every release entry must say what a product
has to know before upgrading. The release workflow reads the matching section as
the GitHub Release body, so a release with no entry here fails.

Versioning follows [docs/versioning.md](docs/versioning.md).

## [0.7.0] - 2026-09-04

### Changed

- Tier 1 deterministic compaction (`KeepLastNToolResultsCompactor`) no longer
  leaves a bare placeholder where it drops a tool body. It now appends a bounded
  card naming the call (tool name + a single-line, 120-char arguments preview)
  and up to three source URLs found in the discarded body, above the existing
  recovery pointer. Both fields already existed in the history and in the body,
  so this adds no LLM call, no storage, and no second model-visible recovery
  route. Measured cost is ~+220 characters per elided result, hard-capped at 400.

  Rationale: Tier 2's summary preserves arguments and URLs, but Tier 2 only fires
  when Tier 1 did not free enough, so on a Tier1-only turn the model lost exactly
  the two things it needs in order not to re-issue a query it already ran.

  A card that would not be shorter than the body it replaces is skipped and the
  body kept verbatim — but only when no spill callback is configured. With a
  recovery pointer the body is always replaced even if the card is longer: the
  pointer reaches the model only through `spill_refs` → the Tier 2 recovery
  index, and such a body is itself already an upstream-truncated preview, so
  keeping it would strand the spilled full text as unrecoverable.

- `KeepLastNToolResultsCompactor` now resolves an existing recovery handle with
  `spill_refs` taking precedence over `result_store_ref`. A ref pinned by an
  earlier compaction pass describes the content still on the message, while the
  loop-cap handle describes the pre-truncation body upstream shed; reading the
  latter first re-spilled a body that was already stored and pinned the wrong
  handle into the recovery index.

### Added

- `TieredCompactor` takes `manifest_max_paths` and `manifest_max_chars`, either
  of which may be `None` to remove that bound. The defaults are unchanged (20 /
  3,000) and are sized for a handle rendered as a filesystem path. A product
  whose handles are short content-addressed ids pays a fraction of that per entry
  and should raise or remove the cap: when a cap binds, the OLDEST handles are
  dropped, and a product measured decisive early evidence becoming unrecoverable
  after a long unrelated detour for exactly that reason. The cap is charged
  against rendered characters, which is the only quantity the two handle shapes
  share — cap and handle shape are therefore not independent choices. The
  character cap covers the complete rendered index, including its header and
  list syntax. Non-`None` bounds must be large enough to retain at least one
  entry; invalid zero or header-only bounds fail at construction time.
- `ToolResult.host_metadata` carries whatever `ToolExecutionHooks.result_metadata`
  returned, verbatim, through to `AgentLoopHooks.render_tool_result`. This is the
  seam for a product that words its own note about a repeated call: whether a call
  *counts* as a repeat is per-tool product policy (`repeat_count`), while whether
  the body came back byte-identical is a separate observed fact with no typed
  field, and both are needed to avoid asserting "identical output" for a body that
  differs. The pass-through is verbatim rather than filtered to unrecognised keys,
  so adopting a new reserved key here cannot silently remove something a product
  already reads.

- `agent_core.components.deliverables`: deliverable reconciliation, extracted
  from ApodexHarness `workflows/agent_team/publication.py` so every consumer
  shares one definition of "delivered" instead of inferring it from `stopped_by`.
  A run is complete only if the loop ended on a stop reason the caller counts as
  completing AND the answer made good on every deliverable it claimed. Judging on
  `stopped_by` alone mis-reported in both directions: a publisher that emitted
  its write as a fenced bash block never ran the tool yet reported the deliverable
  as shipped, while a model that answered in plain text after writing its file was
  reported incomplete because it stopped on `no_tool`.

  Delivery is decided on content identity — a path holding non-empty bytes whose
  `(relpath, size, sha256)` is new relative to this round's baseline. `/outputs`
  persists across rounds, so the baseline is what separates "delivered this
  round" from "a previous round's file is still there"; `touch` and symlinks
  deliver nothing. Both host-specific concerns are injected rather than imported,
  which is what lets the module live in the portable library: `host_root` (where
  the sandbox's `/outputs` really is — core has no mount-resolver fallback,
  because a guessed root reconciles against the wrong tree) and `ignore_matcher`
  (the host's `.outputsignore` policy). Every filesystem entry point has an
  `_async` twin, since hashing a large tree on the event loop stalls every
  concurrent sub-agent while the caller holds the publication lock.

  Claim scanning treats paths as non-whitespace runs rather than an ASCII
  allowlist: an allowlist read `/outputs/报告.pdf` as no claim at all and turned
  `/outputs/résumé.pdf` into `/outputs/r`, so a non-ASCII deliverable was both
  waved through when missing and flagged as unverified when correctly delivered.
  Full-width punctuation terminates a match because Chinese prose leaves no space
  after it. Space-bearing paths are recognised only where the prose delimits them
  unambiguously (backtick span or markdown link target), and where the scan still
  truncates at a space a declared manifest entry outranks the guess.

  New surface only — no existing behaviour changes. See
  [docs/deliverables-boundary.md](docs/deliverables-boundary.md).

### Consumer action

- This changes text the model reads and is therefore a compaction-decision
  change under `docs/versioning.md`. A consumer asserting equality against
  `OMITTED_TOOL_RESULT_PLACEHOLDER` must switch to `startswith`; the placeholder
  remains the first line precisely so that check keeps working.
- No API change for the card. No configuration flag for it either: it has no
  failure mode of its own, and a switch would be one more configuration dimension
  to maintain.
- A product that words a note from `repeat_count`, `repeat_recovery_id`,
  `result_id` or `error_kind` in its own loop copy **must port that note into
  `render_tool_result` before adopting `run_agent_loop`**. AgentCore reads none of
  those fields, so nothing fails if the note is forgotten — it just stops reaching
  the model. `docs/agent-loop-boundary.md` records why, and
  `scripts/check_unconsumed_fields.py` now fails CI on a new field in that state.

### Documented

- `docs/deliverables-boundary.md` states the deliverable-reconciliation split:
  what core decides (the three verdicts, baseline semantics, what it refuses to
  count) and what the host still owns (`host_root`, `.outputsignore` matching,
  which stop reasons complete a run, quarantine of anything refused).
- `docs/agent-loop-boundary.md` now records that `ToolResult.repeat_count`,
  `repeat_recovery_id`, `result_id` and `error_kind` have no consumer inside
  AgentCore, that this is by construction, and that a product moving onto
  `run_agent_loop` therefore loses any note it words from them *silently*. The
  two ways to close it are stated, with `host_metadata` as the chosen route.
- `scripts/check_unconsumed_fields.py` runs in CI: a field on a watched model with
  no attribute read anywhere in `agent_core/` must be named in a boundary
  document. Deciding not to consume a field is a boundary decision, and an
  undocumented one is indistinguishable from an oversight — which is how four
  fields reached 0.4.0 with no consumer and no note.

## [0.6.0] - 2026-09-04

**Never published.** Merged to `main` but never tagged; its contents ship in
0.7.0. Nothing pins it.

### Added

- `agent_core.components.middleware.base.MiddlewareChain` — the phase/tool
  middleware runner for the `ExecutionMiddleware` contract that
  `agent_core.protocols` already declared. Core previously shipped the contract
  and the structural `PhaseMiddlewareChain` without a runner.
- Portable middlewares, extracted from ApodexHarness:
  `middleware.rate_limit` (token-bucket RPM/TPM), `middleware.tool_audit`
  (pattern-based bash / web_fetch risk classification with veto),
  `middleware.status_report` (sub-agent phase heartbeat over the agent bus),
  `middleware.todo` (task-progress injection), and under `middleware.llm`:
  `retry`, `tracing`, `loop_detection`, `output_repair`, `api_key_rotation`,
  `compaction`, `token_accounting`.
- `agent_core.components.memory` — `WorkingMemory` and its snapshot
  recovery, which `middleware.todo` reads.
- `agent_core.protocols.CostSink` and `agent_core.protocols.CostPersister`.
  `CostSink.record` is synchronous and runs per call; `CostPersister.persist`
  runs once at completion and is the only path that reaches durable storage.

See [docs/middleware-boundary.md](docs/middleware-boundary.md) for what stays
with the host — the composition root, chain ordering, trace/event sinks, and any
phase middleware that needs host services.

### Changed

- `LLMCallContext.metadata`'s field factory is now parametrised. No behavioral
  change; it stopped every consumer of `ctx.metadata` from type-checking as
  unknown.
- `CostSink` now states its full contract: `record(...)` plus
  `get_summary(task_id)`. Configuring a `CostPersister` with a record-only sink
  fails immediately instead of silently skipping final persistence.
- `StatusReportMiddleware` no longer knows the research result schema. Hosts can
  pass `result_summarizer(result)` to add their own status fields.
- `TodoMiddleware` obtains domain-specific progress through
  `WorkingMemory.one_line_summary()`, which subclasses can override.
- `ToolAuditMiddleware` accepts an optional host-owned `bash_classifier`. Its
  built-in regex classifier is a conservative defense-in-depth fallback, not a
  replacement for a sandbox or a host filesystem policy.

### Fixed

- `TokenBucket.acquire` now rechecks and reserves capacity atomically after
  sleeping, so concurrent waiters cannot spend the same refill and drive the
  bucket negative.
- `LoopDetectionMiddleware` isolates history and pending hints by task/session,
  role, and phase, and bounds retained scopes. One task can no longer inject a
  loop warning into another.
- Recursive-force `rm` classification recognizes split, reordered, and long
  flags, including `--no-preserve-root`.
- `LLMTracingMiddleware` keeps its start time on `LLMCallContext`; a terminal
  chat failure can no longer leak an entry in a process-long timer dictionary.

### Consumer action

**`TokenAccountingMiddleware` no longer takes `session_factory`.** It takes
`cost_persister: CostPersister | None` instead, and `persist_cost` forwards the
summary rather than writing a table itself. A host that passed a SQLAlchemy
session factory must now pass an object with
`async persist(task_id, summary, model)`; the table name, the column names and
the transaction boundary move with it. Passing neither seam leaves `persist_cost`
a no-op, unchanged.

When `cost_persister` is configured, `cost_sink` must implement both
`record(...)` and `get_summary(task_id)`. Record-only sinks remain valid when
durable persistence is not configured.

Hosts that want research-specific status fields should construct
`StatusReportMiddleware(result_summarizer=...)`; AgentCore no longer reads
`evidence_cards` or `assertions` directly.

Everything else only adds modules. A product adopting these should replace its
own copies with import aliases rather than keeping both.

Note for anyone carrying a private fork of the compaction prompt:
`agent_core.runtime.loop.summary_prompt` has moved on — it now offers
`RESEARCH_COMPACTION_PROMPT`, `HANDOFF_COMPACTION_PROMPT` and a
`compaction_prompt()` selector, with `COMPACTION_PROMPT` aliasing the research
shape. `middleware.llm.compaction` uses it directly.

## [0.5.0] - 2026-09-04

**Never published.** Merged to `main` but never tagged; its contents ship in
0.6.0. Nothing pins it.

### Added

- `agent_core.components.cycle` — the product-neutral write → audit → feedback →
  re-write loop, extracted from ApodexHarness. `WriteAuditCycle` owns max-rounds,
  the wall-clock budget with between-round estimation, the missing-artifact
  output check, writer-exception and missing-output capture as synthetic audits,
  per-round persistence of the audit trail, and failure-isolated round-observer
  dispatch. Ships the `AuditReport` / `WriterOutput` / `AuditFinding` /
  `CycleOutput` contracts with JSON round-trip, `DefaultFeedbackRenderer`,
  `SessionBackedWriter` / `SessionBackedAuditor`, `ScoreThresholdAuditor`,
  `MajorityVoteAuditor`, `select_best_attempt`, `select_by_answer_consensus`, and
  the `BestSoFarObserver` / `PlateauAbortObserver` / `MetricsObserver` built-ins.
- `agent_core.components.verifier` — the `Verifier` and `Generator` protocols,
  the `Verdict` model with nesting, `GroundTruth` / `VerifierContext` with
  automatic oracle-field stripping at runtime, the six composers (Pipeline,
  Ensemble, Fallback, Cascade, Parallel, ConsensusVerifier), and the
  `AuditReport` ↔ `Verdict` bridge so a `Verifier` can stand in for a
  `CycleAuditor`.
- `agent_core.components.observers.conclude_phase_observer.ConcludePhaseObserver`
  — nudges a session toward emitting its structured output once it crosses a
  ratio of its turn budget, instead of exploring until `max_turns` kills it and
  returning empty content. `cycle.builders` depends on it.

Scoring vocabularies stay free-form strings by design, and concrete writers and
auditors stay in the product. See
[docs/cycle-verifier-boundary.md](docs/cycle-verifier-boundary.md).

### Fixed

- `SessionBackedWriter._ensure_session` declared `-> str` while returning the
  `str | None` attribute it caches into. Behavior is unchanged — the bus does
  return an id — but the annotation no longer overstates what was proven.

### Consumer action

None required. This release only adds modules; nothing existing changed shape.
A product adopting these should replace its own copies with import aliases
rather than keeping both, since divergence is the failure this package exists to
prevent.

## [0.4.0] - 2026-09-03

### Added

- `ToolResult` now carries optional structured failure, recovery, and repeated-
  invocation metadata. `ToolExecutionHooks.result_metadata` lets products
  populate those fields without moving product-specific classifiers or stores
  into AgentCore. Timeouts and exceptions receive stable core-owned
  `error_kind` values.
- Loop observers can inspect the exact pre-provider LLM input, correlate calls
  and retries through stable identifiers, and receive pre/post snapshots for
  rollback, overflow recovery, and policy-driven context compaction.
- Products can dynamically constrain the tools available to an individual turn,
  supply callable per-call addenda as system or user messages, customize tool
  history rendering, and preserve host-owned recovery metadata.
- Deterministic and LLM-backed compaction now retain the original task,
  tool-call/result pairing, recoverability references, and recovery-index order.

### Fixed

- Context-discard notifications now fire when overflow recovery replaces a
  message without changing history length and when a custom compactor mutates
  the history list in place.

## [0.3.0] - 2026-09-03

First published release. AgentCore is now open source under Apache-2.0 and
distributed on PyPI as [`apodex-agent-core`](https://pypi.org/project/apodex-agent-core/).

### Consumer action

Replace the Git dependency with the published package:

```toml
dependencies = [
  "apodex-agent-core==0.3.0",   # was: apodex-agent-core @ git+ssh://…@<rev>
]
```

Then delete the credential plumbing this required — the AgentCore deploy key,
`AGENT_CORE_REPO_TOKEN`, and any `insteadOf` rewriting. Installing needs no
credential now, `docker build` included.

### Changed

- **Distribution.** Releases publish to PyPI from the tagged tree using Trusted
  Publishing (OIDC): no API token is stored anywhere, and `id-token: write` is
  scoped to the publishing job alone. GitHub Releases still carry the wheel,
  sdist, and changelog section.
- **Downstream bumps are Dependabot's job.** The `repository_dispatch` +
  repin-script mechanism is deleted along with `.github/downstream/`. It existed
  only because a private Git dependency could not be resolved without a
  credential; a public PyPI package needs none, and Dependabot's `uv` ecosystem
  updates `pyproject.toml` and `uv.lock` and opens a pull request that the
  product's own CI validates. This also removes the need for an organization-level
  GitHub App or a cross-repository PAT.
- Package metadata: SPDX `license = "Apache-2.0"` with `license-files` (PEP 639),
  trove classifiers, and project URLs.
- `docs/versioning.md` records why publishing to PyPI reversed the earlier
  decision to stay on Git pins — going public removed that trade's entire cost
  side.

### Added

- The release now runs `twine check` and installs the built wheel in a clean
  environment to `import agent_core` before publishing. A PyPI version number
  can never be reused, so a broken artifact must fail the release rather than
  consume the number.
- `docs/downstream-bump.md`: PyPI publisher registration, the `pypi` environment,
  the Dependabot config, and the one real limitation — a Dependabot pull request
  runs CI as if from a fork, so `secrets.*` resolves against Dependabot secrets
  rather than Actions secrets.

### Fixed

- The version gate accepted a *decreasing* version: it compared for inequality
  rather than an increase, so a pull request could move `0.2.0` back to `0.1.9`
  and pass.
- The `skip-version-bump` label was inert. The default `pull_request` event types
  exclude `labeled`, so applying the label did not re-run the check and the pull
  request stayed red forever.

## [0.2.0] - 2026-09-03

**Never published.** No tag was pushed and no artifact was distributed; the
version exists only in `main`'s history. Consumers went from Git revisions
straight to `0.3.0` on PyPI. The entry below is kept because it records when
version discipline was introduced.

First versioned release. Functionally this is the runtime that products have
already been consuming by commit SHA; what changes is that the revision now has
a name.

Why 0.2.0 and not 0.1.1: `0.1.0` was the bootstrap constant and was never
released or moved. Every commit from the initial foundation through the
cooldown-fallback observability work declared that same version, so `0.1.0`
identifies no particular code and is retired rather than reused.

### Published surface

The importable surface is `agent_core.*` as scoped in the README, with per-area
contracts in `docs/*-boundary.md`: messages and token estimation, context
management and spill storage, LLM runtime (binding, streaming, retry
classification, runaway recovery), the agent loop and its typed product hooks,
durable run journals, MiniDAG and registries, middleware and observers, agent-bus
coordination, skills, and scheduling.

### Consumer action

Repin from a commit SHA to the tag:

```toml
dependencies = [
  "apodex-agent-core @ git+ssh://git@github.com/ApodexAI/AgentCore.git@v0.2.0",
]
```

No source changes are required: the code at `v0.2.0` is `main` as of this
release. Installed metadata now reports `0.2.0`, so `pip list` and the
`dist-info` in a product image finally identify which AgentCore is running.

Note that as of this release neither product declares AgentCore on its `main`
branch — the pin exists only on each product's in-progress migration branch
(`fix/ci-bwrap-soft-probe`, `refactor/agent-core`). The repin above applies
wherever the declaration currently lives.

### Added

- `docs/versioning.md`: version policy, what counts as a breaking change while
  the public surface is still wider than `agent_core.__all__`, and the release
  procedure.
- `CHANGELOG.md` (this file).
- CI now fails a pull request that changes published code without bumping
  `[project].version`, which is what let 80 commits ship as `0.1.0`.
- `Release` workflow: pushing a `v*` tag re-runs the full check suite, verifies
  the tag matches `[project].version`, builds the wheel and sdist, and publishes
  them on a GitHub Release with these notes attached.
- Automated downstream bump PRs: a release dispatches to ApodexHarness and
  FrontierAgentInternal, which repin and open a pull request for their own CI to
  validate. Templates live in `.github/downstream/`; setup is
  [docs/downstream-bump.md](docs/downstream-bump.md). The workflows mint
  least-privilege, short-lived GitHub App tokens per run, strictly validate the
  dispatched version before exposing it to shell, and do not persist write
  credentials in the checkout.
- Corrected the documented pin scheme to `git+ssh://git@github.com/`, which is
  what both products actually declare. The credential note previously rewrote
  `https://github.com/`, a no-op against an `ssh://` dependency URL.
