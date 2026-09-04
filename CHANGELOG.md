# Changelog

Notable changes to `apodex-agent-core`, newest first. This file is the contract
between AgentCore and its consumers: every release entry must say what a product
has to know before upgrading. The release workflow reads the matching section as
the GitHub Release body, so a release with no entry here fails.

Versioning follows [docs/versioning.md](docs/versioning.md).

## [0.6.0] - 2026-09-04

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

### Consumer action

**`TokenAccountingMiddleware` no longer takes `session_factory`.** It takes
`cost_persister: CostPersister | None` instead, and `persist_cost` forwards the
summary rather than writing a table itself. A host that passed a SQLAlchemy
session factory must now pass an object with
`async persist(task_id, summary, model)`; the table name, the column names and
the transaction boundary move with it. Passing neither seam leaves `persist_cost`
a no-op, unchanged.

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
