# AgentCore

AgentCore is the engine runtime package for
[FrontierAgent](https://github.com/ApodexAI/FrontierAgent). It is the single
source of truth for the product-neutral execution engine: FrontierAgent owns
the product experience and composition, while AgentCore owns the reusable
runtime beneath it.

The repository exists to remove a failure-prone workflow: implementing a core
change independently in two products and then opening more pull requests to
reconcile the copies. Shared behavior is changed and reviewed here once;
FrontierAgent consumes an immutable AgentCore release and keeps only its
product adapters, workflows, tools, and user-facing surfaces.

## Current scope

Version `0.2.x` contains the converged foundation layer:

- native message types and constructors;
- token-estimation helpers;
- compaction policy and deterministic compactor;
- context-budget estimation and non-blocking tokenizer access;
- message trimming;
- bounded LLM summaries, projected-token triggers, and tiered compaction;
- session-isolated, content-addressed spill storage for recoverable tool output;
- product-neutral identity/status types, kernel events, and task-local execution
  context propagation;
- failure-isolated async event dispatch and composable fail-closed tool
  permission policy;
- provider-neutral LLM response, stream, and client contracts;
- profile-driven auxiliary-client construction, summary execution, legacy
  cooldown fallback, and task-local external-API metering;
- loop configuration, lifecycle contexts, observer protocol, intervention
  merging, and observer dispatch helpers.
- streamed tool-call recovery checks for missing required arguments.
- LLM binding, response normalization, streaming assembly/watchdogs, retry
  classification, runaway recovery, and physical-call orchestration.
- model profiles, thinking/history normalization, and multi-format tool-call
  parsing;
- parallel tool execution and the complete agent-loop orchestration engine,
  with product behavior isolated behind typed hooks.
- append-only, scope-isolated durable run journals with content-addressed blob
  attachments, projections, and event-sourced notes;
- MiniDAG execution, declarative pipeline models, dynamic graph construction,
  and scoped service/agent/pipeline/topology registries;
- reusable LLM middleware for proxying, summarization, skill injection, and
  stream-repetition detection;
- portable loop observers for budget/context/wall-clock guards, finalization,
  repetition detection, SSE events, task boards, and bounded trajectories;
- complete agent-bus communication, spawn control, reusable sessions,
  result recovery, fan-in policy, shared pools and stop signals;
- portable tools, task/finalization budgets, role-scoped resource management,
  and backend-injected DAG scheduling;
- filesystem skill loaders, session history, renewable wall-time leases, and
  language helpers.

The initial extraction is based on the already-merged integration branches:

- ApodexHarness `c1229050` (PR #501);
- FrontierAgent (originally extracted from FrontierAgentInternal) `63b89c8`
  (PR #92).

Those revisions are provenance, not runtime dependencies. AgentCore tests and
builds without either product checkout.

The shared loop accepts product decisions through explicit hooks. Products own
composition roots, endpoint configuration, metering policy, durable-journal
root/scope selection, and sandbox mount policy; they no longer need private
copies of the orchestration, parsing, context-management, DAG, observer, or
spill-storage engines.

## Repository boundary

Code belongs in AgentCore when it:

- has the same behavioral contract in every product;
- does not import `miroharness`, `frontier_agent`, workflows, plugins, storage,
  or deployment configuration;
- accepts product behavior through typed inputs or explicit hooks;
- has tests that run without either product repository installed.

Provider catalogs, credentials, endpoint selection, host session-affinity
context, billing policy, durable process/event-store implementations,
user/session retention policy, checkpoint association, workflow node
implementations, sandbox mounting/authorization, UI history, and
product-specific observers remain in their product. AgentCore owns the
provider transports and product-neutral affinity lifecycle safeguards.

## Development

```bash
uv sync --frozen --extra dev
uv run ruff check agent_core tests scripts
uv run pyright agent_core
uv run pytest -q
uv build
```

For local work beside the product repositories:

```bash
uv pip install -e ../AgentCore
```

AgentCore uses the import namespace `agent_core` and the distribution name
`apodex-agent-core`.

## Consuming the release package

AgentCore is published on PyPI:

```bash
uv add apodex-agent-core        # or: pip install apodex-agent-core
```

FrontierAgent pins an exact version:

```toml
dependencies = [
  "apodex-agent-core==0.3.0",
]
```

Exact, because this is a `0.x` series where a MINOR bump may be breaking. The
pin makes each upgrade a reviewable event: Dependabot opens a pull request
against it and FrontierAgent CI decides whether the new version is safe. See
[docs/downstream-bump.md](docs/downstream-bump.md).

No credential is involved anywhere in that path — the repository and the package
are both public. Every release also has a GitHub Release carrying the wheel, the
sdist, and its changelog section. See
[docs/versioning.md](docs/versioning.md) for what a version number means and
[CHANGELOG.md](CHANGELOG.md) for what changed.

## Release package contract

Each AgentCore release is the installable engine runtime used by FrontierAgent.
The release must:

- publish both a Python wheel and source distribution built from the tagged,
  clean repository state;
- include the `agent_core` package, `py.typed`, README, and Apache-2.0 license,
  with package metadata matching the tag;
- expose only product-neutral runtime APIs and keep FrontierAgent imports,
  prompts, workflow profiles, tools, UI/TUI code, benchmarks, credentials, and
  deployment configuration out of the package;
- document compatibility or migration impact in `CHANGELOG.md`, and use the
  versioning rules in [docs/versioning.md](docs/versioning.md);
- pass lint, type checking, the complete test suite, and an install/import smoke
  test against the built wheel before publication;
- attach the wheel, source distribution, and matching changelog section to the
  GitHub Release, and publish the same artifacts to PyPI so Dependabot can open
  FrontierAgent's bump pull request.

A release is not complete merely because a tag exists. It is complete when its
artifacts can be installed in a clean Python 3.12 environment, `import
agent_core` succeeds, the artifacts carry the repository license, and
FrontierAgent CI passes against the exact released version. Release artifacts
must not depend on a sibling checkout or files outside the distribution.

## Change and release workflow

1. Reproduce a shared bug with an AgentCore test.
2. Change AgentCore in one pull request. Bump `[project].version`, run
   `uv lock`, and add a `CHANGELOG.md` entry — CI fails a pull request that
   changes published code without a version bump.
3. Merge, then tag and push:

   ```bash
   git switch main && git pull
   git tag -a v0.3.0 -m 'AgentCore 0.3.0'
   git push origin v0.3.0
   ```

   The `Release` workflow verifies the tag matches the declared version,
   re-runs the full check suite against the tagged tree, builds and validates the
   artifacts, publishes them to PyPI through Trusted Publishing, then creates a
   retry-safe GitHub Release with the wheel, sdist, and changelog section.
4. Dependabot normally opens an exact-version bump PR in each product. While
   GitHub's current `uv` updater defect is unresolved, open that PR with the
   documented no-credential manual fallback instead. See
   [docs/downstream-bump.md](docs/downstream-bump.md).
5. Product CI validates adapters and end-to-end behavior. Product PRs must not
   patch vendored/shared implementation code.

[docs/versioning.md](docs/versioning.md) covers the version scheme, what counts
as a breaking change, why AgentCore is not on 1.0 yet, and why publishing to
PyPI replaced the earlier Git-pin approach.

Product-only bugs stay in the product repository. If a proposed fix needs to
edit both products' core copies, that is evidence it belongs here.

## Migration plan

1. **Foundation** (complete): messages, token estimation, deterministic
   compaction, context budget, trimming.
2. **Runtime contracts** (complete): LLM protocols, loop types, errors, retry
   classification, and explicit product hooks are shared.
3. **LLM runtime** (complete): binding, calls, streaming, response
   normalization, runaway recovery, and the public `llm_client` facade.
4. **Agent loop** (complete in AgentCore): model/tool parsing, hook-driven tool
   execution, and `agent_loop` orchestration.
5. **Context management Phase 2** (complete in AgentCore): calibrated trigger,
   bounded LLM summary, tier selection, observability, and spill storage.
6. **Runtime foundation** (complete in AgentCore): task identity/status,
   execution ContextVars, kernel events, event dispatch, and tool permissions.
7. **Portable runtime batch** (complete in AgentCore): durable run journals,
   MiniDAG/graph construction, middleware/observers, agent-bus coordination
   primitives (shared pools and stop signals), skills, session history, models,
   and scoped registries.
8. **Agent-bus scheduling** (complete in AgentCore): communication models,
   durable result-message recovery, runtime, spawn guard, reusable sessions,
   bus/fan-in orchestration, resource management, task budgets, and the
   backend-injected DAG scheduler. Product tool assembly, persistence backends,
   and workflow policy remain downstream.
9. **Provider substrate** (complete in AgentCore): shared provider transports,
   fallback engine, prompt cache, usage metadata normalization, and non-blocking streaming
   behind product configuration and session-affinity adapters.
10. **Shared runtime closeout**: portable usage metering, configurable aux-LLM
    construction, summary execution, tool-call repair/guardrails, and safe
    workflow-default merging.
11. Remove product compatibility facades after downstream imports have moved to
    `agent_core` and the compatibility window has elapsed.

Each slice must leave product CI green and must not depend on a floating branch.

## License

AgentCore is licensed under the [Apache License 2.0](LICENSE), the same license
as FrontierAgent.
