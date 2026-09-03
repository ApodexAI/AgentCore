# Changelog

Notable changes to `apodex-agent-core`, newest first. This file is the contract
between AgentCore and its consumers: every release entry must say what a product
has to know before upgrading. The release workflow reads the matching section as
the GitHub Release body, so a release with no entry here fails.

Versioning follows [docs/versioning.md](docs/versioning.md).

## [0.4.0] - 2026-09-03

### Added

- `ToolResult` now carries optional structured failure, recovery, and repeated-
  invocation metadata. `ToolExecutionHooks.result_metadata` lets products
  populate those fields without moving product-specific classifiers or stores
  into AgentCore. Timeouts and exceptions receive stable core-owned
  `error_kind` values.

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
