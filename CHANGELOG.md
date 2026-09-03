# Changelog

Notable changes to `apodex-agent-core`, newest first. This file is the contract
between AgentCore and its consumers: every release entry must say what a product
has to know before upgrading. The release workflow reads the matching section as
the GitHub Release body, so a release with no entry here fails.

Versioning follows [docs/versioning.md](docs/versioning.md).

## [0.2.0] - 2026-09-03

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
  [docs/downstream-bump.md](docs/downstream-bump.md).
- Corrected the documented pin scheme to `git+ssh://git@github.com/`, which is
  what both products actually declare. The credential note previously rewrote
  `https://github.com/`, a no-op against an `ssh://` dependency URL.
