# Versioning and release

AgentCore is consumed by ApodexHarness and FrontierAgentInternal, which pin an
immutable revision. This document defines what a version number means here, when
to bump it, and how to publish one.

## Scheme: `0.MINOR.PATCH`

- **MINOR** — a breaking change to the published surface, or new capability.
- **PATCH** — a fix or internal change that cannot alter how a correct consumer
  behaves.

Both products live in the same organization and upgrade deliberately, so MINOR
carries breaking changes rather than reserving a MAJOR for them. This is the
standard reading of a `0.x` series: treat MINOR as the compatibility boundary and
pin accordingly.

### Why not 1.0 yet

1.0 is a promise about a stable, enumerated API. AgentCore cannot make it today:
`agent_core.__all__` exports 14 symbols, but consumers import from deep paths
(`agent_core.runtime.loop`, `agent_core.components.middleware.llm`, and others),
so the *effective* public surface is far larger than the declared one. Until the
surface is deliberately narrowed — or the deep paths are explicitly blessed as
public — "is this a breaking change?" cannot be answered consistently, and a 1.0
would be a number without a guarantee behind it.

Prerequisite for 1.0: an explicit statement of which import paths are public,
with everything else moved under a private prefix or re-exported.

## What counts as breaking

Because the deep paths are in practice public, assume any of these is breaking
until shown otherwise:

- removing or renaming a module, class, function, or attribute reachable from
  `agent_core.*`, including deep paths;
- changing a function signature other than by adding a keyword argument with a
  default;
- changing a `Protocol` that products implement (a new required method breaks
  every product implementation) — see `docs/*-boundary.md` for the contracts
  products are expected to satisfy;
- changing the type or meaning of a field on a shared model (`Message`,
  `LLMResponse`, `StreamDelta`, …);
- changing observable runtime behavior a product depends on: emitted event types
  and their payloads, error types raised, retry classification outcomes,
  compaction or trimming decisions;
- tightening a dependency floor in a way that can conflict with a product's
  own pins.

Not breaking: added modules and symbols, added optional keyword arguments, added
event fields consumers can ignore, internal refactors with identical observable
behavior, tests, docs, and tooling.

When in doubt, bump MINOR. The cost of an unnecessary MINOR is nothing; the cost
of a breaking PATCH is a product discovering it in production.

## Bumping

CI fails any pull request that touches `agent_core/` or `pyproject.toml` without
increasing the three-part `[project].version`. Equal versions, downgrades, and
malformed versions are rejected. To bump:

```bash
# 1. Edit [project].version in pyproject.toml.
# 2. Sync the lockfile — uv.lock records this project's own version, and a stale
#    lock makes `uv sync --frozen` fail in CI and in both products.
uv lock
# 3. Add a '## [<version>] - <YYYY-MM-DD>' section to CHANGELOG.md.
```

If a change genuinely cannot affect consumers and the check is wrong, apply the
`skip-version-bump` label to the pull request and say why in the description.
Adding or removing that label triggers a fresh CI run, so the gate reflects the
current escape-hatch decision without requiring an unrelated commit.

## Releasing

Releases are cut from `main` after CI is green:

```bash
git switch main && git pull
python3 scripts/version.py          # confirm the version you are about to tag
git tag -a v0.2.0 -m 'AgentCore 0.2.0'
git push origin v0.2.0
```

Pushing the tag triggers `.github/workflows/release.yml`, which:

1. verifies the tag matches `[project].version` (a mismatch fails the release);
2. re-runs ruff, pyright, and pytest against the tagged tree;
3. runs `uv build`;
4. creates a GitHub Release carrying the wheel, the sdist, and the CHANGELOG
   section for that version.

A tag is never moved or deleted once pushed — products may already have resolved
it. To correct a bad release, bump to the next PATCH and tag again.

## How products consume a release

Pin the tag, not a branch and not a SHA:

```toml
dependencies = [
  "apodex-agent-core @ git+ssh://git@github.com/ApodexAI/AgentCore.git@v0.2.0",
]
```

A tag is as immutable as a SHA in practice (it is never moved, per above) and it
is readable in a diff, so a bump PR states plainly which version a product is
moving to.

## On a private package registry

Not currently used, and not currently needed. A registry would buy convenience —
no SSH/token plumbing for `docker build` and CI runners, no full-repository clone
during `uv lock`, and clean resolution if AgentCore ever becomes a *transitive*
dependency. It buys nothing in reproducibility: a Git tag is already immutable,
whereas a registry version can in principle be yanked or replaced.

With two first-party consumers in one organization, the credential rotation,
availability, and backup burden of a private index (CodeArtifact, Artifactory,
Gemfury) outweighs that convenience. Revisit when either becomes true:

- a third consumer appears, or AgentCore becomes a transitive dependency;
- Git credential distribution starts causing real build failures.

The intermediate step, if only the plumbing hurts: attach the wheel built by the
release workflow — already published on each GitHub Release — and install via
`--find-links`, with no index to operate.
