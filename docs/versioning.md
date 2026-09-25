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

## Feature pull requests

Feature and fix PRs do not bump the package version or edit `CHANGELOG.md`.
Add one independent file under `changes/`, using the PR number or a unique slug:

- `changes/42.fix.md` for a bug fix or internal change;
- `changes/43.feature.md` for a new capability;
- `changes/example.breaking.md` for a compatibility change.

Write a non-empty release-note paragraph without headings. Explain the consumer
impact and any required migration. Each PR gets its own file, so merging another
feature PR does not conflict on a shared version, lockfile or changelog entry.
CI requires a newly added fragment for published code changes and validates all
pending fragments. Docs/tests/tooling-only PRs need no fragment. The CI status
keeps the name `version-bump` for branch-protection compatibility; the old
`skip-version-bump` label no longer bypasses the gate.

Between releases, main keeps the last released package version. Use a commit SHA
to identify development snapshots; distribution versions identify releases.
Consumers requiring a distinct package version should consume published releases.

## Preparing a release

Create one release PR from current main after the desired feature PRs merge:

```bash
python3 scripts/prepare_release.py --dry-run  # review the aggregated notes
python3 scripts/prepare_release.py
uv lock
```

The script selects the next PATCH for fixes only, or MINOR for any feature or
breaking change. `--version 0.X.Y` can select a higher version. It updates
`pyproject.toml`, prepends one dated `CHANGELOG.md` section, and removes the
consumed fragments. Commit those changes together with `uv.lock`. CI rejects a
stale lock, a missing release entry, remaining fragments, or a version downgrade.
Only this release PR edits the shared version/changelog, so serialize releases.
If more feature PRs merge while the release PR is open, regenerate the release
from updated main to include their fragments before tagging.

Tags additionally reject any pending fragments and stale project lock version,
so unreleased main cannot accidentally publish changes under the old version.

## Releasing

Releases are cut from `main` after CI is green:

```bash
git switch main && git pull
python3 scripts/version.py          # confirm the version you are about to tag
git tag -a v0.3.0 -m 'AgentCore 0.3.0'
git push origin v0.3.0
```

Pushing the tag triggers `.github/workflows/release.yml`, which:

1. verifies the tag matches `[project].version` (a mismatch fails the release);
2. re-runs ruff, pyright, and pytest against the tagged tree;
3. runs `uv build` and `twine check`;
4. publishes to PyPI from a separate job using Trusted Publishing;
5. creates or repairs a GitHub Release carrying the wheel, the sdist, and the
   CHANGELOG section for that version.

**A PyPI version number is consumed permanently.** It cannot be re-uploaded even
after deleting the release or the entire project; yanking hides a release from
resolution but does not free the number. So a botched release is never fixed in
place — bump to the next PATCH and tag again. The same rule applies to tags,
which are never moved or deleted once pushed because a consumer may already have
resolved one.

Because the number cannot be reclaimed, the release job runs `twine check`
before publishing, and it is worth doing a first-time dry run against TestPyPI
rather than discovering a metadata problem on the real index.

## How products consume a release

Depend on the published package with an exact pin:

```toml
dependencies = [
  "apodex-agent-core==0.3.0",
]
```

Exact, because this is a `0.x` series where a MINOR bump may be breaking. The
pin is what makes each upgrade an explicit, reviewable event: Dependabot opens a
pull request against it and the product's own CI decides whether the new version
is safe. See [downstream-bump.md](downstream-bump.md).

## On the package registry

AgentCore publishes to PyPI. This became the obvious choice when the repository
went public, and it is worth recording why the earlier answer was the opposite.

While the repository was private, the options were a private index
(CodeArtifact, Artifactory, Gemfury) or Git pins. Git pins won: a private index
costs credential rotation, availability, and backup work, while buying only
convenience — a Git tag was already immutable, so there was nothing to gain in
reproducibility.

Going public removed the entire cost side of that trade. Public PyPI needs no
credentials to read, nothing to operate, and Trusted Publishing means nothing to
store on the publishing side either. It also removed the *reason* for Git pins:
every consumer previously needed a credential just to resolve the dependency.

What PyPI adds beyond convenience:

- installation with no credential at all, including inside `docker build`;
- no full-repository clone during `uv lock`;
- clean resolution if AgentCore ever becomes a *transitive* dependency;
- Dependabot support, which replaces a bespoke cross-repository bump mechanism
  that would otherwise need organization-level administration;
- stronger immutability than a Git tag, not weaker: a published version number
  can never be reused, whereas a tag is only immutable by convention.
