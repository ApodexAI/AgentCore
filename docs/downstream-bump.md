# Automated downstream bump PRs

When AgentCore publishes a release, both products should get a pull request
moving their pin — without anyone remembering to open it. This document is the
one-time setup.

Flow: release tag pushed → `Release` workflow publishes the GitHub Release →
`repository_dispatch` (`agent-core-release`) fires at both products → each opens
`chore/agent-core-<tag>` with the repinned `pyproject.toml` and `uv.lock`, and
its own CI validates the upgrade.

## One-time setup

### 1. A GitHub App for release automation

Create a GitHub App installed only on `ApodexAI/AgentCore`,
`ApodexAI/ApodexHarness`, and `ApodexAI/FrontierAgentInternal`. Grant the App:

- **contents: write**, needed to dispatch and push bump branches;
- **pull-requests: write**, needed to open bump pull requests.

GitHub App permissions apply to the installation rather than varying per
repository. The workflows therefore generate separate, short-lived tokens and
downscope each one to only the repositories and permissions needed by that
step: read-only for resolving AgentCore, product-write for a bump PR, and
downstream-only write access for dispatch.

Do not create an installation token by hand and save it as a secret.
Installation tokens expire after one hour; the workflows use
`actions/create-github-app-token` to mint and revoke one for every run.

### 2. In all three repositories

Configure the same two values in AgentCore and each product:

- repository variable `AGENT_CORE_AUTOMATION_APP_CLIENT_ID` — the App's client
  ID;
- repository secret `AGENT_CORE_AUTOMATION_APP_PRIVATE_KEY` — the App's private
  key.

If those values are absent or invalid in AgentCore, publishing the release still
succeeds and logs a warning — dispatching is downstream plumbing and must never
fail a published release. Missing or invalid credentials in a product correctly
fail its bump workflow because it cannot safely resolve or push the update.

### 3. In each product repository

Copy both files out of `.github/downstream/` in this repository:

| From | To |
| --- | --- |
| `.github/downstream/bump-agent-core.yml` | `.github/workflows/bump-agent-core.yml` |
| `.github/downstream/repin_agent_core.py` | `.github/workflows/repin_agent_core.py` |

The workflow creates one read-only token for AgentCore and a separate write
token for the current product. It checks out with `persist-credentials: false`,
so neither token remains embedded in the repository's Git configuration. Do not
replace the App token with the default `GITHUB_TOKEN`: pull requests created
with `GITHUB_TOKEN` do not trigger workflow runs, so the product's CI would never
run against the bump — which is the only reason the PR exists.

## Verifying the wiring without cutting a release

Each product's workflow also accepts `workflow_dispatch` with a version input.
Run it manually against an existing tag: it should open a PR, or report
`Already pinned`. Use the same path to recover a dispatch that was missed
because a secret was absent when the release ran.

## Two things that will bite

**The pin's declaration shape.** `repin_agent_core.py` substitutes the rev in
place and fails loudly if it does not find exactly one pin. Do not "simplify" it
to `uv add`: that rewrites the PEP 508 direct URL into uv's `[tool.uv.sources]`
table, which pip ignores, breaking non-uv install paths such as a Dockerfile
running `pip install .`. If the check reports a count other than 1, the
dependency was redeclared and the script needs updating — that is the intended
behavior, not an obstacle.

**`uv.lock` must be committed with `pyproject.toml`.** The lock records both the
tag and the commit it resolved to, plus AgentCore's own version. A bump that
edits only `pyproject.toml` leaves the lock stale and fails `uv sync --frozen`.
The workflow commits both.

## Prerequisite: the pin must be on the product's default branch

As of AgentCore 0.2.0, **neither product declares AgentCore on `main`**. The pin
lives only on each product's in-progress migration branch:

| Product | Branch carrying the pin | Pinned revision |
| --- | --- | --- |
| ApodexHarness | `fix/ci-bwrap-soft-probe` | `a9b5272` |
| FrontierAgentInternal | `refactor/agent-core` | `a9b5272` |

Until one of those merges, this workflow has nothing to repin on `main`:
`repin_agent_core.py` will report `found 0` and exit non-zero rather than commit
a half-edited pin. That is the intended behavior, but it means the automation is
inert — install it now so it is ready, and expect its first real run only after
the migration lands.

Both branches pin `a9b5272` (the #21 merge) and both lockfiles report
`apodex-agent-core 0.1.0`. Whichever merges first, moving it to `v0.2.0` also
picks up the five commits after that merge — the cooldown-fallback observability
fixes in `components/middleware/llm/base.py`, `providers/fallback.py`, and
`retry_policy.py`.
