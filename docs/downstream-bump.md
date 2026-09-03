# Automated downstream bump PRs

When AgentCore publishes a release, both products should get a pull request
moving their pin — without anyone remembering to open it. This document is the
one-time setup.

Flow: release tag pushed → `Release` workflow publishes the GitHub Release →
`repository_dispatch` (`agent-core-release`) fires at both products → each opens
`chore/agent-core-<tag>` with the repinned `pyproject.toml` and `uv.lock`, and
its own CI validates the upgrade.

## One-time setup

### 1. A token that can reach both repositories

Create a GitHub App installed on `ApodexAI/AgentCore`,
`ApodexAI/ApodexHarness`, and `ApodexAI/FrontierAgentInternal`, with:

- **contents: read** on AgentCore (so `uv` can resolve the private dependency);
- **contents: write** and **pull-requests: write** on the two products.

A fine-grained PAT works too, but must not be a personal one — it becomes a
single-person dependency for every release. Do not reuse the default
`GITHUB_TOKEN`; see the warning in step 3.

### 2. In AgentCore

Add the token as the `DOWNSTREAM_BUMP_TOKEN` secret. Without it the release
still succeeds and logs a warning — dispatching is downstream plumbing and must
never fail a published release.

### 3. In each product repository

Copy both files out of `.github/downstream/` in this repository:

| From | To |
| --- | --- |
| `.github/downstream/bump-agent-core.yml` | `.github/workflows/bump-agent-core.yml` |
| `.github/downstream/repin_agent_core.py` | `.github/workflows/repin_agent_core.py` |

Then add two secrets:

- `AGENT_CORE_REPO_TOKEN` — read access to AgentCore, for dependency resolution.
- `BUMP_PR_TOKEN` — used to push the branch and open the PR.

**`BUMP_PR_TOKEN` must not be the default `GITHUB_TOKEN`.** Pull requests created
with `GITHUB_TOKEN` do not trigger workflow runs, so the product's CI would never
run against the bump — which is the only reason the PR exists. The failure is
silent: you get a PR with no checks on it.

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

## What the products currently pin

As of AgentCore 0.2.0, both products pin commit
`a9b5272` (the PR #21 merge) and both report `apodex-agent-core 0.1.0` in their
lockfiles. Moving them to `v0.2.0` also picks up the five commits after that
merge — the cooldown-fallback observability fixes in
`components/middleware/llm/base.py`, `providers/fallback.py`, and
`retry_policy.py`.
