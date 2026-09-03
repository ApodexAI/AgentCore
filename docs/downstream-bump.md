# Publishing releases and automating downstream bumps

AgentCore is public and publishes to PyPI. Consumers depend on a normal version
specifier, and **Dependabot** is the long-term bump mechanism — so there is no
cross-repository credential, no dispatch, and no bespoke repin script anywhere
in this design. A no-credential manual fallback covers the current upstream
Dependabot defect documented below.

Flow: tag `v0.3.0` → `Release` workflow verifies, tests, and builds → publishes
to PyPI via Trusted Publishing → creates or repairs the matching GitHub Release
→ Dependabot (or the temporary manual fallback) opens a product pull request
that the product's own CI validates.

## One-time setup in AgentCore

### 1. Register the PyPI publisher (before the first release)

`apodex-agent-core` does not exist on PyPI yet, so use the **pending publisher**
flow: on PyPI, go to your account → *Publishing* (not a project page, since
there is no project yet) and add a GitHub Actions publisher:

| Field | Value |
| --- | --- |
| PyPI project name | `apodex-agent-core` |
| Owner | `ApodexAI` |
| Repository name | `AgentCore` |
| Workflow name | `release.yml` |
| Environment | `pypi` |

This needs **no GitHub organization permission and no GitHub App** — it is
configured entirely on the PyPI side by whoever will own the PyPI project. A
pending publisher does not reserve the name: it converts to a real publisher on
the first successful upload, and if someone else registers `apodex-agent-core`
first it is invalidated.

### 2. Create the `pypi` environment on GitHub

Settings → Environments → *New environment* → `pypi`. The publishing job
declares `environment: pypi`, and the registration above binds PyPI's trust to
that name. Add a required reviewer there if you want releases to pause for
approval.

No secrets are involved: `permissions: id-token: write` on the publishing job
lets PyPI verify an OIDC claim naming this repository, this workflow file, and
that environment, and PyPI then issues its own short-lived upload token.

## One-time setup in each product

### 1. Depend on the published package

```toml
dependencies = [
  "apodex-agent-core==0.3.0",
]
```

Pin exactly (`==`). AgentCore is `0.x`, where a MINOR bump may be breaking (see
[versioning.md](versioning.md)); an exact pin is what makes the Dependabot pull
request the place where an upgrade is reviewed. The old
`git+ssh://git@github.com/ApodexAI/AgentCore.git@<rev>` form and every credential
it required — deploy keys, `AGENT_CORE_REPO_TOKEN`, `insteadOf` rewriting — are
obsolete now that the package is public.

### 2. Add `.github/dependabot.yml`

```yaml
version: 2
updates:
  - package-ecosystem: "uv"
    directory: "/"
    schedule:
      interval: "daily"
    # Watch only AgentCore here. Everything else this product depends on is its
    # own concern and would bury the one bump that needs product review.
    allow:
      - dependency-name: "apodex-agent-core"
    commit-message:
      prefix: "chore"
    labels:
      - "agent-core"
```

Dependabot's `uv` ecosystem is intended to update both `pyproject.toml` and
`uv.lock`, and it works with private and internal repositories. However, as of
2026-09-03 its hosted updater has an open defect that passes the target version
as an invalid positional argument to `uv lock`, so version-update jobs can fail
before opening a pull request. Track
[dependabot-core#15842](https://github.com/dependabot/dependabot-core/issues/15842)
and keep the configuration above in place so automation resumes when GitHub
ships the fix.

Until then, update without any AgentCore credential from a product checkout:

```bash
uv add 'apodex-agent-core==0.3.0'
uv sync --frozen
```

Commit both `pyproject.toml` and `uv.lock` in the same pull request and let the
product's normal CI validate it. Substitute the new release number each time;
the package and repository are public, so this fallback needs no token, deploy
key, GitHub App, or cross-repository permission.

## The one real limitation

**A Dependabot pull request runs CI as if it came from a fork.** Its workflow
run gets a read-only `GITHUB_TOKEN` and, critically, `secrets.*` resolves
against **Dependabot secrets** — a separate store — not Actions secrets.

- Plain test CI (checkout, `uv sync`, `pytest`) works unchanged: it needs only
  read access.
- Any step that consumes a secret sees an empty value unless that secret is
  *also* added under Settings → Secrets and variables → **Dependabot**. Check
  each product for steps that need one (container registry pushes, for instance)
  and either duplicate the secret there or guard the step to skip on
  Dependabot-authored pull requests.
- A step needing write access must ask for it explicitly with a `permissions:`
  block.

This is worth the trade: the alternative was a cross-repository App or PAT,
which needs organization-level administration and puts a long-lived credential
in both products.

## Current state

Neither product declares AgentCore on `main` yet. The dependency lives only on
each product's in-progress migration branch:

| Product | Branch carrying the dependency |
| --- | --- |
| ApodexHarness | `fix/ci-bwrap-soft-probe` |
| FrontierAgentInternal | `refactor/agent-core` |

Both still use the pre-open-source `git+ssh://…@a9b5272` form. Whichever merges
first should switch to `apodex-agent-core==0.3.0` and add the Dependabot config
above; Dependabot has nothing to update until a PyPI dependency is on the
default branch.
