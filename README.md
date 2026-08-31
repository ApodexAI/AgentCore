# AgentCore

AgentCore is the single source of truth for product-neutral agent runtime code
shared by MiroHarness and FrontierAgentInternal.

The repository exists to remove a failure-prone workflow: implementing a core
change independently in two products and then opening more pull requests to
reconcile the copies. Shared behavior is changed and reviewed here once;
products consume an immutable AgentCore revision and keep only their adapters.

## Current scope

Version `0.1.x` contains the converged foundation layer:

- native message types and constructors;
- token-estimation helpers;
- compaction policy and deterministic compactor;
- context-budget estimation and non-blocking tokenizer access;
- message trimming;
- provider-neutral LLM response, stream, and client contracts;
- loop configuration, lifecycle contexts, observer protocol, intervention
  merging, and observer dispatch helpers.

The initial extraction is based on the already-merged integration branches:

- MiroHarness `c1229050` (PR #501);
- FrontierAgentInternal `63b89c8` (PR #92).

Those revisions are provenance, not runtime dependencies. AgentCore tests and
builds without either product checkout.

The LLM call runtime and agent loop remain in the products until tool
execution, model profiles, retry classification, execution-context storage,
and runtime hooks have product-neutral boundaries. Moving those files before
that boundary exists would only hide product coupling inside this package.

## Repository boundary

Code belongs in AgentCore when it:

- has the same behavioral contract in every product;
- does not import `miroharness`, `frontier_agent`, workflows, plugins, storage,
  or deployment configuration;
- accepts product behavior through typed inputs or explicit hooks;
- has tests that run without either product repository installed.

Provider clients, environment/config loading, session affinity, persistence,
workflow assembly, and product-specific observers remain in their product.

## Development

```bash
uv sync --frozen --extra dev
uv run ruff check agent_core tests
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

## Consuming a private revision

Products should pin an immutable commit, never a branch:

```toml
dependencies = [
  "apodex-agent-core @ git+https://github.com/ApodexAI/AgentCore.git@<commit>",
]
```

Because this repository is private, CI needs a read-only credential that can
clone `ApodexAI/AgentCore`. Use a dedicated GitHub App or fine-grained token
stored as `AGENT_CORE_REPO_TOKEN`; do not use a developer's personal token.
Configure Git before `uv sync`:

```bash
git config --global \
  url."https://x-access-token:${AGENT_CORE_REPO_TOKEN}@github.com/".insteadOf \
  "https://github.com/"
```

Until that credential is installed in both product repositories, product
source must not be switched to the private dependency: doing so would make a
clean checkout and CI unreproducible.

## Change and release workflow

1. Reproduce a shared bug with an AgentCore test.
2. Change AgentCore in one pull request and pass its standalone CI.
3. Merge and record the immutable commit SHA (or publish a tagged version).
4. Automation opens dependency-bump PRs in MiroHarness and
   FrontierAgentInternal.
5. Product CI validates adapters and end-to-end behavior. Product PRs must not
   patch vendored/shared implementation code.

Product-only bugs stay in the product repository. If a proposed fix needs to
edit both products' core copies, that is evidence it belongs here.

## Migration plan

1. **Foundation** (this version): messages, token estimation, compaction,
   context budget, trimming.
2. **Runtime contracts** (in progress): LLM protocols and loop types are now
   shared; errors, execution-context storage, retry classification, and
   explicit product hooks remain.
3. **LLM runtime:** binding, calls, streaming, response normalization, runaway
   recovery, and the public `llm_client` facade.
4. **Agent loop:** model/tool parsing, tool execution, and `agent_loop`.
5. Remove product compatibility facades after downstream imports have moved to
   `agent_core`.

Each slice must leave product CI green and must not depend on a floating branch.
