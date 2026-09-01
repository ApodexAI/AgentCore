# AgentCore

AgentCore is the single source of truth for product-neutral agent runtime code
shared by ApodexHarness and FrontierAgentInternal.

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
- bounded LLM summaries, projected-token triggers, and tiered compaction;
- session-isolated, content-addressed spill storage for recoverable tool output;
- product-neutral identity/status types, kernel events, and task-local execution
  context propagation;
- failure-isolated async event dispatch and composable fail-closed tool
  permission policy;
- provider-neutral LLM response, stream, and client contracts;
- loop configuration, lifecycle contexts, observer protocol, intervention
  merging, and observer dispatch helpers.
- streamed tool-call recovery checks for missing required arguments.
- LLM binding, response normalization, streaming assembly/watchdogs, retry
  classification, runaway recovery, and physical-call orchestration.
- model profiles, thinking/history normalization, and multi-format tool-call
  parsing;
- parallel tool execution and the complete agent-loop orchestration engine,
  with product behavior isolated behind typed hooks.

The initial extraction is based on the already-merged integration branches:

- ApodexHarness `c1229050` (PR #501);
- FrontierAgentInternal `63b89c8` (PR #92).

Those revisions are provenance, not runtime dependencies. AgentCore tests and
builds without either product checkout.

The shared loop accepts product decisions through explicit hooks. Products own
execution-context storage, endpoint registries, metering, deadline policy, and
sandbox mount policy; they no longer need private copies of the orchestration,
parsing, context-management, or spill-storage engines.

## Repository boundary

Code belongs in AgentCore when it:

- has the same behavioral contract in every product;
- does not import `miroharness`, `frontier_agent`, workflows, plugins, storage,
  or deployment configuration;
- accepts product behavior through typed inputs or explicit hooks;
- has tests that run without either product repository installed.

Provider clients, environment/config loading, session affinity, checkpoint
persistence, workflow assembly, sandbox mounting, and product-specific
observers remain in their product.

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
4. Automation opens dependency-bump PRs in ApodexHarness and
   FrontierAgentInternal.
5. Product CI validates adapters and end-to-end behavior. Product PRs must not
   patch vendored/shared implementation code.

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
7. Remove product compatibility facades after downstream imports have moved to
   `agent_core`.

Each slice must leave product CI green and must not depend on a floating branch.
