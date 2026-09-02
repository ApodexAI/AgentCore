"""Node-wrapper contracts in ``DynamicGraphBuilder.build``.

Three defects lived in the wrapper stack, all invisible in the default
configuration and all surfacing only once a *second* feature was switched on:

* ``metadata["max_retries"]`` only retried when a middleware *suppressed* the
  error, so with the default ``ExecutionMiddleware.on_error`` (which returns
  the error) it never retried at all;
* a node returning ``None`` — which ``MiniDAGRunner`` explicitly supports —
  crashed the middleware wrapper with ``AttributeError``, so registering a
  middleware chain broke nodes that had always worked;
* the NodeContext wrapper sat *inside* the context filter, so
  ``ctx.task_id`` read the filtered state and came back empty for any node
  whose ``ContextPolicy.include_fields`` omitted ``task_id`` — the normal use
  of the policy.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_core.models.pipeline_spec import (
    ContextPolicy,
    NodeDefinition,
    PipelineSpec,
    TransitionSpec,
)
from agent_core.protocols import ExecutionMiddleware, PhaseContext
from agent_core.runtime.dag.graph_builder import DynamicGraphBuilder

# ── node functions ────────────────────────────────────────────────────────
#
# Injected through ``_resolve_node_function_from_def`` rather than a dotted
# path: importing this module by name gives pytest's copy and importlib's copy
# separate module objects, so a call counter at module level would be counted
# in the one the test cannot see.


def _counting_node(*, fails_before: int) -> tuple[Any, list[int]]:
    """Node that fails its first ``fails_before - 1`` attempts."""
    calls: list[int] = []

    async def node(state):
        calls.append(1)
        if len(calls) < fails_before:
            raise RuntimeError(f"attempt {len(calls)} failed")
        return {"current_phase": "n", "attempts": len(calls)}

    return node, calls


def _always_failing_node() -> tuple[Any, list[int]]:
    calls: list[int] = []

    async def node(state):
        calls.append(1)
        raise RuntimeError("permanently broken")

    return node, calls


async def none_node(state):
    """A node that reports no state change — legal per MiniDAGRunner."""
    return None


async def task_id_node(state, ctx):
    return {"current_phase": "tid", "seen_task_id": ctx.task_id}


class _Chain:
    """Minimal ``PhaseMiddlewareChain`` over real ``ExecutionMiddleware``."""

    def __init__(self, *mws: ExecutionMiddleware) -> None:
        self._mws = list(mws)

    @property
    def middlewares(self) -> list[ExecutionMiddleware]:
        return list(self._mws)

    async def run_before_phase(self, ctx: PhaseContext) -> PhaseContext:
        for mw in self._mws:
            ctx = await mw.before_phase(ctx)
        return ctx

    async def run_after_phase(
        self, ctx: PhaseContext, result: dict[str, Any],
    ) -> dict[str, Any]:
        for mw in self._mws:
            result = await mw.after_phase(ctx, result)
        return result

    async def run_on_error(
        self, ctx: PhaseContext, error: Exception,
    ) -> Exception | None:
        out: Exception | None = error
        for mw in self._mws:
            out = await mw.on_error(ctx, error)
        return out


class _Suppressing(ExecutionMiddleware):
    async def on_error(self, ctx: PhaseContext, error: Exception) -> None:
        return None


def _builder(chain: Any | None, node_fn: Any = None) -> DynamicGraphBuilder:
    builder = DynamicGraphBuilder()
    builder._get_middleware_chain = lambda: chain  # type: ignore[method-assign]
    if node_fn is not None:
        builder._resolve_node_function_from_def = (  # type: ignore[method-assign]
            lambda node_def: node_fn
        )
    return builder


def _spec(node: NodeDefinition) -> PipelineSpec:
    return PipelineSpec(
        pipeline_id="t",
        nodes=[node],
        transitions=[TransitionSpec(from_phase=node.node_id, to_phase="__END__")],
        entry_point=node.node_id,
        terminal_nodes=[node.node_id],
    )


def _node(fn: str, **kw: Any) -> NodeDefinition:
    return NodeDefinition(
        node_id=kw.pop("node_id", "n"), role_id="researcher",
        node_function=f"tests.test_graph_builder_node_contract.{fn}", **kw,
    )


# ── max_retries ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_max_retries_works_with_the_default_middleware():
    """The default ``on_error`` returns the error; that must not disable retry."""
    node, calls = _counting_node(fails_before=3)
    dag = _builder(_Chain(ExecutionMiddleware()), node).build(
        _spec(_node("x", node_id="n", metadata={"max_retries": 2})),
    )

    result = await dag.nodes["n"]({"task_id": "t1"})

    assert len(calls) == 3
    assert result["attempts"] == 3


@pytest.mark.asyncio
async def test_retries_are_bounded_and_the_error_surfaces():
    node, calls = _always_failing_node()
    dag = _builder(_Chain(ExecutionMiddleware()), node).build(
        _spec(_node("x", node_id="n", metadata={"max_retries": 2})),
    )

    with pytest.raises(RuntimeError, match="permanently broken"):
        await dag.nodes["n"]({"task_id": "t1"})

    assert len(calls) == 3  # 1 initial + 2 retries


@pytest.mark.asyncio
async def test_no_retries_declared_means_one_attempt():
    node, calls = _always_failing_node()
    dag = _builder(_Chain(ExecutionMiddleware()), node).build(
        _spec(_node("x", node_id="n")),
    )

    with pytest.raises(RuntimeError):
        await dag.nodes["n"]({"task_id": "t1"})

    assert len(calls) == 1


@pytest.mark.asyncio
async def test_suppressing_middleware_still_retries_then_advances():
    node, calls = _always_failing_node()
    dag = _builder(_Chain(_Suppressing()), node).build(
        _spec(_node("x", node_id="n", metadata={"max_retries": 1})),
    )

    result = await dag.nodes["n"]({"task_id": "t1"})

    assert len(calls) == 2
    assert result["current_phase"] == "n"


# ── a node may return None ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_none_result_survives_a_registered_middleware_chain():
    """Worked with no chain; must not start raising once one is registered."""
    dag = _builder(_Chain(ExecutionMiddleware())).build(
        _spec(_node("none_node", node_id="none")),
    )

    result = await dag.nodes["none"]({"task_id": "t1"})

    assert isinstance(result, dict)
    assert "execution_context" in result


@pytest.mark.asyncio
async def test_none_result_without_middleware_is_unchanged():
    dag = _builder(None).build(_spec(_node("none_node", node_id="none")))

    assert await dag.nodes["none"]({"task_id": "t1"}) is None


# ── NodeContext sees the unfiltered state ─────────────────────────────────


@pytest.mark.asyncio
async def test_task_id_survives_a_context_policy_that_omits_it():
    dag = _builder(None).build(_spec(_node(
        "task_id_node", node_id="tid",
        context_policy=ContextPolicy(include_fields=["original_question"]),
    )))

    result = await dag.nodes["tid"]({
        "task_id": "task-42", "original_question": "q", "secret": "hidden",
    })

    assert result["seen_task_id"] == "task-42"


@pytest.mark.asyncio
async def test_context_policy_still_filters_the_state_the_node_receives():
    """Guard against the reorder leaking unfiltered state into the node."""
    seen: dict[str, Any] = {}

    async def spy(state, ctx):
        seen["keys"] = sorted(state.keys())
        return {"current_phase": "spy"}

    builder = _builder(None)
    builder._resolve_node_function_from_def = lambda node_def: spy  # type: ignore[method-assign]
    dag = builder.build(_spec(_node(
        "task_id_node", node_id="spy",
        context_policy=ContextPolicy(include_fields=["original_question"]),
    )))

    await dag.nodes["spy"]({
        "task_id": "task-42", "original_question": "q", "secret": "hidden",
    })

    assert seen["keys"] == ["original_question"]


@pytest.mark.asyncio
async def test_legacy_single_param_nodes_still_pass_through():
    async def legacy(state):
        return {"current_phase": "legacy", "keys": sorted(state.keys())}

    builder = _builder(None)
    builder._resolve_node_function_from_def = lambda node_def: legacy  # type: ignore[method-assign]
    dag = builder.build(_spec(_node(
        "task_id_node", node_id="legacy",
        context_policy=ContextPolicy(include_fields=["original_question"]),
    )))

    result = await dag.nodes["legacy"]({"task_id": "t", "original_question": "q"})

    assert result["keys"] == ["original_question"]
