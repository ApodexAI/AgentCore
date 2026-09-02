"""DynamicGraphBuilder — builds MiniDAG from PipelineSpec."""

from __future__ import annotations

import importlib
import inspect
import logging
from collections.abc import Callable
from typing import Any, Protocol, cast

from agent_core.models.pipeline_spec import (
    CompressionConfig,
    ContextPolicy,
    NodeDefinition,
    PipelineSpec,
    TransitionSpec,
)
from agent_core.protocols import (
    LLMResourceProvider,
    PhaseContext,
    PhaseMiddlewareChain,
    SubAgentProfileRegistry,
)
from agent_core.runtime.dag.base_state import BaseTaskState
from agent_core.runtime.dag.minidag import MiniDAG, MiniDAGRunner, extract_reducers

logger = logging.getLogger(__name__)


class NodeContextFactory(Protocol):
    def __call__(
        self,
        *,
        node_def: NodeDefinition,
        task_id_getter: Callable[[], str],
    ) -> object: ...


class DynamicGraphBuilder:
    """Builds a MiniDAG from a declarative PipelineSpec.

    Supports middleware wrapping: if a MiddlewareChain is registered in
    the service registry, all node functions are wrapped with before/after/error
    hooks at graph construction time (not at stream-processing time).
    """

    def __init__(self, *, node_context_factory: NodeContextFactory | None = None) -> None:
        """Construct the builder.

        ``node_context_factory`` builds the object injected as the second
        argument of every ``(state, ctx)`` node. The default is
        :class:`~agent_core.models.node_context.NodeContext`, which carries
        identity only — hosts whose nodes call ``ctx.call_llm`` /
        ``ctx.call_tool`` / ``ctx.run_agent_loop`` must pass a factory for a
        context that provides those; the identity-only default raises an
        AttributeError naming this parameter if they don't.
        """
        if node_context_factory is None:
            from agent_core.models.node_context import DefaultNodeContext

            node_context_factory = DefaultNodeContext
        self._node_context_factory = node_context_factory

    def build(self, spec: PipelineSpec, state_type: type | None = None) -> MiniDAG:
        """Build (but do not compile) a MiniDAG from PipelineSpec.

        Args:
            spec: The pipeline specification.
            state_type: Optional TypedDict class for the state. If None, uses dict.
        """
        dag = MiniDAG()
        if state_type:
            dag.set_reducers(extract_reducers(state_type))

        middleware_chain = self._get_middleware_chain()

        # Use resolved_nodes (works with both nodes and phases via bridge)
        for node_def in spec.resolved_nodes:
            node_fn = self._resolve_node_function_from_def(node_def)
            # Decided from the original signature: every wrapper below takes
            # ``(state, ...)`` and would fool the arity probe.
            needs_ctx = _needs_context(node_fn)
            # Context filter goes innermost so the NodeContext built outside it
            # still sees the *unfiltered* state. ``task_id_getter`` reads
            # ``state["task_id"]``, which a ``ContextPolicy.include_fields``
            # list normally omits — wrapping the other way round gave those
            # nodes ``ctx.task_id == ""`` and attributed every task-scoped
            # lookup and telemetry emission to the empty task id.
            node_fn = _wrap_with_context_filter(node_def, node_fn)
            node_fn = _wrap_with_node_context(
                node_def, node_fn, self._node_context_factory,
                needs_context=needs_ctx,
            )
            # Wrap with field truncation
            if node_def.compression and node_def.compression.max_field_tokens:
                node_fn = _wrap_with_field_truncation(node_def, node_fn)
            if middleware_chain:
                node_fn = _wrap_with_middleware_nd(
                    node_def, node_fn, middleware_chain,
                )
            dag.add_node(node_def.node_id, node_fn)

            # Register sub-agent profiles with whichever optional component
            # implements the core Protocol. ``core/runtime`` must not import
            # AgentBus directly.
            if node_def.sub_agent_profiles:
                from agent_core.runtime.registries import services as kernel_registry
                profile_registry = kernel_registry.get_optional(
                    SubAgentProfileRegistry,
                )
                if profile_registry is None:
                    logger.warning(
                        "Node '%s' declares sub_agent_profiles but no "
                        "SubAgentProfileRegistry is registered; profiles dropped",
                        node_def.node_id,
                    )
                else:
                    profile_registry.register_sub_agent_profiles(
                        node_def.node_id,
                        node_def.sub_agent_profiles,
                    )

        dag.set_entry_point(spec.entry_point)

        # Transitions — unchanged logic
        from_groups: dict[str, list[TransitionSpec]] = {}
        for t in spec.transitions:
            from_groups.setdefault(t.from_phase, []).append(t)

        for from_phase, transitions in from_groups.items():
            has_conditions = any(t.condition for t in transitions)

            if has_conditions:
                condition_fn = self._resolve_condition(transitions)
                route_map: dict[str, str | None] = {}
                for t in transitions:
                    target = None if t.to_phase == "__END__" else t.to_phase
                    route_map[t.to_phase] = target
                dag.add_conditional_edges(from_phase, condition_fn, route_map)
            else:
                if len(transitions) != 1:
                    logger.warning(
                        "Multiple unconditional transitions from '%s', using first",
                        from_phase,
                    )
                target = (
                    None if transitions[0].to_phase == "__END__"
                    else transitions[0].to_phase
                )
                dag.add_edge(from_phase, target)

        for tp in spec.resolved_terminal_nodes:
            if tp not in from_groups:
                dag.add_edge(tp, None)

        return dag

    def compile(
        self,
        spec: PipelineSpec,
        state_type: type | None = None,
        checkpointer: Any = None,
    ) -> MiniDAGRunner:
        """Build and compile in one step."""
        dag = self.build(spec, state_type)
        return dag.compile(checkpointer=checkpointer)

    def _resolve_node_function_from_def(self, node_def: NodeDefinition) -> Any:
        """Import and return the node function for a NodeDefinition."""
        if node_def.node_function:
            return _import_dotted(node_def.node_function)
        return _make_generic_executor_nd(node_def)

    def _resolve_condition(self, transitions: list[TransitionSpec]) -> Any:
        """Find and return the condition function from conditional transitions."""
        for t in transitions:
            if t.condition:
                return _import_dotted(t.condition)
        raise ValueError("No condition function found in conditional transitions")

    def _get_middleware_chain(self) -> Any | None:
        """Get MiddlewareChain from registry if available."""
        from agent_core.runtime.registries import services as kernel_registry
        return kernel_registry.get_optional(PhaseMiddlewareChain)


def _needs_context(fn: Any) -> bool:
    """Check if a node function accepts a ctx parameter (2+ params)."""
    try:
        sig = inspect.signature(fn)
        return len(list(sig.parameters.keys())) >= 2
    except (ValueError, TypeError):
        return False


def _wrap_with_node_context(
    node_def: NodeDefinition,
    fn: Any,
    node_context_factory: NodeContextFactory,
    *,
    needs_context: bool | None = None,
) -> Any:
    """Wrap node function to inject the node context as second arg.

    The context comes from ``node_context_factory`` — identity-only by
    default; see :meth:`DynamicGraphBuilder.__init__`.

    If the function has a single-param ``(state)`` signature (legacy),
    it passes through unchanged — backward compatible with solver specs
    that predate the NodeContext facade (e.g. solvers/gaia).

    ``needs_context`` lets the caller decide from the *original* node
    signature. :meth:`DynamicGraphBuilder.build` must pass it, because it
    applies this wrapper on top of the context filter, whose own
    ``(state, *extra)`` signature says nothing about what the node wants.
    """
    if not (needs_context if needs_context is not None else _needs_context(fn)):
        return fn  # legacy (state) signature — no injection

    async def wrapped(state: dict[str, Any]) -> dict[str, Any]:
        ctx = node_context_factory(
            node_def=node_def,
            task_id_getter=lambda: state.get("task_id", ""),
        )
        return await fn(state, ctx)

    wrapped.__name__ = getattr(
        fn, "__name__", f"node_{node_def.node_id}",
    )
    wrapped.__qualname__ = getattr(
        fn, "__qualname__", f"node_{node_def.node_id}",
    )
    return wrapped


def apply_context_filter(policy: ContextPolicy, state: dict[str, Any]) -> dict[str, Any]:
    """Filter pipeline state according to a ContextPolicy."""
    if policy.filter_fn:
        custom_filter = _import_dotted(policy.filter_fn)
        return custom_filter(state)
    if policy.include_fields is None:
        return state
    allowed = set(policy.include_fields) | set(policy.inject_fields)
    return {k: v for k, v in state.items() if k in allowed}


def apply_field_truncation(
    state: dict[str, Any], config: CompressionConfig | None
) -> dict[str, Any]:
    """Truncate specific state fields based on CompressionConfig.max_field_tokens."""
    if not config or not config.max_field_tokens:
        return state
    result = dict(state)
    for field, max_tokens in config.max_field_tokens.items():
        if field not in result:
            continue
        value = result[field]
        if isinstance(value, str):
            max_chars = max_tokens * 3  # 1 token ≈ 3 chars conservative
            if len(value) > max_chars:
                result[field] = value[:max_chars] + "\n...[truncated]"
        elif isinstance(value, list):
            items = cast("list[Any]", value)
            if not items:
                continue
            avg_item_chars = sum(len(str(item)) for item in items[:5]) / min(
                len(items), 5
            )
            avg_item_tokens = max(avg_item_chars / 3, 1)
            max_items = max(int(max_tokens / avg_item_tokens), 1)
            if len(items) > max_items:
                result[field] = items[:max_items]
    return result


def _wrap_with_context_filter(
    node_def: NodeDefinition, node_fn: Any,
) -> Any:
    """Wrap node_fn to filter state via ContextPolicy before calling."""
    policy = node_def.context_policy
    if policy.filter_fn is None and policy.include_fields is None:
        return node_fn  # full state, skip wrapping

    async def wrapped(
        state: dict[str, Any], *extra: Any,
    ) -> dict[str, Any]:
        filtered = apply_context_filter(policy, state)
        return await node_fn(filtered, *extra)

    wrapped.__name__ = getattr(
        node_fn, "__name__", f"node_{node_def.node_id}",
    )
    wrapped.__qualname__ = getattr(
        node_fn, "__qualname__", f"node_{node_def.node_id}",
    )
    return wrapped


def _wrap_with_field_truncation(
    node_def: NodeDefinition, node_fn: Any,
) -> Any:
    """Wrap node_fn to truncate oversized state fields."""
    config = node_def.compression

    async def wrapped(state: dict[str, Any]) -> dict[str, Any]:
        truncated = apply_field_truncation(state, config)
        return await node_fn(truncated)

    wrapped.__name__ = getattr(
        node_fn, "__name__", f"node_{node_def.node_id}",
    )
    wrapped.__qualname__ = getattr(
        node_fn, "__qualname__", f"node_{node_def.node_id}",
    )
    return wrapped


def _wrap_with_middleware_nd(
    node_def: NodeDefinition,
    node_fn: Any,
    middleware_chain: Any,
) -> Any:
    """Wrap a node function with middleware before/after/error hooks.

    Injects compression config into ExecutionScope metadata.
    Called at graph construction time so middleware hooks execute
    inside the compiled node wrapper.
    """
    from agent_core.execution_context import (
        build_execution_scope,
        ensure_trace_metadata,
        reset_current_execution_scope,
        set_current_execution_scope,
    )
    async def wrapped(state: dict[str, Any]) -> dict[str, Any]:
        scope = build_execution_scope(
            task_id=state.get("task_id", ""),
            phase_id=node_def.node_id,
            role_id=node_def.role_id,
            state=state,
        )
        ensure_trace_metadata(
            scope.metadata,
            default_step_id=f"{node_def.node_id}:phase",
            refresh_prompt_id=False,
        )
        if node_def.compression:
            scope.metadata["compression"] = (
                node_def.compression.model_dump()
            )
        ctx = PhaseContext(
            task_id=scope.task_id,
            phase_id=node_def.node_id,
            role_id=node_def.role_id,
            display_label=getattr(node_def, "display_label", "") or "",
            state=state,
            metadata=scope.metadata,
        )
        token = set_current_execution_scope(scope)

        try:
            ctx = await middleware_chain.run_before_phase(ctx)
            state["execution_context"] = dict(ctx.metadata)

            max_retries_value = node_def.metadata.get("max_retries", 0)
            max_retries = (
                max_retries_value
                if isinstance(max_retries_value, int) and not isinstance(max_retries_value, bool)
                else 0
            )
            last_error: Exception | None = None

            for attempt in range(1 + max_retries):
                try:
                    result = await node_fn(state)
                    # ``MiniDAGRunner`` treats a ``None`` delta as "no state
                    # change" (``if delta is None: delta = {}``), so a node is
                    # allowed to return it. Normalise here or the middleware
                    # hook below is handed None and ``result.get`` raises — a
                    # node that worked with no middleware registered would
                    # start failing the moment a chain was added, and the
                    # surrounding ``except`` would misreport the contract
                    # violation as a middleware error.
                    if result is None:
                        result = {}
                    result = await middleware_chain.run_after_phase(
                        ctx, result,
                    )
                    merged_execution_context = dict(
                        state.get("execution_context") or {},
                    )
                    merged_execution_context.update(ctx.metadata)
                    result_ctx = result.get("execution_context")
                    if isinstance(result_ctx, dict):
                        merged_execution_context.update(
                            cast("dict[str, Any]", result_ctx)
                        )
                    result["execution_context"] = (
                        merged_execution_context
                    )
                    return result
                except Exception as e:
                    last_error = e
                    err_result = await middleware_chain.run_on_error(
                        ctx, e,
                    )
                    # Retry on any failure while attempts remain. Gating this
                    # on ``err_result is None`` made retrying depend on a
                    # middleware *suppressing* the error, so with the default
                    # ``ExecutionMiddleware.on_error`` (which returns the
                    # error) a node declaring ``metadata["max_retries"]`` was
                    # never retried at all.
                    if attempt < max_retries:
                        logger.warning(
                            "Node '%s' failed, retrying (%d/%d): %s",
                            node_def.node_id,
                            attempt + 1,
                            max_retries,
                            e,
                        )
                        continue
                    if err_result is None:
                        # Middleware returned None = "handled, do not raise".
                        # The DAG then advances on a synthetic delta, so log
                        # the swallowed exception itself — otherwise a genuine
                        # node crash leaves no trace anywhere.
                        logger.warning(
                            "Middleware suppressed error in '%s' "
                            "(retries exhausted); advancing the DAG",
                            node_def.node_id,
                            exc_info=e,
                        )
                        return {
                            "current_phase": node_def.node_id,
                            "execution_context": dict(
                                state.get("execution_context")
                                or ctx.metadata
                            ),
                        }
                    raise err_result from e

            if last_error:
                raise last_error
            return {
                "current_phase": node_def.node_id,
                "execution_context": dict(
                    state.get("execution_context") or ctx.metadata
                ),
            }
        finally:
            reset_current_execution_scope(token)

    wrapped.__name__ = getattr(
        node_fn, "__name__", f"node_{node_def.node_id}",
    )
    wrapped.__qualname__ = getattr(
        node_fn, "__qualname__", f"node_{node_def.node_id}",
    )
    return wrapped


def _make_generic_executor_nd(node_def: NodeDefinition) -> Any:
    """Create a generic executor from NodeDefinition with prompt_template."""

    async def executor(state: dict[str, Any]) -> dict[str, Any]:
        import json

        from jinja2 import Template

        from agent_core.messages import system_msg, text_of, user_msg
        from agent_core.runtime.registries import services as kernel_registry
        from agent_core.runtime.registries.agents import AgentRegistry

        resource_mgr = kernel_registry.get_optional(LLMResourceProvider)
        if resource_mgr is None:
            raise RuntimeError("No LLMResourceProvider is registered")
        agent_reg = kernel_registry.get(AgentRegistry)

        template = Template(node_def.prompt_template)
        prompt = template.render(**state)
        system_prompt = agent_reg.get_prompt_for(node_def.role_id)

        llm = resource_mgr.get_llm(node_def.role_id)
        response = await llm.chat([
            system_msg(system_prompt),
            user_msg(prompt),
        ])

        content = text_of(response.content)

        result: dict[str, Any] = {"current_phase": node_def.node_id}
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                parsed = json.loads(content[start:end])
                if isinstance(parsed, dict):
                    for field in node_def.output_fields:
                        if field in parsed:
                            result[field] = parsed[field]
                    return result
        except (json.JSONDecodeError, ValueError):
            pass
        if node_def.output_fields:
            result[node_def.output_fields[0]] = content
        else:
            result["output"] = content
        return result

    executor.__name__ = f"node_{node_def.node_id}"
    executor.__qualname__ = f"node_{node_def.node_id}"
    return executor


def _import_dotted(path: str) -> Any:
    """Import a function from a dotted path like 'workflows.default_research.edges.should_continue_collecting'."""
    module_path, func_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def resolve_state_type(spec: PipelineSpec) -> type:
    """Import the state class a spec declares, or fall back to ``BaseTaskState``.

    The state type is a property of the workflow, so each spec declares its
    own ``state_type`` dotted path (e.g. research specs point at
    ``workflows.default_research.state.ResearchState`` for its extra fan-in
    fields). A spec that declares nothing gets ``BaseTaskState`` — the
    universal fields plus their fan-in reducers (errors / messages /
    agent_messages). Core stays generic — it never names a workflow here.
    """
    state_type = getattr(spec, "state_type", None)
    if not state_type:
        return BaseTaskState
    try:
        return _import_dotted(state_type)
    except Exception:
        logger.error(
            "spec %r declares state_type %r that failed to import",
            getattr(spec, "pipeline_id", "?"),
            state_type,
        )
        raise
