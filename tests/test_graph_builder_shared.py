"""Integration tests for DynamicGraphBuilder with NodeDefinition."""
import pytest

from agent_core.models.pipeline_spec import (
    ContextPolicy,
    NodeDefinition,
    PhaseSpec,
    PipelineSpec,
    TransitionSpec,
)
from agent_core.runtime.dag.graph_builder import DynamicGraphBuilder


async def echo_node(state):
    """Returns all keys it received."""
    return {"received_keys": sorted(state.keys()), "current_phase": "echo"}


async def adder_node(state):
    return {"value": state.get("value", 0) + 1, "current_phase": "adder"}


class TestBuildWithNodeDefinition:
    def test_build_from_nodes(self):
        spec = PipelineSpec(
            pipeline_id="test_nd",
            nodes=[NodeDefinition(
                node_id="echo", role_id="researcher",
                node_function="tests.test_graph_builder_shared.echo_node",
                context_policy=ContextPolicy(
                    include_fields=["original_question"],
                ),
            )],
            transitions=[
                TransitionSpec(from_phase="echo", to_phase="__END__"),
            ],
            entry_point="echo",
            terminal_nodes=["echo"],
        )
        builder = DynamicGraphBuilder()
        dag = builder.build(spec)
        assert "echo" in dag.nodes

    @pytest.mark.asyncio
    async def test_context_filter_applied(self):
        spec = PipelineSpec(
            pipeline_id="test_filter",
            nodes=[NodeDefinition(
                node_id="echo", role_id="researcher",
                node_function="tests.test_graph_builder_shared.echo_node",
                context_policy=ContextPolicy(
                    include_fields=["original_question", "task_id"],
                ),
            )],
            transitions=[
                TransitionSpec(from_phase="echo", to_phase="__END__"),
            ],
            entry_point="echo",
            terminal_nodes=["echo"],
        )
        builder = DynamicGraphBuilder()
        runner = builder.compile(spec)
        results = {}
        async for chunk in runner.astream(
            {
                "original_question": "test?",
                "task_id": "t1",
                "secret_field": "should_not_see",
            },
            config={"configurable": {"thread_id": "test-1"}},
        ):
            results.update(chunk)
        received = results.get("echo", {}).get("received_keys", [])
        assert "original_question" in received
        assert "task_id" in received
        assert "secret_field" not in received

    def test_backward_compat_with_phases(self):
        spec = PipelineSpec(
            pipeline_id="test_old",
            phases=[PhaseSpec(
                phase_id="adder", role_id="researcher",
                node_function="tests.test_graph_builder_shared.adder_node",
                input_fields=["value"], output_fields=["value"],
            )],
            transitions=[
                TransitionSpec(from_phase="adder", to_phase="__END__"),
            ],
            entry_point="adder",
            terminal_phases=["adder"],
        )
        builder = DynamicGraphBuilder()
        dag = builder.build(spec)
        assert "adder" in dag.nodes


# ── Identity-only node context ────────────────────────────────────────────


async def identity_ctx_node(state, ctx):
    """Uses only what the default context actually provides."""
    return {"seen": [ctx.node_id, ctx.role_id, ctx.task_id]}


async def llm_ctx_node(state, ctx):
    """Reaches for a capability the identity-only default lacks."""
    return {"answer": await ctx.call_llm([{"role": "human", "content": "hi"}])}


class _RichContext:
    """Stand-in for a host-supplied context with LLM access."""

    def __init__(self, *, node_def, task_id_getter):
        self._node_def = node_def
        self._task_id_getter = task_id_getter

    @property
    def node_id(self) -> str:
        return self._node_def.node_id

    @property
    def role_id(self) -> str:
        return self._node_def.role_id

    @property
    def task_id(self) -> str:
        return self._task_id_getter()

    async def call_llm(self, messages, **kwargs) -> str:
        return "from-host-llm"


class TestNodeContextInjection:
    @staticmethod
    def _spec(fn_name: str) -> PipelineSpec:
        return PipelineSpec(
            pipeline_id=f"ctx_{fn_name}",
            nodes=[NodeDefinition(
                node_id="work", role_id="researcher",
                node_function=f"tests.test_graph_builder_shared.{fn_name}",
            )],
            transitions=[
                TransitionSpec(from_phase="work", to_phase="__END__"),
            ],
            entry_point="work",
            terminal_nodes=["work"],
        )

    @pytest.mark.asyncio
    async def test_default_context_supplies_identity(self):
        dag = DynamicGraphBuilder().build(self._spec("identity_ctx_node"))
        out = await dag.nodes["work"]({"task_id": "task-9"})
        assert out["seen"] == ["work", "researcher", "task-9"]

    @pytest.mark.asyncio
    async def test_default_context_names_the_factory_seam(self):
        """A missing capability points at ``node_context_factory``.

        The identity-only default cannot serve ``ctx.call_llm``; the error
        has to say where a context that can comes from, rather than a bare
        "'NodeContext' object has no attribute 'call_llm'".
        """
        dag = DynamicGraphBuilder().build(self._spec("llm_ctx_node"))
        with pytest.raises(AttributeError) as exc:
            await dag.nodes["work"]({"task_id": "task-9"})
        message = str(exc.value)
        assert "call_llm" in message
        assert "node_context_factory" in message
        assert "'work'" in message and "'researcher'" in message

    @pytest.mark.asyncio
    async def test_injected_factory_serves_richer_contract(self):
        dag = DynamicGraphBuilder(
            node_context_factory=_RichContext,
        ).build(self._spec("llm_ctx_node"))
        out = await dag.nodes["work"]({"task_id": "task-9"})
        assert out["answer"] == "from-host-llm"
