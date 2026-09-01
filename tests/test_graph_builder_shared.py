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
