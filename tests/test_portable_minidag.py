"""Tests for MiniDAG engine — topology, execution, state merging, routing."""

import operator
from typing import Annotated, TypedDict

import pytest

from agent_core.runtime.dag.minidag import MiniDAG, extract_reducers

# ── Helper nodes ─────────────────────────────────────────────────────────


async def node_a(state):
    return {"phase": "a", "value": state.get("value", 0) + 1}


async def node_b(state):
    return {"phase": "b", "value": state.get("value", 0) + 10}


async def node_c(state):
    return {"phase": "c", "value": state.get("value", 0) + 100}


async def node_append(state):
    return {"items": ["new_item"], "phase": "append"}


async def node_failing(state):
    raise RuntimeError("intentional failure")


# ── Topology validation ─────────────────────────────────────────────────


class TestMiniDAGValidation:
    def test_valid_linear(self):
        dag = MiniDAG()
        dag.add_node("a", node_a)
        dag.add_node("b", node_b)
        dag.set_entry_point("a")
        dag.add_edge("a", "b")
        dag.add_edge("b", None)
        assert dag.validate() == []

    def test_missing_entry_point(self):
        dag = MiniDAG()
        dag.add_node("a", node_a)
        errors = dag.validate()
        assert any("entry point" in e.lower() for e in errors)

    def test_entry_point_not_in_nodes(self):
        dag = MiniDAG()
        dag.add_node("a", node_a)
        dag.set_entry_point("missing")
        errors = dag.validate()
        assert any("missing" in e for e in errors)

    def test_edge_to_unknown_node(self):
        dag = MiniDAG()
        dag.add_node("a", node_a)
        dag.set_entry_point("a")
        dag.add_edge("a", "nonexistent")
        errors = dag.validate()
        assert any("nonexistent" in e for e in errors)

    def test_unreachable_node(self):
        dag = MiniDAG()
        dag.add_node("a", node_a)
        dag.add_node("b", node_b)
        dag.add_node("orphan", node_c)
        dag.set_entry_point("a")
        dag.add_edge("a", "b")
        dag.add_edge("b", None)
        errors = dag.validate()
        assert any("orphan" in e for e in errors)

    def test_compile_fails_on_invalid(self):
        dag = MiniDAG()
        with pytest.raises(ValueError, match="Invalid DAG"):
            dag.compile()


# ── Linear execution ────────────────────────────────────────────────────


class TestLinearExecution:
    @pytest.mark.asyncio
    async def test_two_node_linear(self):
        dag = MiniDAG()
        dag.add_node("a", node_a)
        dag.add_node("b", node_b)
        dag.set_entry_point("a")
        dag.add_edge("a", "b")
        dag.add_edge("b", None)

        runner = dag.compile()
        chunks = []
        async for chunk in runner.astream({"value": 0}):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert "a" in chunks[0]
        assert "b" in chunks[1]
        # State accumulates: 0 + 1 = 1 (after a), 1 + 10 = 11 (after b)
        assert chunks[1]["b"]["value"] == 11

    @pytest.mark.asyncio
    async def test_three_node_linear(self):
        dag = MiniDAG()
        dag.add_node("a", node_a)
        dag.add_node("b", node_b)
        dag.add_node("c", node_c)
        dag.set_entry_point("a")
        dag.add_edge("a", "b")
        dag.add_edge("b", "c")
        dag.add_edge("c", None)

        runner = dag.compile()
        chunks = []
        async for chunk in runner.astream({"value": 0}):
            chunks.append(chunk)

        assert len(chunks) == 3
        # 0 + 1 = 1, 1 + 10 = 11, 11 + 100 = 111
        assert chunks[2]["c"]["value"] == 111

    @pytest.mark.asyncio
    async def test_single_node(self):
        dag = MiniDAG()
        dag.add_node("only", node_a)
        dag.set_entry_point("only")
        dag.add_edge("only", None)

        runner = dag.compile()
        chunks = []
        async for chunk in runner.astream({}):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0]["only"]["value"] == 1


# ── Conditional routing ─────────────────────────────────────────────────


class TestConditionalRouting:
    @pytest.mark.asyncio
    async def test_condition_routes_to_b(self):
        def route(state):
            return "go_b" if state.get("value", 0) < 5 else "go_c"

        dag = MiniDAG()
        dag.add_node("a", node_a)
        dag.add_node("b", node_b)
        dag.add_node("c", node_c)
        dag.set_entry_point("a")
        dag.add_conditional_edges("a", route, {"go_b": "b", "go_c": "c"})
        dag.add_edge("b", None)
        dag.add_edge("c", None)

        runner = dag.compile()
        chunks = []
        async for chunk in runner.astream({"value": 0}):
            chunks.append(chunk)

        # value after a = 1, which is < 5, so goes to b
        assert len(chunks) == 2
        assert "b" in chunks[1]

    @pytest.mark.asyncio
    async def test_condition_routes_to_c(self):
        def route(state):
            return "go_b" if state.get("value", 0) < 5 else "go_c"

        dag = MiniDAG()
        dag.add_node("a", node_a)
        dag.add_node("b", node_b)
        dag.add_node("c", node_c)
        dag.set_entry_point("a")
        dag.add_conditional_edges("a", route, {"go_b": "b", "go_c": "c"})
        dag.add_edge("b", None)
        dag.add_edge("c", None)

        runner = dag.compile()
        chunks = []
        async for chunk in runner.astream({"value": 10}):
            chunks.append(chunk)

        # value after a = 11, which is >= 5, so goes to c
        assert len(chunks) == 2
        assert "c" in chunks[1]

    @pytest.mark.asyncio
    async def test_condition_routes_to_end(self):
        def route(state):
            return "end" if state.get("done") else "continue"

        dag = MiniDAG()
        dag.add_node("a", node_a)
        dag.add_node("b", node_b)
        dag.set_entry_point("a")
        dag.add_conditional_edges("a", route, {"end": None, "continue": "b"})
        dag.add_edge("b", None)

        runner = dag.compile()
        chunks = []
        async for chunk in runner.astream({"done": True}):
            chunks.append(chunk)

        # Should stop after a (routed to END)
        assert len(chunks) == 1


# ── Loop (verification gate pattern) ────────────────────────────────────


class TestLoopRouting:
    @pytest.mark.asyncio
    async def test_loop_back_then_proceed(self):
        """Simulates verify → react_solve loop → verify → report."""
        call_count = {"verify": 0}

        async def verify_node(state):
            call_count["verify"] += 1
            return {"verified": call_count["verify"] >= 2, "phase": "verify"}

        async def solve_node(state):
            return {"evidence": True, "phase": "solve"}

        async def report_node(state):
            return {"report": "done", "phase": "report"}

        def gate(state):
            return "report" if state.get("verified") else "solve"

        dag = MiniDAG()
        dag.add_node("solve", solve_node)
        dag.add_node("verify", verify_node)
        dag.add_node("report", report_node)
        dag.set_entry_point("solve")
        dag.add_edge("solve", "verify")
        dag.add_conditional_edges("verify", gate, {"report": "report", "solve": "solve"})
        dag.add_edge("report", None)

        runner = dag.compile()
        chunks = []
        async for chunk in runner.astream({}):
            chunks.append(chunk)

        # solve → verify (not verified) → solve → verify (verified) → report
        node_sequence = [next(iter(c)) for c in chunks]
        assert node_sequence == ["solve", "verify", "solve", "verify", "report"]


# ── State merging (reducers) ────────────────────────────────────────────


class TestStateMerging:
    @pytest.mark.asyncio
    async def test_last_write_wins(self):
        dag = MiniDAG()
        dag.add_node("a", node_a)
        dag.add_node("b", node_b)
        dag.set_entry_point("a")
        dag.add_edge("a", "b")
        dag.add_edge("b", None)

        runner = dag.compile()
        chunks = []
        async for chunk in runner.astream({"value": 0}):
            chunks.append(chunk)

        # "phase" is last-write-wins: a sets "a", b sets "b"
        state = await runner.aget_state({"configurable": {"thread_id": "default"}})
        assert state.values["phase"] == "b"

    @pytest.mark.asyncio
    async def test_reducer_appends(self):
        dag = MiniDAG()
        dag.add_node("a", node_append)
        dag.add_node("b", node_append)
        dag.set_entry_point("a")
        dag.add_edge("a", "b")
        dag.add_edge("b", None)
        dag.set_reducers({"items": operator.add})

        runner = dag.compile()
        async for _ in runner.astream({"items": ["initial"]}):
            pass

        state = await runner.aget_state({"configurable": {"thread_id": "default"}})
        # initial + new_item (from a) + new_item (from b)
        assert state.values["items"] == ["initial", "new_item", "new_item"]


# ── extract_reducers ────────────────────────────────────────────────────


class TestExtractReducers:
    def test_extract_from_research_state(self):
        class PortableState(TypedDict):
            evidence_cards: Annotated[list[str], operator.add]
            messages: Annotated[list[str], operator.add]
            errors: Annotated[list[str], operator.add]
            react_steps: Annotated[list[str], operator.add]
            task_id: str
            original_question: str

        reducers = extract_reducers(PortableState)
        # Should find operator.add for these fields
        assert "evidence_cards" in reducers
        assert "messages" in reducers
        assert "errors" in reducers
        assert "react_steps" in reducers
        # Regular fields should NOT be in reducers
        assert "task_id" not in reducers
        assert "original_question" not in reducers

    def test_extract_from_plain_dict(self):
        reducers = extract_reducers(dict)
        assert reducers == {}


# ── aget_state / aget_state_history ─────────────────────────────────────


class TestStateAccess:
    @pytest.mark.asyncio
    async def test_aget_state_returns_latest(self):
        dag = MiniDAG()
        dag.add_node("a", node_a)
        dag.set_entry_point("a")
        dag.add_edge("a", None)

        runner = dag.compile()
        async for _ in runner.astream({"value": 5}, config={"configurable": {"thread_id": "t1"}}):
            pass

        state = await runner.aget_state({"configurable": {"thread_id": "t1"}})
        assert state.values["value"] == 6  # 5 + 1

    @pytest.mark.asyncio
    async def test_aget_state_empty_for_unknown_thread(self):
        dag = MiniDAG()
        dag.add_node("a", node_a)
        dag.set_entry_point("a")
        dag.add_edge("a", None)

        runner = dag.compile()
        state = await runner.aget_state({"configurable": {"thread_id": "nonexistent"}})
        assert state.values == {}


# ── Error handling ──────────────────────────────────────────────────────


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_node_failure_propagates(self):
        dag = MiniDAG()
        dag.add_node("a", node_a)
        dag.add_node("fail", node_failing)
        dag.set_entry_point("a")
        dag.add_edge("a", "fail")
        dag.add_edge("fail", None)

        runner = dag.compile()
        chunks = []
        with pytest.raises(RuntimeError, match="intentional failure"):
            async for chunk in runner.astream({}):
                chunks.append(chunk)

        # First node should have completed before failure
        assert len(chunks) == 1
        assert "a" in chunks[0]
