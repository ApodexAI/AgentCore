"""loop-v1 §4 — the three merge rule sets.

Pure functions, no LLM: whatever else drifts between the two repos, these
rules decide what an observer can actually make the loop do, so they are the
cheapest high-value thing to pin.
"""

from __future__ import annotations

import asyncio

import pytest

import agent_core.loop_types as lt

Intervention = lt.Intervention
ToolCallIntervention = lt.ToolCallIntervention
ToolResult = lt.ToolResult
merge_interventions = lt.merge_interventions


def _tool_result(name: str = "t", result: str = "r"):
    return ToolResult(
        name=name,
        args={},
        result=result,
        duration_ms=0,
        tool_call_id="c1",
        is_error=False,
    )


# --- merge_interventions (§4.2) -------------------------------------------


def test_inject_messages_concatenate_in_observer_order():
    merged = merge_interventions(
        [
            Intervention(inject_messages=["a", "b"]),
            Intervention(inject_messages=None),
            Intervention(inject_messages=["c"]),
        ]
    )
    assert merged.inject_messages == ["a", "b", "c"]


def test_no_inject_messages_stays_none_not_empty_list():
    """``None`` and ``[]`` must not be conflated: the loop treats a list as
    'inject this', so an empty list would be a request to inject nothing."""
    assert merge_interventions([Intervention(), Intervention()]).inject_messages is None


def test_stop_reason_first_non_none_wins():
    merged = merge_interventions(
        [
            Intervention(),
            Intervention(stop_reason="first"),
            Intervention(stop_reason="second"),
        ]
    )
    assert merged.stop_reason == "first"


@pytest.mark.parametrize(
    "flag",
    ["skip_tool_execution", "pop_last_message", "continue_to_next_turn"],
)
def test_boolean_flags_are_any_true(flag: str):
    merged = merge_interventions(
        [
            Intervention(),
            Intervention(**{flag: True}),
            Intervention(),
        ]
    )
    assert getattr(merged, flag) is True
    assert getattr(merge_interventions([Intervention()] * 3), flag) is False


def test_empty_merge_is_the_neutral_intervention():
    merged = merge_interventions([])
    assert merged.inject_messages is None
    assert merged.stop_reason is None
    assert not merged.skip_tool_execution
    assert not merged.pop_last_message
    assert not merged.continue_to_next_turn


# --- notify_tool_call (§4.3) ----------------------------------------------


class _ToolCallObserver:
    critical = True

    def __init__(self, iv):
        self._iv = iv

    async def on_tool_call(self, ctx, tool_call):
        return self._iv


@pytest.mark.asyncio
async def test_rewrite_args_last_writer_wins():
    merged = await lt.notify_tool_call(
        [
            _ToolCallObserver(ToolCallIntervention(rewrite_args={"v": 1})),
            _ToolCallObserver(ToolCallIntervention(rewrite_args={"v": 2})),
        ],
        None,
        {"name": "t"},
    )
    assert merged.rewrite_args == {"v": 2}


@pytest.mark.asyncio
async def test_skip_with_result_first_writer_wins():
    """Opposite of ``rewrite_args`` on purpose: once a call is skipped, a
    later rewrite of its arguments would have nothing to apply to."""
    merged = await lt.notify_tool_call(
        [
            _ToolCallObserver(ToolCallIntervention(skip_with_result="first")),
            _ToolCallObserver(ToolCallIntervention(skip_with_result="second")),
        ],
        None,
        {"name": "t"},
    )
    assert merged.skip_with_result == "first"


@pytest.mark.asyncio
async def test_metadata_updates_merge_dict_wise():
    merged = await lt.notify_tool_call(
        [
            _ToolCallObserver(ToolCallIntervention(metadata_updates={"a": 1})),
            _ToolCallObserver(ToolCallIntervention(metadata_updates={"b": 2})),
        ],
        None,
        {"name": "t"},
    )
    assert merged.metadata_updates == {"a": 1, "b": 2}


@pytest.mark.asyncio
async def test_tool_call_observer_crash_cannot_break_dispatch():
    """§3.4 — a buggy observer must never crash a tool dispatch."""

    class _Boom:
        critical = True

        async def on_tool_call(self, ctx, tool_call):
            raise RuntimeError("observer bug")

    merged = await lt.notify_tool_call(
        [_Boom(), _ToolCallObserver(ToolCallIntervention(rewrite_args={"v": 9}))],
        None,
        {"name": "t"},
    )
    assert merged.rewrite_args == {"v": 9}


# --- notify_tool_result (§4.4) --------------------------------------------


@pytest.mark.asyncio
async def test_tool_result_mutation_chains_through_observers():
    """Each non-None return replaces what the next observer sees."""

    class _Appender:
        critical = True

        def __init__(self, suffix: str):
            self.suffix = suffix
            self.seen: list[str] = []

        async def on_tool_result(self, ctx, result):
            self.seen.append(result.result)
            return _tool_result(result=result.result + self.suffix)

    a, b = _Appender("-a"), _Appender("-b")
    out = await lt.notify_tool_result([a, b], None, _tool_result(result="base"))
    assert out.result == "base-a-b"
    assert a.seen == ["base"], "first observer sees the original"
    assert b.seen == ["base-a"], "second observer sees the first's mutation"


@pytest.mark.asyncio
async def test_tool_result_none_return_is_read_only():
    class _Reader:
        critical = True

        async def on_tool_result(self, ctx, result):
            return None

    out = await lt.notify_tool_result([_Reader()], None, _tool_result(result="base"))
    assert out.result == "base"


@pytest.mark.asyncio
async def test_observer_without_the_hook_is_skipped_not_an_error():
    """Hooks are probed, so an observer implementing only some of them is
    a first-class citizen (this is what keeps optional hooks optional)."""

    class _Bare:
        critical = True

    out = await lt.notify_tool_result([_Bare()], None, _tool_result(result="base"))
    assert out.result == "base"


def test_deadline_accepts_structural_lease_and_absolute_time(monkeypatch) -> None:
    class Lease:
        def remaining_s(self) -> float:
            return 12.5

    metadata = {lt.WALL_DEADLINE_MONOTONIC_KEY: Lease()}
    assert lt.deadline_remaining_s(metadata) == 12.5

    metadata[lt.WALL_DEADLINE_MONOTONIC_KEY] = 125.0
    monkeypatch.setattr(lt.time, "monotonic", lambda: 100.0)
    assert lt.deadline_remaining_s(metadata) == 25.0


def test_deadline_rejects_bad_shapes_and_failing_lease() -> None:
    class BrokenLease:
        def remaining_s(self) -> float:
            raise RuntimeError("expired backing store")

    key = lt.WALL_DEADLINE_MONOTONIC_KEY
    assert lt.deadline_remaining_s({key: BrokenLease()}) is None
    assert lt.deadline_remaining_s({key: object()}) is None
    assert lt.deadline_remaining_s(None) is None


@pytest.mark.asyncio
async def test_passive_observer_is_non_blocking_and_return_is_ignored() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class Passive(lt.BaseObserver):
        async def on_llm_response(self, ctx):
            entered.set()
            await release.wait()
            return lt.Intervention(stop_reason="must-be-ignored")

    interventions = await lt.notify_observers([Passive()], "on_llm_response", None)
    assert interventions == []
    await asyncio.wait_for(entered.wait(), timeout=1)
    release.set()
    await lt.drain_background_observers()


def test_legacy_observer_satisfies_runtime_protocol() -> None:
    class Legacy:
        critical = True

        async def on_loop_start(self, config): ...
        async def on_llm_delta(self, ctx): ...
        async def on_llm_attempt(self, ctx): ...
        async def on_llm_response(self, ctx): ...
        async def on_tool_call(self, ctx, tool_call): ...
        async def on_tool_result(self, ctx, result): ...
        async def on_turn_end(self, ctx): ...
        async def on_loop_end(self, result): ...

    assert isinstance(Legacy(), lt.LoopObserver)
