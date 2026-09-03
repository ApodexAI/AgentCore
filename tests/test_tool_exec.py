from __future__ import annotations

import asyncio
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

import pytest

from agent_core.runtime.loop.tool_exec import ToolExecutionHooks, execute_tools


class FakeTool:
    def __init__(self, name: str, result: Any = "ok", *, delay: float = 0) -> None:
        self.name = name
        self.result = result
        self.delay = delay

    async def ainvoke(self, args: dict[str, Any]) -> Any:
        await asyncio.sleep(self.delay)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.mark.asyncio
async def test_parallel_batch_converts_success_unknown_and_failure() -> None:
    calls = [
        {"name": "ok", "args": {"x": 1}, "id": "a"},
        {"name": "missing", "args": {}, "id": "b"},
        {"name": "bad", "args": {}, "id": "c"},
    ]
    results = await execute_tools(
        calls,
        {"ok": FakeTool("ok", "done"), "bad": FakeTool("bad", ValueError("boom"))},
        timeout=1,
        turn=2,
        count_offset=4,
    )

    assert [result.tool_call_id for result in results] == ["a", "b", "c"]
    assert results[0].result == "done"
    assert results[0].is_error is False
    assert "unknown tool 'missing'" in results[1].result
    assert "bad, ok" in results[1].result
    assert results[1].is_error is True
    assert "boom" in results[2].result


@pytest.mark.asyncio
async def test_host_hooks_own_timeout_context_and_result_policies() -> None:
    events: list[str] = []

    @contextmanager
    def scope(call: dict[str, Any], timeout: float):
        events.append(f"enter:{call['id']}:{timeout}")
        try:
            yield
        finally:
            events.append("exit")

    async def await_call(awaitable, name, args, timeout):
        events.append(f"await:{name}:{args['x']}:{timeout}")
        return await awaitable

    hooks = ToolExecutionHooks(
        resolve_timeout=lambda _name, _args, configured: configured + 7,
        await_call=await_call,
        call_scope=scope,
        on_call=lambda name: events.append(f"meter:{name}"),
        transform_result=lambda name, value: f"{name}={value}",
        result_metadata=lambda _name, _raw, _rendered: {
            "error_kind": "command_exit",
            "result_id": "spill-1",
            "repeat_count": 2,
            "repeat_recovery_id": "spill-0",
        },
        transform_batch=lambda results: [replace(results[0], result="batch")],
    )
    results = await execute_tools(
        [{"name": "echo", "args": {"x": 3}, "id": "tc"}],
        {"echo": FakeTool("echo", "value")},
        timeout=5,
        turn=1,
        count_offset=0,
        hooks=hooks,
    )

    assert results[0].result == "batch"
    assert results[0].is_error is True
    assert results[0].error_kind == "command_exit"
    assert results[0].result_id == "spill-1"
    assert results[0].repeat_count == 2
    assert results[0].repeat_recovery_id == "spill-0"
    assert events == ["meter:echo", "enter:tc:12", "await:echo:3:12", "exit"]


@pytest.mark.asyncio
async def test_timeout_and_exception_have_structured_error_kinds() -> None:
    results = await execute_tools(
        [
            {"name": "slow", "args": {}, "id": "a"},
            {"name": "bad", "args": {}, "id": "b"},
        ],
        {
            "slow": FakeTool("slow", delay=1),
            "bad": FakeTool("bad", ValueError("boom")),
        },
        timeout=1,
        turn=1,
        count_offset=0,
        hooks=ToolExecutionHooks(
            resolve_timeout=lambda name, _args, _configured: (
                0.001 if name == "slow" else 1.0
            )
        ),
    )

    assert results[0].error_kind == "timeout"
    assert results[1].error_kind == "exception"


@pytest.mark.asyncio
async def test_fan_in_interrupt_cancels_invocation_and_returns_result() -> None:
    async def interrupt(_call: dict[str, Any]) -> bool:
        await asyncio.sleep(0)
        return True

    results = await execute_tools(
        [{"name": "collect_reports", "args": {}, "id": "fan"}],
        {"collect_reports": FakeTool("collect_reports", delay=10)},
        timeout=20,
        turn=1,
        count_offset=0,
        interrupt_waiter=interrupt,
    )

    assert results[0].interrupted is True
    assert results[0].result.startswith("[interrupted]")


@pytest.mark.asyncio
async def test_external_cancellation_is_not_converted_to_tool_error() -> None:
    task = asyncio.create_task(
        execute_tools(
            [{"name": "slow", "args": {}, "id": "slow"}],
            {"slow": FakeTool("slow", delay=10)},
            timeout=20,
            turn=1,
            count_offset=0,
        )
    )
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_waiter_returning_false_leaves_fan_in_tool_running() -> None:
    """A waiter resolving to ``False`` is not an interrupt.

    ``_wait_for_tool_interrupt`` returns ``False`` whenever no observer asked
    to interrupt, so racing it must not cancel the fan-in tool nor report the
    collected work as abandoned.
    """

    async def no_interrupt(_call: dict[str, Any]) -> bool:
        await asyncio.sleep(0)
        return False

    results = await execute_tools(
        [{"name": "collect_reports", "args": {}, "id": "fan"}],
        {"collect_reports": FakeTool("collect_reports", "gathered", delay=0.02)},
        timeout=20,
        turn=1,
        count_offset=0,
        interrupt_waiter=no_interrupt,
    )

    assert results[0].result == "gathered"
    assert results[0].interrupted is False
    assert results[0].is_error is False


@pytest.mark.asyncio
async def test_raising_waiter_neither_interrupts_nor_fails_the_tool() -> None:
    async def broken(_call: dict[str, Any]) -> bool:
        await asyncio.sleep(0)
        raise RuntimeError("observer exploded")

    results = await execute_tools(
        [{"name": "collect_reports", "args": {}, "id": "fan"}],
        {"collect_reports": FakeTool("collect_reports", "gathered", delay=0.02)},
        timeout=20,
        turn=1,
        count_offset=0,
        interrupt_waiter=broken,
    )

    assert results[0].result == "gathered"
    assert results[0].interrupted is False
    assert results[0].is_error is False


@pytest.mark.asyncio
async def test_real_interrupt_still_wins_over_a_slow_tool() -> None:
    async def interrupt(_call: dict[str, Any]) -> bool:
        await asyncio.sleep(0)
        return True

    results = await execute_tools(
        [{"name": "collect_reports", "args": {}, "id": "fan"}],
        {"collect_reports": FakeTool("collect_reports", delay=10)},
        timeout=20,
        turn=1,
        count_offset=0,
        interrupt_waiter=interrupt,
    )

    assert results[0].interrupted is True
    assert results[0].result.startswith("[interrupted]")


@pytest.mark.asyncio
async def test_finished_tool_result_wins_when_interrupt_lands_together() -> None:
    """Work that already completed is never thrown away for an interrupt."""

    async def interrupt(_call: dict[str, Any]) -> bool:
        await asyncio.sleep(0.02)
        return True

    results = await execute_tools(
        [{"name": "collect_reports", "args": {}, "id": "fan"}],
        {"collect_reports": FakeTool("collect_reports", "gathered")},
        timeout=20,
        turn=1,
        count_offset=0,
        interrupt_waiter=interrupt,
    )

    assert results[0].result == "gathered"
    assert results[0].interrupted is False


@pytest.mark.asyncio
async def test_raising_host_hooks_fall_back_instead_of_failing_the_batch() -> None:
    """Metering and budgeting are advisory; a broken one must not kill tools."""

    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("hook exploded")

    hooks = ToolExecutionHooks(
        resolve_timeout=boom,
        on_call=boom,
        transform_batch=boom,
        unknown_result=boom,
        failure_result=boom,
    )
    results = await execute_tools(
        [
            {"name": "ok", "args": {}, "id": "a"},
            {"name": "missing", "args": {}, "id": "b"},
            {"name": "bad", "args": {}, "id": "c"},
        ],
        {"ok": FakeTool("ok", "done"), "bad": FakeTool("bad", ValueError("boom"))},
        timeout=1,
        turn=1,
        count_offset=0,
        hooks=hooks,
    )

    assert [result.tool_call_id for result in results] == ["a", "b", "c"]
    assert results[0].result == "done"
    assert "unknown tool 'missing'" in results[1].result
    assert "boom" in results[2].result


def test_non_positive_result_cap_means_unlimited_not_tail_trimming() -> None:
    """``-1`` is a common "no cap" sentinel; it must not slice from the end."""
    from agent_core.loop_types import ToolResult
    from agent_core.runtime.loop.tool_exec import DefaultToolResultPostProcessor

    body = "abcdefghij"
    result = ToolResult(
        name="t", args={}, result=body, duration_ms=0,
        tool_call_id="x", is_error=False,
    )

    assert DefaultToolResultPostProcessor(None).process(result) == body
    assert DefaultToolResultPostProcessor(0).process(result) == body
    assert DefaultToolResultPostProcessor(-3).process(result) == body

    capped = DefaultToolResultPostProcessor(4).process(result)
    assert capped.startswith("abcd")
    assert "truncated 6 chars" in capped
