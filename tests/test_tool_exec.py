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
    assert events == ["meter:echo", "enter:tc:12", "await:echo:3:12", "exit"]


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
