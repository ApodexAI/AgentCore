# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnnecessaryIsInstance=false
"""Product-neutral parallel tool execution with explicit host hooks."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from contextlib import AbstractContextManager, nullcontext, suppress
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from agent_core.loop_types import ToolResult

__all__ = [
    "PROTECTED_FANIN_TOOLS",
    "TOOL_RESULT_MAX_CHARS",
    "DefaultToolResultPostProcessor",
    "ToolExecutionHooks",
    "ToolLike",
    "ToolResultPostProcessor",
    "execute_tools",
]

TOOL_RESULT_MAX_CHARS = 150_000
_AGGREGATION_TOOLS = frozenset({"collect_reports", "collect_results"})
_SELF_TIMING_TOOL_FLOORS: dict[str, int] = {
    "run_python_code": 660,
    "download_file": 660,
}
PROTECTED_FANIN_TOOLS = frozenset(
    {"collect_reports", "collect_results", "delegate_subtask", "assign_task"}
)


@runtime_checkable
class ToolLike(Protocol):
    """Small structural contract required by the execution engine."""

    name: str

    async def ainvoke(self, args: dict[str, Any]) -> Any: ...


@runtime_checkable
class ToolResultPostProcessor(Protocol):
    def process(self, tool_result: ToolResult) -> str: ...


class DefaultToolResultPostProcessor:
    def __init__(self, max_chars: int | None = None) -> None:
        self._max_chars = max_chars

    def process(self, tool_result: ToolResult) -> str:
        content = tool_result.result
        cap = self._max_chars
        if cap and isinstance(content, str) and len(content) > cap:
            return content[:cap] + (
                f"\n\n[... truncated {len(content) - cap} chars past "
                f"{cap}-char cap]"
            )
        return content if isinstance(content, str) else str(content)


TimeoutResolver = Callable[[str, dict[str, Any], int], float]
AwaitCall = Callable[[Awaitable[Any], str, dict[str, Any], float], Awaitable[Any]]
CallScopeFactory = Callable[
    [dict[str, Any], float], AbstractContextManager[Any]
]
ResultTransform = Callable[[str, str], str]
BatchTransform = Callable[[list[ToolResult]], list[ToolResult]]
CallObserver = Callable[[str], None]
UnknownResult = Callable[[str, tuple[str, ...]], str]
TimeoutResult = Callable[[str, float, float], str]
FailureResult = Callable[[str, Exception], str]
InterruptedResult = Callable[[str], str]


def _default_timeout(name: str, args: dict[str, Any], configured: int) -> float:
    floor = _SELF_TIMING_TOOL_FLOORS.get(name)
    if floor is not None:
        return float(max(configured, floor))
    if name not in _AGGREGATION_TOOLS:
        return float(configured)
    try:
        requested = int(args.get("timeout", 0) or 0)
    except (TypeError, ValueError):
        requested = 0
    return float(max(configured, requested + 5)) if requested > 0 else float(configured)


async def _default_await(
    awaitable: Awaitable[Any],
    _name: str,
    _args: dict[str, Any],
    timeout: float,
) -> Any:
    return await asyncio.wait_for(awaitable, timeout=timeout)


def _default_scope(
    _call: dict[str, Any], _timeout: float
) -> AbstractContextManager[Any]:
    return nullcontext()


def _identity_result(_name: str, value: str) -> str:
    if len(value) <= TOOL_RESULT_MAX_CHARS:
        return value
    return value[:TOOL_RESULT_MAX_CHARS] + (
        f"\n... [truncated, {len(value)} chars total]"
    )


def _identity_batch(results: list[ToolResult]) -> list[ToolResult]:
    return results


def _noop_call(_name: str) -> None:
    return None


def _unknown_result(name: str, available: tuple[str, ...]) -> str:
    choices = ", ".join(available) or "(none)"
    return (
        f"Error: unknown tool '{name}' is not available. "
        f"Available tools: {choices}. Call one of these instead."
    )


def _timeout_result(name: str, elapsed_s: float, _effective_timeout: float) -> str:
    return f"Error: tool '{name}' timed out after {elapsed_s:.1f}s"


def _failure_result(name: str, exc: Exception) -> str:
    del name
    return f"Error: {type(exc).__name__}: {exc}"


def _interrupted_result(_name: str) -> str:
    return (
        "[interrupted] Waiting for sub-agent reports was cancelled because "
        "a new user message arrived."
    )


@dataclass(frozen=True)
class ToolExecutionHooks:
    """Host decisions around a shared execution lifecycle.

    The core owns dispatch, interrupt races, cancellation hygiene and result
    construction. Products inject deadline policy, contextvars, metering,
    spill/truncation behavior and aggregate budgeting through these hooks.
    """

    resolve_timeout: TimeoutResolver = _default_timeout
    await_call: AwaitCall = _default_await
    call_scope: CallScopeFactory = _default_scope
    transform_result: ResultTransform = _identity_result
    transform_batch: BatchTransform = _identity_batch
    on_call: CallObserver = _noop_call
    unknown_result: UnknownResult = _unknown_result
    timeout_result: TimeoutResult = _timeout_result
    failure_result: FailureResult = _failure_result
    interrupted_result: InterruptedResult = _interrupted_result


async def execute_tools(
    tool_calls: list[dict[str, Any]],
    tool_map: dict[str, ToolLike],
    *,
    timeout: int,
    turn: int,
    count_offset: int,
    interrupt_waiter: Callable[[dict[str, Any]], Awaitable[bool]] | None = None,
    hooks: ToolExecutionHooks | None = None,
) -> list[ToolResult]:
    """Execute a tool-call batch concurrently and convert every outcome.

    Exceptions and timeouts become error results; no raw tool failure escapes
    into the loop. A fan-in tool may race an interrupt waiter so a follow-up
    user message can wake it without abandoning already collected work.
    """

    runtime = hooks or ToolExecutionHooks()

    async def _run_one(call: dict[str, Any], index: int) -> ToolResult:
        name = str(call.get("name", "") or "")
        raw_args = call.get("args", {}) or {}
        args = raw_args if isinstance(raw_args, dict) else {}
        tool_call_id = str(
            call.get("id", "") or f"call_{turn}_{count_offset + index}"
        )
        start = time.monotonic()
        tool = tool_map.get(name)
        if tool is None:
            return ToolResult(
                name=name,
                args=args,
                result=runtime.unknown_result(name, tuple(sorted(tool_map))),
                duration_ms=0,
                tool_call_id=tool_call_id,
                is_error=True,
            )

        effective_timeout = runtime.resolve_timeout(name, args, timeout)
        runtime.on_call(name)
        invoke_task: asyncio.Future[Any] | None = None
        interrupt_task: asyncio.Future[bool] | None = None
        woke_for_interrupt = False
        try:
            with runtime.call_scope(
                {**call, "id": tool_call_id}, effective_timeout
            ):
                invocation = runtime.await_call(
                    tool.ainvoke(args), name, args, effective_timeout
                )
                if interrupt_waiter is not None and name in _AGGREGATION_TOOLS:
                    invoke_task = asyncio.ensure_future(invocation)
                    interrupt_task = asyncio.ensure_future(interrupt_waiter(call))
                    done, _ = await asyncio.wait(
                        {invoke_task, interrupt_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if interrupt_task in done and bool(interrupt_task.result()):
                        woke_for_interrupt = True
                    if invoke_task not in done:
                        invoke_task.cancel()
                        with suppress(asyncio.CancelledError):
                            await invoke_task
                        raw = runtime.interrupted_result(name)
                    else:
                        raw = invoke_task.result()
                else:
                    raw = await invocation

            result = runtime.transform_result(name, str(raw) if raw is not None else "")
            return ToolResult(
                name=name,
                args=args,
                result=result,
                duration_ms=int((time.monotonic() - start) * 1000),
                tool_call_id=tool_call_id,
                is_error=False,
                interrupted=woke_for_interrupt,
            )
        except TimeoutError:
            elapsed = int((time.monotonic() - start) * 1000)
            return ToolResult(
                name=name,
                args=args,
                result=runtime.timeout_result(
                    name, elapsed / 1000, effective_timeout
                ),
                duration_ms=elapsed,
                tool_call_id=tool_call_id,
                is_error=True,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return ToolResult(
                name=name,
                args=args,
                result=runtime.failure_result(name, exc),
                duration_ms=int((time.monotonic() - start) * 1000),
                tool_call_id=tool_call_id,
                is_error=True,
            )
        finally:
            if interrupt_task is not None and not interrupt_task.done():
                interrupt_task.cancel()
                with suppress(asyncio.CancelledError):
                    await interrupt_task
            if invoke_task is not None and not invoke_task.done():
                invoke_task.cancel()
                with suppress(asyncio.CancelledError):
                    await invoke_task
            elif invoke_task is not None and not invoke_task.cancelled():
                with contextlib.suppress(Exception):
                    invoke_task.exception()

    tasks = [_run_one(call, index) for index, call in enumerate(tool_calls)]
    return runtime.transform_batch(list(await asyncio.gather(*tasks)))
