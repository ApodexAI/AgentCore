# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnnecessaryIsInstance=false
"""Product-neutral parallel tool execution with explicit host hooks."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import AbstractContextManager, nullcontext, suppress
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from agent_core.loop_types import ToolResult

logger = logging.getLogger(__name__)

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
        if not isinstance(content, str):
            content = str(content)
        cap = self._max_chars
        # A non-positive cap is the conventional "unlimited" sentinel. Letting
        # it through would slice from the end (``content[:-3]``) and silently
        # drop the tail of every tool result.
        if cap is None or cap <= 0 or len(content) <= cap:
            return content
        return content[:cap] + (
            f"\n\n[... truncated {len(content) - cap} chars past "
            f"{cap}-char cap]"
        )


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


def _safe_hook[T](
    what: str,
    call: Callable[[], T],
    fallback: Callable[[], T],
) -> T:
    """Run a host hook, falling back to core behavior if it raises.

    Metering, timeout policy and batch budgeting are host concerns layered
    around execution; a broken one must not turn a whole tool batch -- or the
    surrounding agent loop -- into a failure. Mirrors the attempt-observer
    rule in ``_call.py``: passive observability never breaks a valid call.
    """

    try:
        return call()
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.warning(
            "tool execution hook %s raised; falling back to core default",
            what,
            exc_info=True,
        )
        return fallback()


def _interrupt_requested(task: asyncio.Future[bool], name: str) -> bool:
    """Did the waiter actually ask for an interrupt?

    A waiter that returns ``False`` means "no interrupt was requested", and a
    waiter that raises is a broken host observer -- neither is a user
    interrupt. Treating either as one would cancel a healthy fan-in tool and
    report collected work as abandoned, so both resolve to ``False`` here.
    """

    try:
        return bool(task.result())
    except asyncio.CancelledError:
        # The waiter itself was cancelled; that is not a user interrupt.
        return False
    except Exception:
        logger.warning(
            "tool interrupt waiter for %r raised; treating as no interrupt",
            name,
            exc_info=True,
        )
        return False


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
    # Which tools may be woken by an interrupt waiter. The default names are
    # the ones this engine was extracted from; a host whose fan-in tool is
    # called something else has to be able to say so, or its interrupt waiter
    # is never consulted no matter that it was supplied.
    aggregation_tools: frozenset[str] = _AGGREGATION_TOOLS


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
                result=_safe_hook(
                    "unknown_result",
                    lambda: runtime.unknown_result(
                        name, tuple(sorted(tool_map))
                    ),
                    lambda: _unknown_result(name, tuple(sorted(tool_map))),
                ),
                duration_ms=0,
                tool_call_id=tool_call_id,
                is_error=True,
            )

        effective_timeout = _safe_hook(
            "resolve_timeout",
            lambda: runtime.resolve_timeout(name, args, timeout),
            lambda: float(timeout),
        )
        _safe_hook("on_call", lambda: runtime.on_call(name), lambda: None)
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
                if (
                    interrupt_waiter is not None
                    and name in runtime.aggregation_tools
                ):
                    invoke_task = asyncio.ensure_future(invocation)
                    interrupt_task = asyncio.ensure_future(interrupt_waiter(call))
                    pending: set[asyncio.Future[Any]] = {
                        invoke_task,
                        interrupt_task,
                    }
                    while pending:
                        done, pending = await asyncio.wait(
                            pending, return_when=asyncio.FIRST_COMPLETED
                        )
                        # Completed work always wins: if the tool finished in
                        # the same wake-up as the waiter, keep its result.
                        if invoke_task in done:
                            break
                        # A waiter that finished without requesting an
                        # interrupt is not a reason to abandon the tool --
                        # keep waiting on the invocation alone.
                        if interrupt_task in done and _interrupt_requested(
                            interrupt_task, name
                        ):
                            woke_for_interrupt = True
                            break
                    if woke_for_interrupt:
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
                result=_safe_hook(
                    "timeout_result",
                    lambda: runtime.timeout_result(
                        name, elapsed / 1000, effective_timeout
                    ),
                    lambda: _timeout_result(
                        name, elapsed / 1000, effective_timeout
                    ),
                ),
                duration_ms=elapsed,
                tool_call_id=tool_call_id,
                is_error=True,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            failure = exc
            return ToolResult(
                name=name,
                args=args,
                result=_safe_hook(
                    "failure_result",
                    lambda: runtime.failure_result(name, failure),
                    lambda: _failure_result(name, failure),
                ),
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
    results = list(await asyncio.gather(*tasks))
    return _safe_hook(
        "transform_batch",
        lambda: runtime.transform_batch(results),
        lambda: results,
    )
