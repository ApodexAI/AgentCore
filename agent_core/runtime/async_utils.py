"""Bound asynchronous I/O without waiting indefinitely for cancellation."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Generator
from contextvars import ContextVar
from typing import Any

_pending_cancellations: set[asyncio.Future[Any]] = set()
# Set by :func:`hold_until_settled`. Abandoned operations started with
# ``hold=True`` keep the enclosing lease (e.g. a concurrency slot) open.
_ABANDON_HOOK: ContextVar[Callable[[asyncio.Future[Any]], None] | None] = ContextVar(
    "agent_core_bounded_abandon_hook", default=None,
)


def _finish_cancelled(task: asyncio.Future[Any]) -> None:
    _pending_cancellations.discard(task)
    if not task.cancelled():
        task.exception()  # Retrieve errors from cleanup that finishes later.


async def await_bounded[T](
    operation: Awaitable[T], timeout: float, *, hold: bool = False,
) -> T:
    """Cancel overdue I/O and return without awaiting its cancellation handler.

    ``wait_for`` can exceed its timeout while a transport or callback unwinds.
    Keep unfinished cancellations referenced until they finish, retrieving any
    late exception. Cooperative operations still stop at the next checkpoint.

    The operation runs in its own task (with a copy of the caller's context).
    Async generators closed this way are finalized in that task, so a
    generator must not hold an ``asyncio.timeout``/anyio cancel scope open
    across a ``yield`` it may be closed at.

    ``hold=True`` marks provider I/O: if it is abandoned still running, the
    enclosing :func:`hold_until_settled` lease stays open until it finishes.
    """
    task = asyncio.ensure_future(operation)
    try:
        done, _ = await asyncio.wait({task}, timeout=max(timeout, 0.0))
        if not done:
            raise TimeoutError("asynchronous operation exceeded its deadline")
        return task.result()
    finally:
        if task.done():
            _finish_cancelled(task)
        else:
            task.cancel()
            _pending_cancellations.add(task)
            task.add_done_callback(_finish_cancelled)
            hook = _ABANDON_HOOK.get() if hold else None
            if hook is not None:
                hook(task)


class _Lease:
    def __init__(self, release: Callable[[], None]) -> None:
        self._release = release
        self._pending = 1  # The ``with`` block itself.
        self._released = False

    def track(self, task: asyncio.Future[Any]) -> None:
        if self._released:
            return
        self._pending += 1
        task.add_done_callback(self.settle)

    def settle(self, _task: Any = None) -> None:
        self._pending -= 1
        if self._pending == 0 and not self._released:
            self._released = True
            self._release()


@contextlib.contextmanager
def hold_until_settled(release: Callable[[], None]) -> Generator[None]:
    """Call ``release`` once the block exits and held abandoned I/O finishes.

    A caller may return at its deadline while a cancelled provider request is
    still unwinding; releasing a concurrency slot then would let the retry run
    alongside it. The slot is returned when the last such request settles.
    """
    lease = _Lease(release)
    token = _ABANDON_HOOK.set(lease.track)
    try:
        yield
    finally:
        _ABANDON_HOOK.reset(token)
        lease.settle()
