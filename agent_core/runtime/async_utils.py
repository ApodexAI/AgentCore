"""Bound asynchronous I/O without waiting indefinitely for cancellation."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any

_pending_cancellations: set[asyncio.Future[Any]] = set()


def _finish_cancelled(task: asyncio.Future[Any]) -> None:
    _pending_cancellations.discard(task)
    if not task.cancelled():
        task.exception()  # Retrieve errors from cleanup that finishes later.


async def await_bounded[T](operation: Awaitable[T], timeout: float) -> T:
    """Cancel overdue I/O and return without awaiting its cancellation handler.

    ``wait_for`` can exceed its timeout while a transport or callback unwinds.
    Keep unfinished cancellations referenced until they finish, retrieving any
    late exception. Cooperative operations still stop at the next checkpoint.
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
