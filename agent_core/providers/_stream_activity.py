"""Preserve HTTP activity that an SDK's event iterator may filter out."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from typing import Any

import httpx


class _ActivityByteStream(httpx.AsyncByteStream):
    def __init__(self, inner: Any, on_bytes: Callable[[], None]) -> None:
        self._inner = inner
        self._on_bytes = on_bytes

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._inner:
            if chunk:
                self._on_bytes()
            yield chunk

    async def aclose(self) -> None:
        await self._inner.aclose()


def _wrap_byte_stream(inner: Any, on_bytes: Callable[[], None]) -> Any:
    # SDKs can use different transports in the same process. Match the actual
    # response stream, rather than selecting a base from one SDK's version.
    if isinstance(inner, httpx.AsyncByteStream):
        return _ActivityByteStream(inner, on_bytes)

    from httpx2 import AsyncByteStream

    class _Httpx2ActivityByteStream(_ActivityByteStream, AsyncByteStream):
        pass

    return _Httpx2ActivityByteStream(inner, on_bytes)


async def stream_events_with_activity(stream: Any) -> AsyncGenerator[Any, None]:
    """Yield SDK events and ``None`` for real HTTP progress, including pings.

    SDKs drop SSE comments or pings before yielding typed events. Observe the
    public response byte stream before that filtering, and let each SDK continue
    to own parsing, errors, signatures and Bedrock event decoding. A bounded
    queue preserves event order without accumulating tokens or heartbeats.
    No timer invents activity: a silent socket still trips the caller's guard.
    """
    response = getattr(stream, "response", None)
    if response is None:
        # Also support custom/test SDK iterators without an HTTP response.
        async for event in stream:
            yield event
        return

    queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=1)
    end = object()

    def on_bytes() -> None:
        with contextlib.suppress(asyncio.QueueFull):
            queue.put_nowait(None)

    response.stream = _wrap_byte_stream(response.stream, on_bytes)

    async def read_events() -> None:
        try:
            async for event in stream:
                await queue.put(event)
        except asyncio.CancelledError:
            raise
        except BaseException:
            await queue.put(end)
            raise
        else:
            await queue.put(end)

    reader = asyncio.create_task(read_events())
    try:
        # Receiving HTTP headers is progress too; it ends the first-byte wait.
        yield None
        while True:
            event = await queue.get()
            if event is end:
                await reader  # Propagate SDK/network errors without swallowing them.
                break
            yield event
    finally:
        reader.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await reader
        await stream.close()
