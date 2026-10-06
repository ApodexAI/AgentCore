"""Preserve HTTP activity that an SDK's event iterator may filter out."""

from __future__ import annotations

import asyncio
import contextlib
import functools
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


@functools.cache
def _httpx2_stream_types() -> tuple[type, type] | None:
    try:
        from httpx2 import AsyncByteStream
    except ImportError:
        return None

    class _Httpx2ActivityByteStream(_ActivityByteStream, AsyncByteStream):
        pass

    return AsyncByteStream, _Httpx2ActivityByteStream


def _wrap_byte_stream(inner: Any, on_bytes: Callable[[], None]) -> Any | None:
    # SDKs can use different transports in the same process. Match the actual
    # response stream, rather than selecting a base from one SDK's version.
    # ``None`` means an unknown stream: leave it untouched rather than break it.
    if isinstance(inner, httpx.AsyncByteStream):
        return _ActivityByteStream(inner, on_bytes)
    httpx2_types = _httpx2_stream_types()
    if httpx2_types is not None and isinstance(inner, httpx2_types[0]):
        return httpx2_types[1](inner, on_bytes)
    return None


async def stream_events_with_activity(stream: Any) -> AsyncGenerator[Any, None]:
    """Yield SDK events and ``None`` for real HTTP progress, including pings.

    SDKs drop SSE comments or pings before yielding typed events. Observe the
    public response byte stream before that filtering, and let each SDK continue
    to own parsing, errors, signatures and Bedrock event decoding. A bounded
    queue preserves event order without accumulating tokens or heartbeats.
    No timer invents activity: a silent socket still trips the caller's guard.
    """
    queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=1)
    end = object()
    bytes_pending = False

    def flush_activity() -> None:
        nonlocal bytes_pending
        if bytes_pending:
            bytes_pending = False
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(None)

    def on_bytes() -> None:
        # Deferred until the reader yields: bytes the SDK turns into an event
        # in the same step are already proven by that event, so token-dense
        # streams do not carry a heartbeat for every chunk.
        nonlocal bytes_pending
        if not bytes_pending:
            bytes_pending = True
            asyncio.get_running_loop().call_soon(flush_activity)

    response = getattr(stream, "response", None)
    wrapped = _wrap_byte_stream(getattr(response, "stream", None), on_bytes)
    if response is None or wrapped is None:
        # Custom/test SDK iterators, or a transport we cannot observe safely.
        async for event in stream:
            yield event
        return
    response.stream = wrapped

    async def read_events() -> None:
        nonlocal bytes_pending
        try:
            async for event in stream:
                bytes_pending = False
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
        # Receiving HTTP headers is socket progress too (not first output).
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
