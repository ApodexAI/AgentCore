"""Preserve HTTP activity that an SDK's event iterator may filter out."""

from __future__ import annotations

import asyncio
import contextlib
import functools
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from typing import Any

import httpx

from agent_core.runtime.async_utils import await_bounded

_CLEANUP_TIMEOUT_S = 5.0


class StreamActivity:
    """Let the adapter acknowledge SDK events it forwards as model deltas."""

    def __init__(self) -> None:
        self.output_reported = False
        # The OpenAI SDK consumes the Chat Completions [DONE] sentinel without
        # yielding it. Keep this wire-level signal for the adapter's end check.
        self.saw_done = False
        self._line = b""
        self._line_long = False

    def mark_output(self) -> None:
        self.output_reported = True

    def observe_bytes(self, chunk: bytes) -> None:
        if self.saw_done:
            return
        parts = chunk.split(b"\n")
        for part in parts[:-1]:
            self._append_line(part)
            if not self._line_long:
                line = self._line.strip()
                if line.startswith(b"data:") and line[5:].strip() == b"[DONE]":
                    self.saw_done = True
            self._line = b""
            self._line_long = False
        self._append_line(parts[-1])

    def _append_line(self, part: bytes) -> None:
        # Only the short sentinel line matters; never retain an unbounded SSE
        # JSON line containing model output.
        if not self._line_long:
            if len(self._line) + len(part) <= 32:
                self._line += part
            else:
                self._line = b""
                self._line_long = True


class _ActivityByteStream(httpx.AsyncByteStream):
    def __init__(self, inner: Any, on_bytes: Callable[[bytes], None]) -> None:
        self._inner = inner
        self._on_bytes = on_bytes

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._inner:
            if chunk:
                self._on_bytes(chunk)
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


def _wrap_byte_stream(inner: Any, on_bytes: Callable[[bytes], None]) -> Any | None:
    # SDKs can use different transports in the same process. Match the actual
    # response stream, rather than selecting a base from one SDK's version.
    # ``None`` means an unknown stream: leave it untouched rather than break it.
    if isinstance(inner, httpx.AsyncByteStream):
        return _ActivityByteStream(inner, on_bytes)
    httpx2_types = _httpx2_stream_types()
    if httpx2_types is not None and isinstance(inner, httpx2_types[0]):
        return httpx2_types[1](inner, on_bytes)
    return None


async def stream_events_with_activity(
    stream: Any, activity: StreamActivity | None = None,
) -> AsyncGenerator[Any, None]:
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
    bytes_epoch = 0
    activity = activity if activity is not None else StreamActivity()

    def flush_activity() -> None:
        nonlocal bytes_pending
        if bytes_pending:
            bytes_pending = False
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(None)

    def on_bytes(chunk: bytes) -> None:
        # Defer event-less progress; parsed events carry their byte progress
        # to the adapter acknowledgement below instead.
        nonlocal bytes_pending, bytes_epoch
        activity.observe_bytes(chunk)
        bytes_epoch += 1
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
        event_epoch = 0
        try:
            async for event in stream:
                had_bytes = bytes_epoch != event_epoch
                event_epoch = bytes_epoch
                bytes_pending = False
                await queue.put((event, had_bytes))
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
            if event is None:
                yield None
                continue
            sdk_event, had_bytes = event
            activity.output_reported = False
            yield sdk_event
            # A signature/status event can be consumed without an adapter
            # delta. Its bytes must still keep the stall watchdog alive.
            if had_bytes and not activity.output_reported:
                yield None
    finally:
        deadline = asyncio.get_running_loop().time() + _CLEANUP_TIMEOUT_S
        caller = asyncio.current_task()
        cancellations = caller.cancelling() if caller is not None else 0
        reader.cancel()
        try:
            try:
                await await_bounded(reader, _CLEANUP_TIMEOUT_S, hold=True)
            except asyncio.CancelledError:
                # The reader is intentionally cancelled; a NEW cancellation of
                # the consumer during this wait must still propagate.
                if caller is not None and caller.cancelling() > cancellations:
                    raise
            except Exception:
                pass
        finally:
            remaining = max(deadline - asyncio.get_running_loop().time(), 0.0)
            with contextlib.suppress(TimeoutError):
                await await_bounded(stream.close(), remaining, hold=True)
