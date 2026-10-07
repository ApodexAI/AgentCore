"""Preserve HTTP activity that an SDK's event iterator may filter out."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import os
import re
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from typing import Any

import httpx

from agent_core.runtime.async_utils import await_bounded

_CLEANUP_TIMEOUT_S = 5.0
_SSE_LINE_BREAK = re.compile(rb"[\r\n]")
STREAM_TERMINATOR_ENV = "AGENT_CORE_STREAM_REQUIRE_TERMINATOR"


class StreamActivity:
    """Let the adapter acknowledge SDK events it forwards as model deltas."""

    def __init__(self) -> None:
        self.output_reported = False
        # The OpenAI SDK consumes the Chat Completions [DONE] sentinel without
        # yielding it. Keep this wire-level signal for the adapter's end check;
        # ``done_observable`` is False when no observer could be installed.
        self.saw_done = False
        self.done_observable = False
        self._line = b""
        self._line_long = False

    def mark_output(self) -> None:
        self.output_reported = True

    def observe_bytes(self, chunk: bytes) -> None:
        """Scan DECODED SSE bytes for the ``data: [DONE]`` line."""
        if self.saw_done:
            return
        # SSE ends a line on CRLF, LF or a lone CR. Splitting on both bytes
        # turns CRLF into an extra empty line, which is harmless here.
        parts = _SSE_LINE_BREAK.split(chunk)
        for part in parts[:-1]:
            self._append_line(part)
            self._check_line()
            self._line = b""
            self._line_long = False
        self._append_line(parts[-1])

    def finish_bytes(self) -> None:
        """The body ended: a final line without its line break still counts."""
        if not self.saw_done:
            self._check_line()
        self._line = b""
        self._line_long = False

    def _check_line(self) -> None:
        # Same rule as the SDK: field ``data``, one optional leading space,
        # value starting with ``[DONE]``.
        if self._line_long or not self._line.startswith(b"data:"):
            return
        value = self._line[5:]
        if value.startswith(b" "):
            value = value[1:]
        if value.startswith(b"[DONE]"):
            self.saw_done = True

    def _append_line(self, part: bytes) -> None:
        # Only the short sentinel line matters; never retain an unbounded SSE
        # JSON line containing model output.
        if not self._line_long:
            if len(self._line) + len(part) <= 32:
                self._line += part
            else:
                self._line = b""
                self._line_long = True


class _DoneSentinelDecoder:
    """Feed the SDK's SSE decoder through :meth:`StreamActivity.observe_bytes`.

    The decoder receives ``response.aiter_bytes()`` — already decompressed —
    whereas ``response.stream`` carries the raw body, where a gzip/deflate
    encoded SSE stream hides the sentinel entirely.
    """

    def __init__(self, inner: Any, activity: StreamActivity) -> None:
        self._inner = inner
        self._activity = activity

    def aiter_bytes(self, iterator: AsyncIterator[bytes]) -> AsyncIterator[Any]:
        activity = self._activity

        async def observed() -> AsyncIterator[bytes]:
            async for chunk in iterator:
                activity.observe_bytes(chunk)
                yield chunk
            activity.finish_bytes()

        return self._inner.aiter_bytes(observed())

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def watch_done_sentinel(stream: Any, activity: StreamActivity) -> None:
    """Record whether an OpenAI SDK stream delivered ``data: [DONE]``.

    Must run before the stream is iterated: the SDK reads ``_decoder`` when its
    event iterator starts. An object without one (a custom/test iterator)
    leaves ``done_observable`` False, and the adapter must not demand it.
    """
    decoder = getattr(stream, "_decoder", None)
    if decoder is None or not callable(getattr(decoder, "aiter_bytes", None)):
        return
    try:
        stream._decoder = _DoneSentinelDecoder(decoder, activity)
    except (AttributeError, TypeError):
        return
    activity.done_observable = True


def stream_terminator_required() -> bool:
    """Whether a stream that ends without its protocol terminator raises.

    ``AGENT_CORE_STREAM_REQUIRE_TERMINATOR=0`` is the escape hatch for a
    gateway that never sends one: the adapters then log the missing
    terminator and accept the turn, which is the pre-0.14.3 behaviour.
    """
    return os.getenv(STREAM_TERMINATOR_ENV, "1").strip() != "0"


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

    def on_bytes() -> None:
        # Defer event-less progress; parsed events carry their byte progress
        # to the adapter acknowledgement below instead.
        nonlocal bytes_pending, bytes_epoch
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
