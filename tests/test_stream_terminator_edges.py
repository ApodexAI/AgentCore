"""Edge cases of the stream-terminator check across the real OpenAI SDK.

The Chat Completions ``[DONE]`` sentinel is consumed by the SDK, so the adapter
watches the bytes the SDK's SSE decoder reads. Those are DECODED bytes: a
gateway that gzip/deflate-encodes its event stream must not hide the sentinel,
and every SSE line ending the SDK accepts must count.
"""
from __future__ import annotations

import gzip
import zlib
from types import SimpleNamespace

import pytest
from openai import AsyncOpenAI

from agent_core.errors import LLMOpenAITruncatedStream, LLMTruncatedStream
from agent_core.llm import StreamDelta
from agent_core.messages import user_msg
from agent_core.providers._stream_activity import (
    STREAM_TERMINATOR_ENV,
    StreamActivity,
)
from agent_core.providers.openai_chat import OpenAIClient
from agent_core.providers.openai_responses import OpenAIResponsesClient
from agent_core.runtime.loop._streaming import _stream_llm_response
from agent_core.runtime.loop.llm_client import LLMCallExhausted, call_llm
from agent_core.runtime.retriable import classify_error, is_transient_network
from tests.test_openai_stream_activity import _chat_chunk, _sse, sdk_httpx

MODEL = "m"


async def _ignore(*_args, **_kwargs) -> None:
    pass


class _Body(sdk_httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self) -> None:
        pass


async def _run(cls, chunks: list[bytes], headers: dict[str, str] | None = None):
    def respond(request):
        return sdk_httpx.Response(200, stream=_Body(chunks), headers={
            "content-type": "text/event-stream", **(headers or {}),
        })

    client = cls(MODEL, api_key="k")
    await client._client.close()
    async with AsyncOpenAI(api_key="k", max_retries=0, http_client=sdk_httpx.AsyncClient(
        transport=sdk_httpx.MockTransport(respond),
    )) as sdk:
        client._client = sdk
        return await _stream_llm_response(client, [user_msg("hi")], 5, _ignore)


def _content(text: str = "done", finish: str | None = None) -> bytes:
    return _sse(_chat_chunk(MODEL, {"content": text}, finish))


def _split(data: bytes, size: int = 7) -> list[bytes]:
    return [data[i:i + size] for i in range(0, len(data), size)]


def _deflate(data: bytes) -> bytes:
    return zlib.compress(data)


# ── Chat Completions: complete turns ─────────────────────────────────────

@pytest.mark.parametrize("encoding,compress", [
    ("gzip", gzip.compress), ("deflate", _deflate),
])
async def test_encoded_stream_still_shows_done(encoding, compress):
    """No finish_reason, only [DONE] — behind Content-Encoding."""
    body = compress(_content() + b"data: [DONE]\n\n")
    response = await _run(OpenAIClient, _split(body), {"content-encoding": encoding})
    assert response.content == "done"


async def test_encoded_stream_without_done_is_still_truncated():
    body = gzip.compress(_content())
    with pytest.raises(LLMOpenAITruncatedStream):
        await _run(OpenAIClient, [body], {"content-encoding": "gzip"})


@pytest.mark.parametrize("done", [
    b"data: [DONE]\n\n",
    b"data: [DONE]\r\n\r\n",
    b"data: [DONE]\r\r",
    b"data:[DONE]\n\n",
    b"data: [DONE]",  # body ends without the line break
], ids=["lf", "crlf", "cr", "no-space", "no-trailing-break"])
async def test_done_line_endings(done):
    response = await _run(OpenAIClient, [_content(), done])
    assert response.content == "done"


async def test_done_split_byte_by_byte():
    response = await _run(OpenAIClient, _split(_content() + b"data: [DONE]\n\n", 1))
    assert response.content == "done"


async def test_done_after_long_line_in_one_chunk():
    response = await _run(OpenAIClient, [_content("x" * 500) + b"data: [DONE]\n\n"])
    assert response.content == "x" * 500


async def test_finish_reason_alone_completes_an_encoded_stream():
    body = gzip.compress(_content(finish="stop"))
    response = await _run(OpenAIClient, [body], {"content-encoding": "gzip"})
    assert response.content == "done"


# ── Chat Completions: truncations ────────────────────────────────────────

@pytest.mark.parametrize("chunks", [
    [_content("par")],
    [_sse(_chat_chunk(MODEL, {"tool_calls": [{
        "index": 0, "id": "c", "type": "function",
        "function": {"name": "s", "arguments": '{"q'}}]}))],
    [b""],
    [_sse({"id": "x", "object": "chat.completion.chunk", "created": 1, "model": MODEL,
           "choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 0,
                                    "total_tokens": 1}})],
    [_content("data: [DONE]\n")],  # the sentinel inside model output
    [_content(), b": data: [DONE]\n\n"],  # inside an SSE comment
    [_content(), b"event: x\ndata: [DONE"],  # sentinel cut mid-token
], ids=["text", "mid-tool-call", "empty-body", "usage-only", "in-content",
        "in-comment", "cut-sentinel"])
async def test_chat_truncations_are_rejected(chunks):
    with pytest.raises(LLMOpenAITruncatedStream) as info:
        await _run(OpenAIClient, chunks)
    assert classify_error(info.value) == "truncated_stream"
    assert not is_transient_network(info.value)


async def test_chat_escape_hatch_accepts_and_logs(monkeypatch, caplog):
    monkeypatch.setenv(STREAM_TERMINATOR_ENV, "0")
    with caplog.at_level("WARNING"):
        response = await _run(OpenAIClient, [_content("par")])
    assert response.content == "par"
    assert "ended without [DONE] or finish_reason" in caplog.text


async def test_unobservable_stream_is_not_rejected():
    """A stream with no SDK decoder to watch: a missing finish_reason proves
    nothing, so the turn is accepted rather than failed on every call."""

    class _Iter:
        def __init__(self):
            self._chunks = iter([SimpleNamespace(
                usage=None, model=MODEL,
                choices=[SimpleNamespace(
                    delta=SimpleNamespace(content="done", tool_calls=None),
                    finish_reason=None,
                )],
            )])

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._chunks)
            except StopIteration:
                raise StopAsyncIteration from None

        async def close(self):
            pass

    client = OpenAIClient(MODEL, api_key="k")

    async def _open(_kwargs):
        return _Iter()

    client._open_stream = _open
    deltas = [d async for d in client.stream([user_msg("hi")])]
    assert "".join(d.content for d in deltas) == "done"


# ── Responses ────────────────────────────────────────────────────────────

async def test_responses_error_event_keeps_its_message():
    chunks = [_sse({"type": "error", "code": "server_error", "message": "upstream boom",
                    "param": None, "sequence_number": 1}, "error")]
    with pytest.raises(Exception) as info:
        await _run(OpenAIResponsesClient, chunks)
    assert not isinstance(info.value, LLMOpenAITruncatedStream)
    assert "upstream boom" in str(info.value)
    assert getattr(info.value, "status_code", None) == 500


async def test_responses_escape_hatch(monkeypatch, caplog):
    monkeypatch.setenv(STREAM_TERMINATOR_ENV, "0")
    chunks = [_sse({"type": "response.output_text.delta", "item_id": "i", "output_index": 0,
                    "content_index": 0, "delta": "par", "sequence_number": 1},
                   "response.output_text.delta")]
    with caplog.at_level("WARNING"):
        response = await _run(OpenAIResponsesClient, chunks)
    assert response.content == "par"
    assert "responses ended without" in caplog.text


# ── StreamActivity unit ──────────────────────────────────────────────────

def test_long_lines_are_not_retained():
    activity = StreamActivity()
    activity.observe_bytes(b"data: " + b"x" * 10_000)
    assert activity._line == b"" and activity._line_long
    activity.observe_bytes(b"\ndata: [DONE]\n")
    assert activity.saw_done


# ── Retry routing ────────────────────────────────────────────────────────

class _TruncatingLLM:
    """Raises a truncated stream ``truncate`` times, then completes."""

    def __init__(self, truncate: int) -> None:
        self.truncate = truncate
        self.calls = 0

    async def stream(self, messages, **_kw):
        self.calls += 1
        yield StreamDelta(reasoning_content="thinking")
        if self.calls <= self.truncate:
            raise LLMTruncatedStream(
                last_event="content_block_stop", events_seen=4,
                saw_message_delta=False, block_types=["thinking"], elapsed_s=1.0,
            )
        yield StreamDelta(content="done", finish_reason="stop")


async def test_no_chain_resamples_on_the_same_key():
    llm = _TruncatingLLM(truncate=2)
    response = await call_llm(
        llm, [user_msg("q")], timeout=30, max_retries=4, turn=1,
        on_delta=_ignore, retry_wait_fixed=0,
    )
    assert response.content == "done"
    assert llm.calls == 3


async def test_active_chain_advances_instead_of_resampling():
    llm = _TruncatingLLM(truncate=5)
    with pytest.raises(LLMCallExhausted) as info:
        await call_llm(
            llm, [user_msg("q")], timeout=30, max_retries=4, turn=1,
            on_delta=_ignore, retry_wait_fixed=0,
            chain_fallback_active=lambda: True,
        )
    assert info.value.reason == "chain_advance"
    assert isinstance(info.value.last_exc, LLMTruncatedStream)
    assert llm.calls == 1


def test_openai_truncation_is_classified():
    err = LLMOpenAITruncatedStream(protocol="responses", last_event="", events_seen=0,
                                   expected="response.completed", elapsed_s=0.0)
    assert classify_error(err) == "truncated_stream"
