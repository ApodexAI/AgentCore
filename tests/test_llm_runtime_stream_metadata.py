"""Streaming assembly invariants in ``_stream_llm_response``.

History: the legacy chunk merger combined
``response_metadata`` with ``merge_dicts``, which **concatenated** string
values across chunks. Correct for tokenized content (one stream → one
string) but wrong for header-like fields an OpenAI-compatible proxy
repeats on every chunk (``model_name`` / ``system_fingerprint`` /
``finish_reason``) — ``final.usage.model_usage`` keys came out doubled
(``…claude-4.6-sonnet…claude-4.6-sonnet…``) on new-api streaming runs.

The native streaming path removes the failure mode by construction: the
client adapter normalises each chunk to a :class:`StreamDelta`, which
carries only ``content`` / ``reasoning_content`` / ``tool_call_deltas``
— no ``response_metadata`` to merge, so there is nothing to double.
``_stream_llm_response`` folds the deltas into an ``LLMResponse`` whose
``content`` is the clean concatenation. ``response_metadata`` stays empty
UNLESS a serving-leg wrapper stamps ``StreamDelta.provider``, which the
assembler folds into
``response_metadata["provider_actually_used"]`` so streamed calls carry
billing attribution (the doubling bug still cannot recur — only one
known key is written, never a per-chunk dict merge). ``usage`` /
``finish_reason`` / ``model`` are likewise carried through from the
provider's terminal chunks when present (``StreamDelta`` gained those
fields) — previously they were dropped, zeroing streaming usage/billing.
These tests pin that contract.
"""

from __future__ import annotations

import asyncio

import pytest

from agent_core.llm import StreamDelta
from agent_core.messages import user_msg
from agent_core.runtime.loop.llm_client import (
    call_llm,
    extract_usage,
)


class _StubStreamingLLM:
    """Fake client whose ``stream`` yields a fixed list of
    ``StreamDelta``s — the native shape the OpenAIClient adapter emits
    per SSE chunk."""

    def __init__(self, *, deltas: list[StreamDelta]) -> None:
        self.deltas = deltas

    async def stream(self, _messages, **_kw):
        for delta in self.deltas:
            yield delta


@pytest.mark.asyncio
async def test_streaming_content_concatenated_no_metadata_doubling(monkeypatch):
    """N content deltas fold into a single clean ``LLMResponse``: content
    is concatenated (the merge behaviour we WANT), and because
    ``StreamDelta`` carries no ``response_metadata``, the assembled
    response has empty metadata — the model-name doubling bug cannot
    recur. ``extract_usage`` returns ``None`` since the streamed response
    carries no usage (the non-streaming path supplies usage instead)."""
    real_sleep = asyncio.sleep

    async def _noop(_):
        await real_sleep(0)

    monkeypatch.setattr("asyncio.sleep", _noop)

    deltas = [
        StreamDelta(content="hello "),
        StreamDelta(content="world"),
    ]
    llm = _StubStreamingLLM(deltas=deltas)

    delta_log: list[tuple[str, str, int]] = []

    async def _on_delta(delta, accumulated, idx, thinking=""):
        delta_log.append((delta, accumulated, idx))

    result = await call_llm(
        llm, [user_msg("hi")],
        timeout=10, max_retries=1, turn=0,
        on_delta=_on_delta,
    )

    assert result is not None
    # Content concatenation (the merge behaviour we WANT for body text)
    # is preserved — exactly once, not doubled.
    assert "hello world" in str(result.content)
    # No response_metadata is carried on the streamed response, so there
    # is no ``model_name`` to concatenate / double.
    assert result.response_metadata == {}
    # These deltas carry no terminal usage chunk, so usage stays empty here
    # and extract_usage reports None. (A usage-bearing stream is covered by
    # test_streaming_carries_terminal_usage_and_finish_reason below.)
    assert extract_usage(result) is None
    # on_delta saw the body text deltas in order.
    assert [d for d, _a, _i in delta_log] == ["hello ", "world"]


@pytest.mark.asyncio
async def test_streaming_carries_terminal_usage_and_finish_reason(monkeypatch):
    """Regression (P1): the terminal ``include_usage`` chunk (empty choices,
    usage set) and the last content chunk's ``finish_reason`` must reach the
    assembled ``LLMResponse``. Previously ``OpenAIClient.stream`` dropped the
    empty-choices usage chunk and ``StreamDelta`` had no usage/finish field,
    so streaming runs reported 0 usage and observers never saw
    ``finish_reason='length'`` (truncation / salvage / rollback)."""
    real_sleep = asyncio.sleep

    async def _noop(_):
        await real_sleep(0)

    monkeypatch.setattr("asyncio.sleep", _noop)

    deltas = [
        StreamDelta(content="hi", finish_reason="stop", model="m-1"),
        # Terminal include_usage chunk: empty content, carries final usage.
        StreamDelta(
            usage={
                "prompt_tokens": 12, "completion_tokens": 3,
                "total_tokens": 15, "cached_tokens": 4,
            },
            model="m-1",
        ),
    ]
    llm = _StubStreamingLLM(deltas=deltas)

    async def _on_delta(*_):
        pass

    result = await call_llm(
        llm, [user_msg("hi")],
        timeout=10, max_retries=1, turn=0,
        on_delta=_on_delta,
    )

    assert result is not None
    assert str(result.content) == "hi"
    assert result.finish_reason == "stop"
    assert result.model == "m-1"
    assert result.usage == {
        "prompt_tokens": 12, "completion_tokens": 3,
        "total_tokens": 15, "cached_tokens": 4,
    }
    # extract_usage now surfaces the streamed usage (was None before the fix).
    u = extract_usage(result)
    assert u is not None
    assert u["prompt_tokens"] == 12
    assert u["completion_tokens"] == 3
    assert u["cached_tokens"] == 4


@pytest.mark.asyncio
async def test_streaming_folds_provider_actually_used(monkeypatch):
    """A serving-leg-stamped ``StreamDelta.provider`` is folded into the
    assembled response's ``response_metadata`` so streamed calls bill against
    the right vendor — the gap that split mirothinker into a ``provider=""``
    bucket and an ``@apodex`` bucket. ``extract_usage`` then surfaces it."""
    real_sleep = asyncio.sleep

    async def _noop(_):
        await real_sleep(0)

    monkeypatch.setattr("asyncio.sleep", _noop)

    deltas = [
        StreamDelta(content="hi", provider="apodex"),
        StreamDelta(
            usage={"prompt_tokens": 7, "completion_tokens": 2, "cached_tokens": 0},
            model="mirothinker_v20_397b",
            provider="apodex",
        ),
    ]
    llm = _StubStreamingLLM(deltas=deltas)

    async def _on_delta(*_):
        pass

    result = await call_llm(
        llm, [user_msg("hi")],
        timeout=10, max_retries=1, turn=0,
        on_delta=_on_delta,
    )

    assert result is not None
    assert result.response_metadata == {"provider_actually_used": "apodex"}
    u = extract_usage(result)
    assert u is not None
    assert u["provider"] == "apodex"


@pytest.mark.asyncio
async def test_streaming_many_chunks_content_clean(monkeypatch):
    """Three content deltas still concatenate cleanly into one body with
    no per-chunk metadata bleed — pins that adding chunks never
    re-introduces the doubling the legacy merge produced."""
    real_sleep = asyncio.sleep

    async def _noop(_):
        await real_sleep(0)

    monkeypatch.setattr("asyncio.sleep", _noop)

    deltas = [
        StreamDelta(content="a"),
        StreamDelta(content="b"),
        StreamDelta(content="c"),
    ]
    llm = _StubStreamingLLM(deltas=deltas)

    async def _on_delta(*_):
        pass

    result = await call_llm(
        llm, [user_msg("hi")],
        timeout=10, max_retries=1, turn=0,
        on_delta=_on_delta,
    )

    assert result is not None
    assert str(result.content) == "abc"
    assert result.response_metadata == {}


@pytest.mark.asyncio
async def test_streaming_forwards_tool_call_chunks(monkeypatch):
    real_sleep = asyncio.sleep

    async def _noop(_):
        await real_sleep(0)

    monkeypatch.setattr("asyncio.sleep", _noop)

    # Native StreamDelta tool-call deltas use the OpenAIClient adapter
    # shape: {index, id, name, arguments}.
    deltas = [
        StreamDelta(
            tool_call_deltas=[
                {
                    "index": 0,
                    "id": "call_1",
                    "name": "finalize_answer",
                    "arguments": '{"content":"hel',
                },
            ],
        ),
        StreamDelta(
            tool_call_deltas=[
                {
                    "index": 0,
                    "id": "call_1",
                    "name": None,
                    "arguments": 'lo"}',
                },
            ],
        ),
    ]
    llm = _StubStreamingLLM(deltas=deltas)
    seen: list[list[dict]] = []

    async def _on_delta(
        _delta,
        _accumulated,
        _idx,
        _thinking="",
        *,
        tool_call_args_chunks=None,
    ):
        seen.append(tool_call_args_chunks or [])

    result = await call_llm(
        llm, [user_msg("hi")],
        timeout=10, max_retries=1, turn=0,
        on_delta=_on_delta,
    )

    assert result is not None
    # ``_stream_llm_response`` forwards per-chunk arg deltas in the
    # {name, args, id, index} shape observers decode progressively.
    assert seen == [
        [{
            "name": "finalize_answer",
            "args": '{"content":"hel',
            "id": "call_1",
            "index": 0,
        }],
        [{
            "name": None,
            "args": 'lo"}',
            "id": "call_1",
            "index": 0,
        }],
    ]
    # The stitched tool call assembled across both chunks.
    assert result.tool_calls == [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "finalize_answer",
                "arguments": '{"content":"hello"}',
            },
        },
    ]
