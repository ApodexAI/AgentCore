"""Native OpenAI/Anthropic LLMClient adapters — request shape + response mapping.

The SDK is mocked, so these run offline. They pin the wire-shape decisions that
the migration must not regress (``stream:false`` explicit, ``parallel_tool_calls``
on tool calls, per-call ``timeout`` only when explicitly passed, tool_call
``{type,id,function}`` order, ``cached_tokens`` mapping, Anthropic thinking-block
``signature`` preservation).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import BadRequestError

from agent_core.llm import LLMClient, LLMResponse
from agent_core.messages import assistant_msg, system_msg, tool_msg, user_msg
from agent_core.providers import anthropic as ac
from agent_core.providers.openai_chat import OpenAIClient
from agent_core.runtime.llm_request_overrides import (
    ThinkingRetryOverride,
    thinking_retry_override,
)
from agent_core.runtime.loop.llm_client import extract_usage

# ── OpenAI client ──────────────────────────────────────────────────────────


def _session_query(headers):
    if not headers:
        return {}
    for key, value in headers.items():
        if key.lower() == "x-upstream-session-id" and value:
            return {"x-upstream-session-id": str(value)}
    return {}


def _mk_openai_client():
    c = OpenAIClient("test-model", api_key="x", session_query_resolver=_session_query)
    return c


def _fake_completion():
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content="hello",
                tool_calls=[SimpleNamespace(
                    id="call_1",
                    function=SimpleNamespace(name="search", arguments='{"q":"x"}'),
                )],
                reasoning_content="because",
            ),
            finish_reason="tool_calls",
        )],
        usage=SimpleNamespace(
            prompt_tokens=10, completion_tokens=5, total_tokens=15,
            prompt_tokens_details=SimpleNamespace(cached_tokens=3),
        ),
        model="test-model",
        id="resp_1",
    )


def test_openai_thinking_retry_override_is_scoped_and_profile_aware():
    original = {
        "top_p": 0.95,
        "chat_template_kwargs": {
            "enable_thinking": True,
            "preserve_thinking": True,
        },
    }
    client = OpenAIClient("test-model", api_key="x", extra_body=original)

    with thinking_retry_override(ThinkingRetryOverride(
        mode="reduced", thinking_budget=2048, reasoning_effort="low",
    )):
        reduced = client._effective_extra_body()
    assert reduced["chat_template_kwargs"] == {
        "enable_thinking": True,
        "preserve_thinking": True,
        "thinking_budget": 2048,
    }

    with thinking_retry_override(ThinkingRetryOverride(mode="disabled")):
        disabled = client._effective_extra_body()
    assert disabled["chat_template_kwargs"] == {
        "enable_thinking": False,
        "preserve_thinking": True,
    }

    # The cached client/profile is unchanged after both one-request scopes.
    assert client.extra_body == original
    assert client._effective_extra_body() == original

    # Strict OpenAI profiles without an opted-in thinking dialect receive no
    # SGLang-only request fields.
    strict = OpenAIClient(
        "test-model", api_key="x", extra_body={"top_p": 0.9},
    )
    with thinking_retry_override(ThinkingRetryOverride(mode="disabled")):
        assert strict._effective_extra_body() == {"top_p": 0.9}


@pytest.mark.asyncio
async def test_openai_chat_sends_budget_preserve_and_disable_overrides():
    client = OpenAIClient(
        "test-model",
        api_key="x",
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": True,
                "preserve_thinking": True,
                "thinking_budget": 8192,
            },
        },
    )
    raw_resp = MagicMock()
    raw_resp.parse.return_value = _fake_completion()
    create = AsyncMock(return_value=raw_resp)
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(
            with_raw_response=SimpleNamespace(create=create),
        )),
    )

    await client.chat([user_msg("normal")])
    with thinking_retry_override(ThinkingRetryOverride(
        mode="expanded", thinking_budget=12288, reasoning_effort="high",
    )):
        await client.chat([user_msg("expanded")])
    with thinking_retry_override(ThinkingRetryOverride(
        mode="reduced", thinking_budget=2048, reasoning_effort="low",
    )):
        await client.chat([user_msg("reduced")])
    with thinking_retry_override(ThinkingRetryOverride(mode="disabled")):
        await client.chat([user_msg("disabled")])
    await client.chat([user_msg("normal again")])

    normal, expanded, reduced, disabled, restored = [
        call.kwargs["extra_body"]["chat_template_kwargs"]
        for call in create.call_args_list
    ]
    assert normal == {
        "enable_thinking": True,
        "preserve_thinking": True,
        "thinking_budget": 8192,
    }
    assert expanded == {
        "enable_thinking": True,
        "preserve_thinking": True,
        "thinking_budget": 12288,
    }
    assert reduced == {
        "enable_thinking": True,
        "preserve_thinking": True,
        "thinking_budget": 2048,
    }
    assert disabled == {
        "enable_thinking": False,
        "preserve_thinking": True,
    }
    assert restored == normal


@pytest.mark.asyncio
async def test_thinking_retry_override_is_isolated_across_concurrent_tasks():
    client = OpenAIClient(
        "test-model",
        api_key="x",
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": True,
                "preserve_thinking": True,
                "thinking_budget": 8192,
            },
        },
    )
    both_scoped = asyncio.Event()
    scoped_count = 0
    scoped_lock = asyncio.Lock()

    async def effective_body(override):
        nonlocal scoped_count
        with thinking_retry_override(override):
            async with scoped_lock:
                scoped_count += 1
                if scoped_count == 2:
                    both_scoped.set()
            await both_scoped.wait()
            return client._effective_extra_body()["chat_template_kwargs"]

    reduced, disabled = await asyncio.gather(
        effective_body(ThinkingRetryOverride(
            mode="reduced", thinking_budget=1024,
        )),
        effective_body(ThinkingRetryOverride(mode="disabled")),
    )

    assert reduced == {
        "enable_thinking": True,
        "preserve_thinking": True,
        "thinking_budget": 1024,
    }
    assert disabled == {
        "enable_thinking": False,
        "preserve_thinking": True,
    }
    assert client._effective_extra_body()["chat_template_kwargs"] == {
        "enable_thinking": True,
        "preserve_thinking": True,
        "thinking_budget": 8192,
    }


@pytest.mark.asyncio
async def test_openai_stream_sends_disabled_thinking_for_one_request_only():
    client = OpenAIClient(
        "test-model",
        api_key="x",
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": True,
                "preserve_thinking": True,
                "thinking_budget": 8192,
            },
        },
    )
    request_bodies: list[dict] = []

    async def fake_stream():
        yield SimpleNamespace(
            choices=[SimpleNamespace(
                delta=SimpleNamespace(
                    content="ok", reasoning_content=None, tool_calls=None,
                ),
                finish_reason="stop",
            )],
            model="test-model",
            usage=None,
        )

    async def create(**kwargs):
        request_bodies.append(kwargs["extra_body"])
        return fake_stream()

    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
    )

    with thinking_retry_override(ThinkingRetryOverride(mode="disabled")):
        _ = [delta async for delta in client.stream([user_msg("disabled")])]
    _ = [delta async for delta in client.stream([user_msg("restored")])]

    assert request_bodies[0]["chat_template_kwargs"] == {
        "enable_thinking": False,
        "preserve_thinking": True,
    }
    assert request_bodies[1]["chat_template_kwargs"] == {
        "enable_thinking": True,
        "preserve_thinking": True,
        "thinking_budget": 8192,
    }


@pytest.mark.parametrize("choices", [None, []])
def test_openai_response_rejects_missing_choices_clearly(choices):
    """Malformed 200 responses must not leak a cryptic None subscript error."""
    from agent_core.providers.openai_chat import _to_llm_response

    raw = SimpleNamespace(choices=choices, id="bad", model="test-model")
    with pytest.raises(ValueError, match="no choices"):
        _to_llm_response(raw)


@pytest.mark.asyncio
async def test_openai_chat_request_shape_and_mapping():
    c = _mk_openai_client()
    raw_resp = MagicMock()
    raw_resp.parse.return_value = _fake_completion()
    create = AsyncMock(return_value=raw_resp)
    c._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        with_raw_response=SimpleNamespace(create=create),
    )))

    resp = await c.chat(
        [system_msg("s"), user_msg("hi")],
        tools=[{"type": "function", "function": {"name": "search"}}],
        temperature=0.7,
        max_tokens=256,
        extra_headers={"x-upstream-session-id": "t1"},
        timeout=30,
    )

    kw = create.call_args.kwargs
    assert kw["stream"] is False                       # explicit, byte-aligned
    assert kw["parallel_tool_calls"] is True            # set when tools present
    assert kw["temperature"] == 0.7
    assert kw["max_completion_tokens"] == 256
    assert kw["extra_headers"] == {"x-upstream-session-id": "t1"}
    # EAS UCH affinity hashes the URL query parameter, not the header —
    # the session id must be mirrored into ``extra_query``.
    assert kw["extra_query"] == {"x-upstream-session-id": "t1"}
    assert kw["timeout"] == 30

    assert isinstance(resp, LLMResponse)
    assert resp.content == "hello"
    assert resp.reasoning_content == "because"
    assert resp.finish_reason == "tool_calls"
    # tool_call wire key order {type, id, function}; arguments is a JSON string.
    assert resp.tool_calls == [{
        "type": "function", "id": "call_1",
        "function": {"name": "search", "arguments": '{"q":"x"}'},
    }]
    assert list(resp.tool_calls[0].keys()) == ["type", "id", "function"]
    assert resp.usage == {
        "prompt_tokens": 10, "completion_tokens": 5,
        "total_tokens": 15, "cached_tokens": 3,
    }


def test_openai_session_header_mirrors_into_default_query():
    """Construction-time session header → ``default_query`` on the SDK
    client (EAS UCH affinity keys on the URL query parameter; the header
    alone never pins a replica). Non-session headers must NOT leak into
    the query string."""
    c = OpenAIClient(
        "m", api_key="x",
        default_headers={"HTTP-Referer": "miroharness",
                         "x-upstream-session-id": "task-42"},
        session_query_resolver=_session_query,
    )
    assert dict(c._client._custom_query) == {"x-upstream-session-id": "task-42"}

    c2 = OpenAIClient("m", api_key="x",
                      default_headers={"HTTP-Referer": "miroharness"})
    assert not c2._client._custom_query

    # helper semantics: case-insensitive key match, value coerced to str,
    # empty/None-safe.
    assert _session_query({"X-Upstream-Session-Id": "T"}) == {
        "x-upstream-session-id": "T"}
    assert _session_query(None) == {}
    assert _session_query({"other": "v"}) == {}


def test_cached_openai_client_drops_previous_scope_affinity():
    scope = "task-1"

    def current_scope() -> str:
        return scope

    c = OpenAIClient(
        "m",
        api_key="x",
        default_headers={"x-upstream-session-id": "task-1"},
        session_query_resolver=_session_query,
        session_scope_resolver=current_scope,
    )
    # Task-derived affinity is evaluated by AgentCore instead of being frozen
    # into the SDK client, so a cached client can safely cross task boundaries.
    assert not c._client._custom_query
    assert c._session_query(None) == {"x-upstream-session-id": "task-1"}

    scope = "task-2"
    assert c._session_query(None) == {}
    assert c._session_query({"x-upstream-session-id": "task-2"}) == {
        "x-upstream-session-id": "task-2",
    }


def test_static_openai_affinity_survives_scope_changes():
    c = OpenAIClient(
        "m",
        api_key="x",
        default_headers={"x-upstream-session-id": "static-route"},
        session_query_resolver=_session_query,
        session_scope_resolver=lambda: "",
    )
    assert dict(c._client._custom_query) == {
        "x-upstream-session-id": "static-route",
    }
    assert c._session_query(None) == {
        "x-upstream-session-id": "static-route",
    }


@pytest.mark.asyncio
async def test_openai_chat_omits_extra_query_without_session_header():
    """extra_headers without a session id must not grow an extra_query."""
    c = _mk_openai_client()
    raw_resp = MagicMock()
    raw_resp.parse.return_value = _fake_completion()
    create = AsyncMock(return_value=raw_resp)
    c._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        with_raw_response=SimpleNamespace(create=create),
    )))

    await c.chat([user_msg("hi")], extra_headers={"X-Title": "t"})
    kw = create.call_args.kwargs
    assert kw["extra_headers"] == {"X-Title": "t"}
    assert "extra_query" not in kw


@pytest.mark.asyncio
async def test_openai_stream_mirrors_session_into_extra_query():
    c = _mk_openai_client()

    async def fake_stream():
        yield SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content="ok", reasoning_content=None,
                                  tool_calls=None))])

    create = AsyncMock(return_value=fake_stream())
    c._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=create,
    )))

    _ = [d async for d in c.stream(
        [user_msg("hi")],
        extra_headers={"x-upstream-session-id": "task-7:dag"},
    )]
    kw = create.call_args.kwargs
    assert kw["extra_headers"] == {"x-upstream-session-id": "task-7:dag"}
    assert kw["extra_query"] == {"x-upstream-session-id": "task-7:dag"}


@pytest.mark.asyncio
async def test_openai_chat_omits_timeout_and_tools_when_absent():
    c = _mk_openai_client()
    raw_resp = MagicMock()
    raw_resp.parse.return_value = _fake_completion()
    create = AsyncMock(return_value=raw_resp)
    c._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        with_raw_response=SimpleNamespace(create=create),
    )))

    await c.chat([user_msg("hi")])
    kw = create.call_args.kwargs
    # No explicit timeout -> no ``x-stainless-read-timeout`` header (gotcha #3).
    assert "timeout" not in kw
    # No tools -> neither tools nor parallel_tool_calls is sent.
    assert "tools" not in kw and "parallel_tool_calls" not in kw


@pytest.mark.asyncio
async def test_openai_stream_yields_deltas_and_skips_choiceless_chunks():
    c = _mk_openai_client()

    async def fake_stream():
        yield SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content="he", reasoning_content=None, tool_calls=None))])
        yield SimpleNamespace(choices=[])  # usage-only chunk -> skipped
        yield SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(
                content="llo", reasoning_content="r",
                tool_calls=[SimpleNamespace(
                    index=0, id="c1",
                    function=SimpleNamespace(name="f", arguments='{}'))]))])

    c._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(return_value=fake_stream()),
    )))

    deltas = [d async for d in c.stream([user_msg("hi")])]
    assert len(deltas) == 2                              # choiceless chunk dropped
    assert deltas[0].content == "he"
    assert deltas[1].content == "llo"
    assert deltas[1].reasoning_content == "r"
    assert deltas[1].tool_call_deltas[0] == {
        "index": 0, "id": "c1", "name": "f", "arguments": "{}",
    }


@pytest.mark.asyncio
async def test_openai_stream_forwards_terminal_usage_chunk():
    """The empty-choices ``include_usage`` terminal chunk must surface as a
    usage-bearing StreamDelta (was previously dropped → streaming usage 0)."""
    c = _mk_openai_client()

    async def fake_stream():
        yield SimpleNamespace(
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content="hi", reasoning_content=None,
                                      tool_calls=None),
                finish_reason="stop")],
            model="m-1", usage=None)
        # Terminal include_usage chunk: empty choices, carries final usage.
        yield SimpleNamespace(
            choices=[], model="m-1",
            usage=SimpleNamespace(
                prompt_tokens=12, completion_tokens=3, total_tokens=15,
                prompt_tokens_details=SimpleNamespace(cached_tokens=4)))

    create = AsyncMock(return_value=fake_stream())
    c._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=create,
    )))

    deltas = [d async for d in c.stream([user_msg("hi")])]
    # stream_options.include_usage requested on the create call.
    assert create.call_args.kwargs["stream_options"] == {"include_usage": True}
    assert deltas[0].content == "hi"
    assert deltas[0].finish_reason == "stop"
    # Terminal chunk forwarded with usage (not dropped).
    assert deltas[-1].usage == {
        "prompt_tokens": 12, "completion_tokens": 3,
        "total_tokens": 15, "cached_tokens": 4,
    }


def _bad_request(msg: str) -> BadRequestError:
    req = httpx.Request("POST", "https://gw/v1/chat/completions")
    return BadRequestError(msg, response=httpx.Response(400, request=req), body=None)


@pytest.mark.asyncio
async def test_openai_stream_degrades_when_gateway_rejects_stream_options():
    """A gateway that 400s on ``stream_options`` must not break streaming: the
    client disables the option and retries once, then skips it on later calls."""
    c = _mk_openai_client()
    calls: list[dict] = []

    async def fake_stream():
        yield SimpleNamespace(
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content="ok", reasoning_content=None,
                                      tool_calls=None),
                finish_reason="stop")],
            model="m-1", usage=None)

    async def create(**kwargs):
        calls.append(kwargs)
        if "stream_options" in kwargs:
            raise _bad_request("Unknown parameter: 'stream_options'.")
        return fake_stream()

    c._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=create,
    )))

    deltas = [d async for d in c.stream([user_msg("hi")])]
    # First attempt sent stream_options and was rejected; retried without it.
    assert len(calls) == 2
    assert "stream_options" in calls[0]
    assert "stream_options" not in calls[1]
    assert c._stream_options_supported is False
    assert deltas[0].content == "ok"           # stream still ran

    # Subsequent calls skip stream_options entirely (no wasted round-trip).
    calls.clear()
    _ = [d async for d in c.stream([user_msg("hi")])]
    assert len(calls) == 1
    assert "stream_options" not in calls[0]


@pytest.mark.asyncio
async def test_openai_stream_reraises_unrelated_bad_request():
    """A 400 unrelated to stream_options must propagate, not be swallowed."""
    c = _mk_openai_client()

    async def create(**kwargs):
        raise _bad_request("model `x` does not exist")

    c._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=create,
    )))

    with pytest.raises(BadRequestError):
        _ = [d async for d in c.stream([user_msg("hi")])]


@pytest.mark.asyncio
async def test_openai_stream_demotes_rejected_reasoning_effort_once():
    """gpt-5.x advertises ``none`` but gateways 400 on it — demote to ``low``.

    Without this the reporter's very first call dies and, because report
    synthesis is fail-open, the run silently ships the pre-reporter answer.
    """
    c = OpenAIClient(
        "gpt-5.1", api_key="x", extra_body={"reasoning_effort": "none"},
    )
    calls: list[dict] = []

    async def fake_stream():
        yield SimpleNamespace(
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content="ok", reasoning_content=None,
                                      tool_calls=None),
                finish_reason="stop")],
            model="gpt-5.1", usage=None)

    async def create(**kwargs):
        calls.append(kwargs)
        if kwargs.get("extra_body", {}).get("reasoning_effort") == "none":
            raise _bad_request("Unsupported value for 'reasoning_effort': none")
        return fake_stream()

    c._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=create,
    )))

    deltas = [d async for d in c.stream([user_msg("hi")])]
    assert len(calls) == 2
    assert calls[0]["extra_body"]["reasoning_effort"] == "none"
    assert calls[1]["extra_body"]["reasoning_effort"] == "low"
    assert deltas[0].content == "ok"           # stream still ran

    # Sticky: later calls go straight out at the accepted effort.
    calls.clear()
    _ = [d async for d in c.stream([user_msg("hi")])]
    assert len(calls) == 1
    assert calls[0]["extra_body"]["reasoning_effort"] == "low"


@pytest.mark.asyncio
async def test_openai_chat_demotes_rejected_reasoning_effort():
    c = OpenAIClient(
        "gpt-5.1", api_key="x", extra_body={"reasoning_effort": "none"},
    )
    calls: list[dict] = []
    raw_resp = MagicMock()
    raw_resp.parse.return_value = _fake_completion()

    async def create(**kwargs):
        calls.append(kwargs)
        if kwargs.get("extra_body", {}).get("reasoning_effort") == "none":
            raise _bad_request("reasoning_effort not supported for this model")
        return raw_resp

    c._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        with_raw_response=SimpleNamespace(create=create),
    )))

    resp = await c.chat([user_msg("hi")])
    assert resp.content == "hello"
    assert [k["extra_body"]["reasoning_effort"] for k in calls] == ["none", "low"]


@pytest.mark.asyncio
async def test_openai_stream_reraises_effort_rejection_without_a_fallback():
    """Only the gpt-5 family/``none`` pair has a next enum — the rest raises."""
    c = OpenAIClient(
        "gpt-4o", api_key="x", extra_body={"reasoning_effort": "none"},
    )
    calls: list[dict] = []

    async def create(**kwargs):
        calls.append(kwargs)
        raise _bad_request("Unsupported value for 'reasoning_effort': none")

    c._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=create,
    )))

    with pytest.raises(BadRequestError):
        _ = [d async for d in c.stream([user_msg("hi")])]
    assert len(calls) == 1
    assert c.extra_body["reasoning_effort"] == "none"   # untouched


def test_openai_client_satisfies_protocol():
    assert isinstance(_mk_openai_client(), LLMClient)


# ── Anthropic conversions (pure, no SDK call) ───────────────────────────────


def test_anthropic_split_system():
    sys, rest = ac._split_system([system_msg("you are x"), user_msg("hi")])
    assert sys == "you are x"
    assert rest == [user_msg("hi")]
    # no leading system -> empty system, messages unchanged
    assert ac._split_system([user_msg("hi")]) == ("", [user_msg("hi")])


def test_anthropic_message_conversions():
    # tool result -> user/tool_result block
    assert ac._to_anthropic_msg(tool_msg("out", "c1")) == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "out"}],
    }
    # assistant with text + tool_use
    am = ac._to_anthropic_msg(assistant_msg(
        "thinking done",
        tool_calls=[{"id": "c1", "type": "function",
                     "function": {"name": "f", "arguments": '{"a": 1}'}}],
    ))
    assert am["role"] == "assistant"
    assert am["content"][0] == {"type": "text", "text": "thinking done"}
    assert am["content"][1] == {
        "type": "tool_use", "id": "c1", "name": "f", "input": {"a": 1},
    }
    # plain user
    assert ac._to_anthropic_msg(user_msg("hi")) == {"role": "user", "content": "hi"}


def test_anthropic_tool_conversion():
    out = ac._to_anthropic_tool(
        {"type": "function", "function": {
            "name": "search", "description": "d", "parameters": {"type": "object"}}})
    assert out == {"name": "search", "description": "d",
                   "input_schema": {"type": "object"}}


def test_anthropic_response_keeps_thinking_signature_and_cache_tokens():
    raw = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="reasoning...", signature="sig_abc"),
            SimpleNamespace(type="text", text="the answer"),
            SimpleNamespace(type="tool_use", id="c1", name="f", input={"a": 1}),
        ],
        usage=SimpleNamespace(
            input_tokens=20,
            output_tokens=8,
            cache_read_input_tokens=12,
            cache_creation_input_tokens=7,
            cache_creation=SimpleNamespace(ephemeral_1h_input_tokens=3),
        ),
        stop_reason="tool_use", model="claude-x", id="msg_1",
    )
    resp = ac._to_llm_response(raw)
    # thinking present -> content stays a block list (so content_block parser works)
    assert isinstance(resp.content, list)
    thinking_block = resp.content[0]
    assert thinking_block["type"] == "thinking"
    assert thinking_block["signature"] == "sig_abc"     # must be preserved
    assert resp.reasoning_content == "reasoning..."
    assert resp.tool_calls[0]["function"]["name"] == "f"
    assert resp.usage["cache_read_tokens"] == 12
    assert resp.usage["cache_write_tokens"] == 10
    assert resp.usage["cached_tokens"] == 22
    assert resp.usage["cache_creation_tokens"] == 10
    assert resp.usage["total_tokens"] == 28
    extracted = extract_usage(resp)
    assert extracted is not None
    assert extracted["cache_read_tokens"] == 12
    assert extracted["cache_write_tokens"] == 10
    assert extracted["cached_tokens"] == 22
    assert extracted["cache_creation_tokens"] == 10


@pytest.mark.asyncio
async def test_anthropic_real_streaming_maps_events_to_stream_deltas():
    """Real token-by-token streaming: Anthropic raw events map to the same
    StreamDelta shape the kernel assembler consumes for OpenAI (text /
    thinking / tool_call_deltas), with usage/finish/model on a terminal delta."""
    async def fake_events():
        yield SimpleNamespace(type="message_start", message=SimpleNamespace(
            model="claude-x",
            usage=SimpleNamespace(
                input_tokens=20,
                cache_read_input_tokens=5,
                cache_creation_input_tokens=7,
            )))
        yield SimpleNamespace(type="content_block_start", index=0,
                              content_block=SimpleNamespace(type="text"))
        yield SimpleNamespace(type="content_block_delta", index=0,
                              delta=SimpleNamespace(type="text_delta", text="Hel"))
        yield SimpleNamespace(type="content_block_delta", index=0,
                              delta=SimpleNamespace(type="text_delta", text="lo"))
        # tool_use block: open slot (id+name), then partial-JSON arg fragments.
        yield SimpleNamespace(type="content_block_start", index=1,
                              content_block=SimpleNamespace(
                                  type="tool_use", id="c1", name="search"))
        yield SimpleNamespace(type="content_block_delta", index=1,
                              delta=SimpleNamespace(
                                  type="input_json_delta", partial_json='{"q":'))
        yield SimpleNamespace(type="content_block_delta", index=1,
                              delta=SimpleNamespace(
                                  type="input_json_delta", partial_json='"x"}'))
        yield SimpleNamespace(type="message_delta",
                              delta=SimpleNamespace(stop_reason="tool_use"),
                              usage=SimpleNamespace(output_tokens=8))
        yield SimpleNamespace(type="message_stop")

    c = ac.AnthropicClient("claude-x", api_key="x")
    create = AsyncMock(return_value=fake_events())
    c._client = SimpleNamespace(messages=SimpleNamespace(create=create))

    deltas = [d async for d in c.stream([user_msg("hi")])]
    assert create.call_args.kwargs["stream"] is True

    # Text deltas stream through as content.
    assert [d.content for d in deltas if d.content] == ["Hel", "lo"]
    # Tool-call deltas: one opening (id+name), two partial-JSON arg fragments,
    # all at index 1 — the assembler stitches arguments into '{"q":"x"}'.
    tcds = [tcd for d in deltas for tcd in d.tool_call_deltas]
    assert tcds[0] == {"index": 1, "id": "c1", "name": "search", "arguments": ""}
    assert [t["arguments"] for t in tcds[1:]] == ['{"q":', '"x"}']
    assert all(t["index"] == 1 for t in tcds)
    # Terminal delta folds usage / finish_reason / model.
    terminal = deltas[-1]
    assert terminal.finish_reason == "tool_use"
    assert terminal.model == "claude-x"
    assert terminal.usage == {
        "prompt_tokens": 20, "completion_tokens": 8,
        "total_tokens": 28,
        "cache_read_tokens": 5,
        "cache_write_tokens": 7,
        "cached_tokens": 12,
        "cache_creation_tokens": 7,
    }


# ── reasoning_tokens off completion_tokens_details ─────────────────────────
#
# Pre-existing gap (since the Phase 0.5 native adapters), found while smoke-
# testing the shared-loop migration: ``extract_usage`` reads a FLAT
# ``reasoning_tokens`` key, ``anthropic_client`` and ``openai_responses_client``
# both fill it, and this client did not — so every reasoning model on the
# chat-completions path reported 0 thinking tokens. Verified against live
# deepseek-v4-flash (non-stream 66, stream 47) before the fix landed.


def _usage_with_reasoning(reasoning):
    """OpenAI usage object whose ``completion_tokens_details`` is ``reasoning``.

    Pass ``None`` for the gateways that send ``completion_tokens_details: null``
    (apodex does).
    """
    return SimpleNamespace(
        prompt_tokens=10, completion_tokens=50, total_tokens=60,
        prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        completion_tokens_details=reasoning,
    )


def _plain_completion(usage):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content="answer", tool_calls=None, reasoning_content="thought"
            ),
            finish_reason="stop",
        )],
        usage=usage,
        model="test-model",
        id="resp_r",
    )


def test_openai_usage_carries_reasoning_tokens():
    """Non-streaming: thinking tokens reach ``usage`` and ``extract_usage``."""
    c = _mk_openai_client()
    raw_resp = MagicMock()
    raw_resp.parse.return_value = _plain_completion(
        _usage_with_reasoning(SimpleNamespace(reasoning_tokens=42))
    )
    create = AsyncMock(return_value=raw_resp)
    c._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        with_raw_response=SimpleNamespace(create=create),
    )))

    resp = asyncio.run(c.chat([user_msg("hi")]))

    assert resp.usage["reasoning_tokens"] == 42
    # The whole point of the flat key: the loop reads usage through this.
    assert extract_usage(resp)["reasoning_tokens"] == 42
    # Thinking is billed INSIDE completion_tokens — it must not be added on top.
    assert resp.usage["completion_tokens"] == 50
    assert resp.usage["total_tokens"] == 60


def test_openai_usage_without_reasoning_details_stays_zero():
    """A gateway sending ``completion_tokens_details: null`` must not crash.

    The key is left unset and ``extract_usage`` zero-fills it, rather than the
    adapter inventing a count.
    """
    c = _mk_openai_client()
    raw_resp = MagicMock()
    raw_resp.parse.return_value = _plain_completion(_usage_with_reasoning(None))
    create = AsyncMock(return_value=raw_resp)
    c._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        with_raw_response=SimpleNamespace(create=create),
    )))

    resp = asyncio.run(c.chat([user_msg("hi")]))

    assert "reasoning_tokens" not in resp.usage
    assert extract_usage(resp)["reasoning_tokens"] == 0


async def test_openai_stream_terminal_usage_carries_reasoning_tokens():
    """Streaming: the terminal ``include_usage`` chunk carries them too.

    Non-streaming and streaming share ``_usage_dict``, so the gap hit both —
    and ``serve`` runs streaming, which is where the 0 was observed.
    """
    c = _mk_openai_client()

    async def fake_stream():
        yield SimpleNamespace(
            choices=[SimpleNamespace(
                delta=SimpleNamespace(content="hi", reasoning_content=None,
                                      tool_calls=None),
                finish_reason="stop")],
            model="m-1", usage=None)
        yield SimpleNamespace(
            choices=[], model="m-1",
            usage=_usage_with_reasoning(SimpleNamespace(reasoning_tokens=17)))

    create = AsyncMock(return_value=fake_stream())
    c._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=create,
    )))

    deltas = [d async for d in c.stream([user_msg("hi")])]

    assert deltas[-1].usage["reasoning_tokens"] == 17
