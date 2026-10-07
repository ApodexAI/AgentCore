"""Recovery parity and routing contracts across chat/stream transports."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest

from agent_core.completion import is_wholly_empty_response
from agent_core.errors import LLMCallExhausted
from agent_core.llm import LLMResponse, StreamDelta
from agent_core.loop_types import LoopConfig, LoopPolicy
from agent_core.messages import user_msg
from agent_core.providers.fallback import FallbackEntry, LLMFallbackChain, with_provider_stamp
from agent_core.runtime.loop._bind import (
    bind_max_tokens,
    bind_session_id,
    bind_temperature,
    bind_tools,
)
from agent_core.runtime.loop._call import call_llm
from agent_core.runtime.loop.agent_loop import run_agent_loop


class Script:
    def __init__(self, actions, *, model="m", delay=0):
        self.actions = list(actions)
        self.model = model
        self.delay = delay
        self.calls = []
        self.cancelled = False

    async def _next(self, messages, kwargs):
        self.calls.append((deepcopy(messages), dict(kwargs)))
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        action = self.actions.pop(0)
        if isinstance(action, BaseException):
            raise action
        return action

    async def chat(self, messages, **kwargs):
        return await self._next(messages, kwargs)

    async def stream(self, messages, **kwargs):
        response = await self._next(messages, kwargs)
        if isinstance(response, list):
            for delta in response:
                yield delta
            return
        yield StreamDelta(
            content=response.content if isinstance(response.content, str) else "",
            reasoning_blocks=response.content if isinstance(response.content, list) else [],
            reasoning_content=response.reasoning_content,
            usage=response.usage, usage_source=response.usage_source or response.response_metadata.get("usage_source", ""),
            finish_reason=response.finish_reason, model=self.model,
            refusal=response.response_metadata.get("refusal", ""),
            stop_details=response.response_metadata.get("stop_details", {}),
        )


async def noop(*_args, **_kwargs):
    pass


async def invoke(llm, streaming, events=None, **kwargs):
    async def record(event):
        if events is not None:
            events.append(dict(event))

    return await call_llm(
        llm, [user_msg("u")], timeout=10, max_retries=kwargs.pop("max_retries", 1), turn=1,
        on_delta=noop if streaming else None, retry_wait_fixed=0,
        on_attempt=record, **kwargs,
    )


def finished(events):
    return [e for e in events if e["phase"] == "finished"]


def assert_balanced(events, count):
    assert [e["attempt_index"] for e in events if e["phase"] == "started"] == list(range(1, count + 1))
    assert [e["attempt_index"] for e in finished(events)] == list(range(1, count + 1))


@pytest.mark.parametrize("streaming", [False, True])
async def test_blank_recovery_is_one_logical_call_with_discarded_attempts(streaming):
    llm = Script([LLMResponse(), LLMResponse(content=" \n"), LLMResponse(content="answer")])
    events = []
    response = await invoke(llm, streaming, events)
    assert response.content == "answer"
    assert len(llm.calls) == 3
    assert all(messages == [user_msg("u")] for messages, _ in llm.calls)
    assert_balanced(events, 3)
    assert [(e["outcome"], e["reason"]) for e in finished(events)] == [
        ("discarded", "empty_completion"), ("discarded", "empty_completion"), ("accepted", ""),
    ]


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("budget", [0, 1, 2])
async def test_empty_budget_is_independent_of_generic_retry_allowance(streaming, budget):
    llm = Script([LLMResponse()] * (budget + 1))
    events = []
    with pytest.raises(LLMCallExhausted) as caught:
        await invoke(llm, streaming, events, empty_completion_max_retries=budget)
    assert caught.value.reason == "empty_completion"
    assert len(llm.calls) == budget + 1
    assert_balanced(events, budget + 1)
    assert finished(events)[-1]["outcome"] == "failed"
    assert all(e["reason"] == "empty_completion" for e in finished(events))


@pytest.mark.parametrize("streaming", [False, True])
async def test_mixed_generic_and_blank_errors_do_not_extend_generic_allowance(streaming):
    llm = Script([LLMResponse(), RuntimeError("network reset"), LLMResponse(), LLMResponse(content="answer")])
    events = []
    assert (await invoke(llm, streaming, events, max_retries=2)).content == "answer"
    assert_balanced(events, 4)
    assert [e["reason"] for e in finished(events)] == ["empty_completion", "transient_error", "empty_completion", ""]
    llm = Script([LLMResponse(), RuntimeError("network reset"), LLMResponse(content="unreachable")])
    with pytest.raises(LLMCallExhausted) as caught:
        await invoke(llm, streaming)
    assert caught.value.reason == "exhausted"
    assert len(llm.calls) == 2


@pytest.mark.parametrize("streaming", [False, True])
async def test_native_chain_advances_after_same_leg_resamples_preserving_bindings(streaming):
    primary = Script([LLMResponse()] * 3 + [LLMResponse(content="next call")], model="primary")
    secondary = Script([LLMResponse(content="fallback")], model="secondary")
    chain = LLMFallbackChain([
        FallbackEntry(primary, provider="primary-vendor"), FallbackEntry(secondary, provider="secondary-vendor"),
    ])
    schema = {"type": "function", "function": {"name": "echo", "parameters": {"type": "object"}}}
    llm = bind_max_tokens(bind_temperature(bind_session_id(bind_tools(chain, [schema]), "session"), .7), 123)
    events = []
    result = await invoke(llm, streaming, events)
    assert result.content == "fallback"
    assert result.response_metadata["provider_actually_used"] == "secondary-vendor"
    if not streaming:
        assert result.response_metadata["fallback_used"] == 1
    assert len(primary.calls) == 3
    assert len(secondary.calls) == 1
    for _, kw in primary.calls + secondary.calls:
        assert kw["tools"] == [schema]
        assert kw["temperature"] == .7
        assert kw["max_tokens"] == 123
        assert kw["extra_headers"] == {"x-upstream-session-id": "session"}
    assert_balanced(events, 4)
    assert finished(events)[2]["recovery_action"] == "chain_advance"
    # Cached chain and next logical call always start independently.
    assert (await invoke(llm, streaming)).content == "next call"
    assert chain._start_index == 0


@pytest.mark.parametrize("streaming", [False, True])
async def test_provider_stamp_wrapper_preserves_nested_chain_recovery(streaming):
    first = Script([LLMResponse()] * 3)
    second = Script([LLMResponse(content="answer")])
    llm = with_provider_stamp(LLMFallbackChain.from_models([first, second]), "vendor")
    assert (await invoke(llm, streaming)).content == "answer"
    assert (len(first.calls), len(second.calls)) == (3, 1)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("triggers", [(), ("timeout",)])
async def test_serving_leg_trigger_barrier_is_respected(streaming, triggers):
    first = Script([LLMResponse()] * 3)
    second = Script([LLMResponse(content="unreachable")])
    llm = LLMFallbackChain([FallbackEntry(first, triggers=triggers), FallbackEntry(second)])
    with pytest.raises(LLMCallExhausted) as caught:
        await invoke(llm, streaming)
    assert caught.value.reason == "empty_completion"
    assert len(second.calls) == 0


@pytest.mark.parametrize("streaming", [False, True])
async def test_ordinary_failure_then_blank_routes_from_actual_serving_leg(streaming):
    first = Script([RuntimeError("upstream error")])
    second = Script([LLMResponse()] * 3)
    third = Script([LLMResponse(content="answer")])
    llm = LLMFallbackChain.from_models([first, second, third])
    assert (await invoke(llm, streaming)).content == "answer"
    assert [len(client.calls) for client in (first, second, third)] == [1, 3, 1]


@pytest.mark.parametrize("streaming", [False, True])
async def test_shared_native_chain_has_independent_concurrent_call_cursors(streaming):
    class Primary:
        model = "primary"
        async def chat(self, messages, **_kwargs):
            await asyncio.sleep(.001)
            return LLMResponse() if messages[-1]["content"] == "bad" else LLMResponse(content="primary")
        async def stream(self, messages, **kwargs):
            r = await self.chat(messages, **kwargs)
            yield StreamDelta(content=r.content)
    secondary = Script([LLMResponse(content="secondary")])
    chain = LLMFallbackChain.from_models([Primary(), secondary])
    async def one(text):
        return await call_llm(chain, [user_msg(text)], 10, 1, 1,
            on_delta=noop if streaming else None, retry_wait_fixed=0)
    bad, good = await asyncio.gather(one("bad"), one("good"))
    assert (bad.content, good.content) == ("secondary", "primary")
    assert len(secondary.calls) == 1


@pytest.mark.parametrize("streaming", [False, True])
async def test_empty_episode_and_fallback_share_one_logical_deadline(streaming, monkeypatch):
    monkeypatch.setattr("agent_core.runtime.loop._call._WALL_DEADLINE_FLOOR_S", .001)
    primary = Script([LLMResponse()] * 3, delay=.02)
    secondary = Script([LLMResponse(content="unreachable")])
    events = []
    with pytest.raises(LLMCallExhausted) as caught:
        await invoke(LLMFallbackChain.from_models([primary, secondary]), streaming, events,
            logical_call_timeout_s=.05)
    assert caught.value.reason == "logical_call_deadline"
    assert len(secondary.calls) == 0
    assert len(primary.calls) <= 3
    assert finished(events)[-1]["reason"] == "logical_call_deadline"
    assert all(e["outcome"] != "accepted" for e in finished(events))


@pytest.mark.parametrize("streaming", [False, True])
async def test_run_wall_deadline_precedes_logical_deadline(streaming, monkeypatch):
    monkeypatch.setattr("agent_core.runtime.loop._call._WALL_DEADLINE_FLOOR_S", .001)
    llm = Script([LLMResponse()] * 3)
    with pytest.raises(LLMCallExhausted) as caught:
        await invoke(llm, streaming, logical_call_timeout_s=10,
            wall_deadline_remaining=lambda: .05 if not llm.calls else 0)
    assert caught.value.reason == "wall_deadline"
    assert len(llm.calls) <= 1


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("response", [
    LLMResponse(usage={"prompt_tokens": 0, "completion_tokens": 0, "estimated": True}),
    LLMResponse(usage={"prompt_tokens": 100}, usage_source="estimated"),
    LLMResponse(usage={"prompt_tokens": 100}, response_metadata={"usage_source": "estimated"}),
])
async def test_synthetic_usage_does_not_hide_a_transport_blank(streaming, response):
    llm = Script([response, LLMResponse(content="answer")])
    assert (await invoke(llm, streaming)).content == "answer"
    assert len(llm.calls) == 2


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("response", [
    LLMResponse(usage={"prompt_tokens": 0, "completion_tokens": 0}, usage_source="provider"),
    LLMResponse(reasoning_content="thinking"),
    LLMResponse(content=[{"type": "redacted_thinking", "data": "opaque"}]),
])
async def test_actual_usage_and_reasoning_do_not_trigger_empty_recovery(streaming, response):
    llm = Script([response])
    assert await invoke(llm, streaming) is not None
    assert len(llm.calls) == 1


@pytest.mark.parametrize("response", [
    SimpleNamespace(content="", usage_metadata={"input_tokens": 0, "estimated": True}),
    SimpleNamespace(content="", response_metadata={"token_usage": {"prompt_tokens": 0, "estimated": True}}),
])
def test_legacy_wrapper_usage_estimates_are_not_provider_evidence(response):
    assert is_wholly_empty_response(response)


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("reason", ["refusal", "content_filter"])
async def test_explicit_rejections_stop_even_with_nudge_policy_and_missing_usage(streaming, reason):
    llm = Script([LLMResponse(finish_reason=reason)])
    result = await run_agent_loop(system_prompt="s", user_message="u", llm=llm, tools=[],
        config=LoopConfig(max_turns=5, max_llm_retries=1, stream_llm_tokens=streaming,
            loop_policy=LoopPolicy(no_tool_behavior="nudge"), retry_wait_fixed=0))
    assert result.stopped_by == reason
    assert result.turns_used == 1
    assert len(llm.calls) == 1


@pytest.mark.parametrize("streaming", [False, True])
async def test_outer_chain_receives_exhausted_blank_after_same_key_budget(streaming):
    llm = Script([LLMResponse()] * 3)
    events = []
    with pytest.raises(LLMCallExhausted) as caught:
        await invoke(llm, streaming, events, chain_fallback_active=lambda: True)
    assert caught.value.reason == "chain_advance"
    assert len(llm.calls) == 3
    assert_balanced(events, 3)


@pytest.mark.parametrize("streaming", [False, True])
async def test_direct_native_chain_recovers_without_runtime_cursor(streaming):
    blank = Script([LLMResponse()])
    good = Script([LLMResponse(content="answer")])
    chain = LLMFallbackChain.from_models([blank, good])
    if streaming:
        chunks = [d async for d in chain.stream([user_msg("u")])]
        assert chunks[-1].content == "answer"
    else:
        assert (await chain.chat([user_msg("u")])).content == "answer"
    assert (len(blank.calls), len(good.calls)) == (1, 1)


@pytest.mark.parametrize("streaming", [False, True])
async def test_legacy_empty_errors_share_the_same_recovery_policy(streaming):
    llm = Script([RuntimeError("no generation chunks returned"), LLMResponse(content="answer")])
    events = []
    assert (await invoke(llm, streaming, events)).content == "answer"
    assert [e["reason"] for e in finished(events)] == ["empty_completion", ""]
    assert_balanced(events, 2)


@pytest.mark.parametrize("streaming", [False, True])
async def test_middleware_proxy_preserves_hooks_and_shared_counter_across_recovery(streaming):
    from agent_core.components.middleware.llm.proxy import LLMProxy

    class Middleware:
        def __init__(self):
            self.before = []
            self.after = []
            self.chunks = []
        async def run_before(self, ctx, messages):
            self.before.append((ctx.call_index, ctx.role_id))
            return messages
        async def run_after(self, ctx, response, **kwargs):
            self.after.append(ctx.call_index)
            return response
        async def run_on_chunk(self, ctx, chunk, content):
            self.chunks.append(ctx.call_index)
            return False
        async def run_on_llm_error(self, ctx, error, attempt):
            return False

    primary = Script([LLMResponse()] * 3)
    secondary = Script([LLMResponse(content="answer"), LLMResponse(content="next")])
    middleware = Middleware()
    proxy = LLMProxy(with_provider_stamp(LLMFallbackChain.from_models([primary, secondary]), "p"), middleware, "role")
    assert (await invoke(proxy, streaming)).content == "answer"
    assert middleware.before == [(1, "role"), (2, "role"), (3, "role"), (4, "role")]
    assert middleware.after == [1, 2, 3, 4]
    assert proxy.call_counter == 4
    if streaming:
        assert middleware.chunks == [1, 2, 3, 4]


@pytest.mark.parametrize("streaming", [False, True])
async def test_unknown_transparent_wrapper_is_not_silently_unwrapped(streaming):
    class Wrapper:
        def __init__(self, inner):
            self.inner = inner
            self.calls = 0
        def __getattr__(self, name):
            return getattr(self.inner, name)
        async def chat(self, messages, **kwargs):
            self.calls += 1
            return await self.inner.chat(messages, **kwargs)
        async def stream(self, messages, **kwargs):
            self.calls += 1
            async for delta in self.inner.stream(messages, **kwargs):
                yield delta
    wrapper = Wrapper(with_provider_stamp(Script([LLMResponse(content="answer")]), "p"))
    assert (await invoke(wrapper, streaming)).content == "answer"
    assert wrapper.calls == 1


@pytest.mark.parametrize("streaming", [False, True])
async def test_cooldown_wrapper_detects_blanks_before_its_existing_degrade_policy(streaming):
    from agent_core.providers.fallback import CooldownFallbackLLM

    async def sleep(_delay):
        pass
    primary = Script([LLMResponse()] * 2)
    secondary = Script([LLMResponse(content="fallback"), LLMResponse(content="cooldown")])
    events = []
    async def hook(name, payload):
        events.append((name, payload))
    llm = CooldownFallbackLLM(primary, secondary, max_retries=2, cooldown_seconds=60,
        sleep=sleep, event_hook=hook)
    assert (await invoke(llm, streaming)).content == "fallback"
    assert (await invoke(llm, streaming)).content == "cooldown"
    assert len(primary.calls) == 2
    assert len(secondary.calls) == 2
    assert sum(name == "error" for name, _ in events) == 2


@pytest.mark.parametrize("streaming", [False, True])
async def test_explicit_empty_completion_trigger_routes_without_enabling_all_errors(streaming):
    primary = Script([LLMResponse()] * 3)
    secondary = Script([LLMResponse(content="answer")])
    llm = LLMFallbackChain([FallbackEntry(primary, triggers=("empty_completion",)), FallbackEntry(secondary)])
    assert (await invoke(llm, streaming)).content == "answer"


@pytest.mark.parametrize("streaming", [False, True])
async def test_exhausted_last_leg_is_bounded_and_every_attempt_finishes(streaming):
    primary = Script([LLMResponse()] * 3)
    secondary = Script([LLMResponse()] * 3)
    events = []
    with pytest.raises(LLMCallExhausted) as caught:
        await invoke(LLMFallbackChain.from_models([primary, secondary]), streaming, events)
    assert caught.value.reason == "empty_completion"
    assert (len(primary.calls), len(secondary.calls)) == (3, 3)
    assert_balanced(events, 6)
    assert [e["recovery_action"] for e in finished(events)] == [
        "retry_same_key", "retry_same_key", "chain_advance", "retry_same_key", "retry_same_key", "empty_completion",
    ]


async def test_estimated_usage_remains_labeled_on_discarded_attempt():
    llm = Script([LLMResponse(usage={"prompt_tokens": 100}, usage_source="estimated"), LLMResponse(content="answer")])
    events = []
    await invoke(llm, False, events)
    assert finished(events)[0]["usage"]["estimated"] is True


async def test_rejection_does_not_execute_tools_and_keeps_history_replayable():
    class Tool:
        name = "echo"
        calls = 0
        def to_openai_schema(self):
            return {"type": "function", "function": {"name": "echo", "parameters": {"type": "object"}}}
        async def ainvoke(self, args):
            self.calls += 1
            return "unreachable"
    tool = Tool()
    llm = Script([LLMResponse(finish_reason="content_filter", tool_calls=[{
        "id": "call", "type": "function", "function": {"name": "echo", "arguments": "{}"},
    }])])
    result = await run_agent_loop(system_prompt="s", user_message="u", llm=llm, tools=[tool],
        config=LoopConfig(max_turns=2, max_llm_retries=1, retry_wait_fixed=0))
    assert result.stopped_by == "content_filter"
    assert tool.calls == 0
    assert result.messages[-1]["role"] == "tool"
    assert result.messages[-1]["tool_call_id"] == "call"


@pytest.mark.parametrize("wrapper", ["chain", "cooldown"])
async def test_direct_empty_leg_fragments_cannot_corrupt_fallback_tool_arguments(wrapper):
    from agent_core.providers.fallback import CooldownFallbackLLM
    from agent_core.runtime.loop._streaming import _stream_llm_response

    primary = Script([[StreamDelta(tool_call_deltas=[{"index": 0, "id": "broken", "arguments": '{"x":'}])]])
    secondary = Script([[StreamDelta(tool_call_deltas=[{
        "index": 0, "id": "good", "name": "tool", "arguments": '{"x":1}',
    }])]])
    llm = LLMFallbackChain.from_models([primary, secondary]) if wrapper == "chain" else CooldownFallbackLLM(primary, secondary, max_retries=1)
    result = await _stream_llm_response(llm, [user_msg("u")], 10, noop)
    assert result.tool_calls == [{
        "id": "good", "type": "function", "function": {"name": "tool", "arguments": '{"x":1}'},
    }]


@pytest.mark.parametrize("wrapper", ["chain", "cooldown"])
async def test_candidate_empty_fragments_survive_if_the_same_leg_later_gets_a_name(wrapper):
    from agent_core.providers.fallback import CooldownFallbackLLM
    from agent_core.runtime.loop._streaming import _stream_llm_response

    primary = Script([[
        StreamDelta(tool_call_deltas=[{"index": 0, "id": "good", "arguments": '{"x":'}]),
        StreamDelta(tool_call_deltas=[{"index": 0, "name": "tool", "arguments": '1}'}]),
    ]])
    secondary = Script([LLMResponse(content="unreachable")])
    llm = LLMFallbackChain.from_models([primary, secondary]) if wrapper == "chain" else CooldownFallbackLLM(primary, secondary, max_retries=1)
    result = await _stream_llm_response(llm, [user_msg("u")], 10, noop)
    assert result.tool_calls[0]["function"]["arguments"] == '{"x":1}'
    assert len(secondary.calls) == 0


@pytest.mark.parametrize("replay", ["blank", "success"])
async def test_opportunistic_tool_argument_replay_has_its_own_attempt_events(replay):
    schema = {"type": "function", "function": {"name": "tool", "parameters": {
        "type": "object", "required": ["x"], "properties": {"x": {"type": "integer"}},
    }}}
    recovered = LLMResponse() if replay == "blank" else LLMResponse(tool_calls=[{
        "id": "good", "type": "function", "function": {"name": "tool", "arguments": '{"x":1}'},
    }])
    llm = bind_tools(Script([
        [StreamDelta(tool_call_deltas=[{"index": 0, "id": "original", "name": "tool", "arguments": "{}"}])],
        recovered,
    ]), [schema])
    events = []
    result = await invoke(llm, True, events)
    assert [e["attempt_index"] for e in events if e["phase"] == "started"] == [1, 2]
    by_index = {e["attempt_index"]: e for e in finished(events)}
    assert len(finished(events)) == 2
    if replay == "blank":
        assert by_index[1]["outcome"] == "accepted"
        assert by_index[2]["outcome"] == "failed"
        assert by_index[2]["reason"] == "empty_completion"
        assert result.tool_calls[0]["id"] == "original"
        assert result.response_metadata["stream_empty_args_fallback"] is False
    else:
        assert by_index[1]["outcome"] == "discarded"
        assert by_index[2]["outcome"] == "accepted"
        assert result.tool_calls[0]["id"] == "good"
        assert result.response_metadata["stream_empty_args_fallback"] is True
