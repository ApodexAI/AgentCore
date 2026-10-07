"""Real middleware consumers must receive authoritative streamed completion data."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy

import pytest

from agent_core.components.middleware.llm.base import LLMCallContext, LLMMiddlewareChain
from agent_core.components.middleware.llm.loop_detection import LoopDetectionMiddleware
from agent_core.components.middleware.llm.proxy import LLMProxy
from agent_core.components.middleware.llm.token_accounting import TokenAccountingMiddleware
from agent_core.components.middleware.llm.tracing import LLMTracingMiddleware
from agent_core.components.middleware.rate_limit import RateLimitMiddleware
from agent_core.execution_context import (
    ExecutionScope,
    reset_current_execution_scope,
    set_current_execution_scope,
)
from agent_core.llm import LLMResponse, StreamDelta
from agent_core.messages import system_msg, user_msg
from agent_core.models.task_budget import BudgetState, TaskBudget
from agent_core.runtime.loop._call import call_llm
from agent_core.runtime.loop._streaming import _stream_llm_response


class Script:
    model = "model"
    def __init__(self, actions):
        self.actions = list(actions)
        self.requests = []
    async def _next(self, messages):
        self.requests.append(deepcopy(messages))
        action = self.actions.pop(0)
        if isinstance(action, Exception):
            raise action
        return action
    async def chat(self, messages, **kwargs):
        return await self._next(messages)
    async def stream(self, messages, **kwargs):
        action = await self._next(messages)
        if isinstance(action, list):
            for delta in action:
                if isinstance(delta, Exception):
                    raise delta
                yield delta
        else:
            yield StreamDelta(content=action.content, model=self.model,
                usage=action.usage, usage_source=action.usage_source or action.response_metadata.get("usage_source", ""))


class Cost:
    def __init__(self):
        self.records = []
    def record(self, *args):
        self.records.append(args)
        return 0.0
    def get_summary(self, task_id):
        return {"calls": len(self.records)}


class Aggregator:
    def __init__(self):
        self.records = []
    def record_llm_call(self, **kwargs):
        self.records.append(kwargs)


class Events:
    def __init__(self):
        self.records = []
    async def append(self, **kwargs):
        self.records.append(kwargs)


class Trace:
    def __init__(self):
        self.records = []
    async def log_llm_call(self, **kwargs):
        self.records.append(kwargs)


@contextmanager
def scoped(budget=None, task="task"):
    scope = ExecutionScope(task_id=task, phase_id="phase", role_id="role",
        metadata={"budget_state": budget} if budget is not None else {})
    token = set_current_execution_scope(scope)
    try:
        yield
    finally:
        reset_current_execution_scope(token)


def chain(*middlewares):
    result = LLMMiddlewareChain()
    for mw in middlewares:
        result.add(mw)
    return result


async def consume(proxy, streaming, messages=None):
    messages = messages or [user_msg("hi")]
    if not streaming:
        return await proxy.chat(messages)
    return [delta async for delta in proxy.stream(messages)]


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("marker", ["source", "flag", "metadata"])
async def test_estimates_never_enter_cost_budget_events_or_usage_aggregator(streaming, marker):
    usage = {"prompt_tokens": 100, "completion_tokens": 50}
    response = LLMResponse(content="answer", usage=usage)
    if marker == "source":
        response.usage_source = "estimated"
    elif marker == "flag":
        response.usage["estimated"] = True
    else:
        response.response_metadata["usage_source"] = "estimated"
    cost, aggregator, events, trace = Cost(), Aggregator(), Events(), Trace()
    budget = BudgetState(allocated=TaskBudget(max_tokens=20))
    accounting = TokenAccountingMiddleware(events, cost_sink=cost, usage_aggregator=aggregator)
    proxy = LLMProxy(Script([response]), chain(accounting, LLMTracingMiddleware(trace)))
    with scoped(budget):
        await consume(proxy, streaming)
    assert accounting.get_usage("task")["llm_calls"] == 0
    assert cost.records == aggregator.records == events.records == []
    assert budget.tokens_used == budget.llm_calls_used == 0
    assert not budget.exhausted
    assert len(trace.records) == 1
    assert trace.records[0]["metadata"]["usage_source"] == "estimated"
    assert trace.records[0]["metadata"]["usage"]["prompt_tokens"] == 100


@pytest.mark.parametrize("streaming", [False, True])
async def test_empty_estimated_attempts_do_not_poison_real_cost_or_budget(streaming):
    client = Script([LLMResponse(usage={"prompt_tokens": 100}, usage_source="estimated"),
        LLMResponse(usage={"prompt_tokens": 200, "estimated": True}),
        LLMResponse(content="answer", model="model", usage={"prompt_tokens": 10, "completion_tokens": 5}, usage_source="provider")])
    cost, aggregator, events = Cost(), Aggregator(), Events()
    accounting = TokenAccountingMiddleware(events, cost_sink=cost, usage_aggregator=aggregator)
    budget = BudgetState(allocated=TaskBudget(max_tokens=20))
    proxy = LLMProxy(client, chain(accounting))
    async def noop(*args, **kwargs):
        pass
    with scoped(budget):
        result = await call_llm(proxy, [user_msg("hi")], 5, 1, 1,
            on_delta=noop if streaming else None, retry_wait_fixed=0)
    assert result.content == "answer"
    assert len(client.requests) == 3
    assert accounting.get_usage("task") == {"input": 10, "output": 5, "total": 15, "llm_calls": 1}
    assert cost.records == [("task", "model", 10, 5)]
    assert len(aggregator.records) == len(events.records) == 1
    assert budget.tokens_used == 15 and budget.llm_calls_used == 1
    assert not budget.exhausted


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("read,write", [(10, 20), (0, 20), (10, 0)])
async def test_canonical_cache_reads_and_writes_reach_real_consumers_once(streaming, read, write):
    usage = {"prompt_tokens": 100, "completion_tokens": 5, "cache_read_tokens": read,
        "cache_write_tokens": write, "cached_tokens": read + write, "cache_creation_tokens": write}
    cost, aggregator, events = Cost(), Aggregator(), Events()
    accounting = TokenAccountingMiddleware(events, cost_sink=cost, usage_aggregator=aggregator, scene="test")
    budget = BudgetState()
    proxy = LLMProxy(Script([LLMResponse(content="answer", model="model", usage=usage, usage_source="provider")]), chain(accounting))
    with scoped(budget):
        await consume(proxy, streaming)
    [record] = aggregator.records
    assert record["cache_read_tokens"] == read and record["cache_write_tokens"] == write
    assert record["scene"] == "test"
    [event] = events.records
    assert event["payload"]["this_call"]["cache_read"] == read
    assert event["payload"]["this_call"]["cache_creation"] == write
    assert cost.records == [("task", "model", 100, 5)]
    assert budget.tokens_used == 105 and budget.llm_calls_used == 1


@pytest.mark.parametrize("usage,expected", [
    ({"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 2, "cache_creation_input_tokens": 3}, (10, 5, 2, 3)),
    ({"prompt_tokens": 10, "completion_tokens": 5, "cached_tokens": 2, "cache_creation_tokens": 3}, (10, 5, 2, 3)),
    ({"prompt_tokens": 10, "completion_tokens": 5, "cache_read_tokens": 0, "cache_write_tokens": 0,
        "cached_tokens": 999, "cache_creation_tokens": 999}, (10, 5, 0, 0)),
])
def test_accounting_cache_aliases_and_canonical_zero_precedence(usage, expected):
    assert TokenAccountingMiddleware()._extract_usage(LLMResponse(usage=usage)) == expected


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("prompt,usage,source,expected", [
    ("hi", {"prompt_tokens": 10, "completion_tokens": 90}, "provider", 900),
    ("x" * 40, {"total_tokens": 0}, "provider", 1000),
    ("x" * 40, {}, "", 990),
    ("x" * 40, {"total_tokens": 500}, "estimated", 990),
    ("x" * 40, {"total_tokens": 500, "estimated": True}, "", 990),
    ("x" * 40, {"total_tokens": None}, "provider", 990),
    ("x" * 40, {"total_tokens": -1}, "provider", 990),
    ("x" * 8000, {"total_tokens": 100}, "provider", 900),
], ids=["zero-estimate", "reported-zero", "missing", "estimated-source", "estimated-flag", "null", "negative", "capped"])
async def test_rate_bucket_corrects_real_usage_against_actual_reservation(streaming, prompt, usage, source, expected):
    rate = RateLimitMiddleware(tokens_per_min=1000)
    rate._bucket._refill = lambda: None
    proxy = LLMProxy(Script([LLMResponse(content="answer", usage=usage, usage_source=source)]), chain(rate))
    await consume(proxy, streaming, [user_msg(prompt)])
    assert rate._bucket._token_tokens == expected
    assert rate._bucket._request_tokens == 59


async def test_rate_after_without_before_does_not_invent_a_reservation():
    rate = RateLimitMiddleware(tokens_per_min=1000)
    await rate.after_llm(LLMCallContext(), LLMResponse(usage={"total_tokens": 100}))
    assert rate._bucket._token_tokens == 1000


@pytest.mark.parametrize("streaming", [False, True])
async def test_real_loop_detector_gets_tool_calls_and_injects_the_next_hint(streaming):
    call = {"id": "call", "type": "function", "function": {"name": "echo", "arguments": '{"x":1}'}}
    actions = [LLMResponse(tool_calls=[deepcopy(call)]) for _ in range(3)]
    if streaming:
        actions = [[StreamDelta(tool_call_deltas=[{"index": 0, "id": "call", "name": "ec", "arguments": '{"x":'}]),
            StreamDelta(tool_call_deltas=[{"index": 0, "name": "ho", "arguments": '1}'}])] for _ in range(3)]
    client = Script(actions)
    detection = LoopDetectionMiddleware(trigger_count=2)
    proxy = LLMProxy(client, chain(detection), role_id="role")
    with scoped():
        for _ in range(3):
            await consume(proxy, streaming, [system_msg("system"), user_msg("hi")])
    assert "[Loop detected]" not in client.requests[1][0]["content"]
    assert "[Loop detected]" in client.requests[2][0]["content"]


async def test_stream_tool_assembly_matches_runtime_and_drops_nameless_slots():
    seen = []
    from agent_core.components.middleware.llm.base import LLMMiddleware
    class Capture(LLMMiddleware):
        name = "capture"
        async def after_llm(self, ctx, response):
            seen.append(response)
            return response
    client = Script([[
        StreamDelta(tool_call_deltas=[{"index": 2, "id": "b", "name": "second", "arguments": "{"}]),
        StreamDelta(tool_call_deltas=[{"index": 1, "id": "broken", "arguments": "{"}]),
        StreamDelta(tool_call_deltas=[{"index": 0, "id": "a", "name": "fi", "arguments": '{"x":'}]),
        StreamDelta(tool_call_deltas=[{"index": 2, "arguments": "}"}, {"index": 0, "name": "rst", "arguments": '1}'}]),
    ]])
    async def noop(*args, **kwargs):
        pass
    response = await _stream_llm_response(LLMProxy(client, chain(Capture())), [], 5, noop)
    assert seen[0].tool_calls == response.tool_calls
    assert [c["function"]["name"] for c in response.tool_calls] == ["first", "second"]
    assert [c["function"]["arguments"] for c in response.tool_calls] == ['{"x":1}', '{}']


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("rewrite_prompt", [False, True])
async def test_multiple_limiters_keep_independent_capped_reservations(streaming, rewrite_prompt):
    from agent_core.components.middleware.llm.base import LLMMiddleware
    class Rewrite(LLMMiddleware):
        name = "rewrite"
        async def before_llm(self, ctx, messages):
            return [user_msg("x" * 40)]
    first = RateLimitMiddleware(tokens_per_min=100)
    second = RateLimitMiddleware(tokens_per_min=20)
    first._bucket._refill = second._bucket._refill = lambda: None
    nodes = (first, Rewrite(), second) if rewrite_prompt else (first, second)
    proxy = LLMProxy(Script([LLMResponse(content="answer", usage={"total_tokens": 10}, usage_source="provider")]), chain(*nodes))
    await consume(proxy, streaming, [user_msg("x" * 400)])
    assert first._bucket._token_tokens == 90
    assert second._bucket._token_tokens == 10
    assert first._reserved_key != second._reserved_key


async def test_failed_stream_proposals_do_not_inject_false_loop_hints():
    delta = StreamDelta(tool_call_deltas=[{"index": 0, "id": "call", "name": "echo", "arguments": "{}"}])
    client = Script([[delta, RuntimeError("reset")], [delta, RuntimeError("reset")], [delta]])
    detection = LoopDetectionMiddleware(trigger_count=2)
    proxy = LLMProxy(client, chain(detection), role_id="role")
    with scoped():
        for _ in range(2):
            with pytest.raises(RuntimeError, match="reset"):
                await consume(proxy, True, [system_msg("system"), user_msg("hi")])
        await consume(proxy, True, [system_msg("system"), user_msg("hi")])
    assert all("[Loop detected]" not in request[0]["content"] for request in client.requests)
    assert len(detection._histories[("task", "role", "phase")]) == 1


async def test_consumer_closed_stream_proposal_does_not_enter_loop_history():
    client = Script([[StreamDelta(tool_call_deltas=[{"index": 0, "id": "call", "name": "echo", "arguments": "{}"}])]])
    detection = LoopDetectionMiddleware(trigger_count=2)
    proxy = LLMProxy(client, chain(detection), role_id="role")
    with scoped():
        stream = proxy.stream([system_msg("system"), user_msg("hi")])
        await anext(stream)
        await stream.aclose()
    assert not detection._histories and not detection._pending_hints


@pytest.mark.parametrize("streaming", [False, True])
async def test_reported_cache_only_call_is_not_lost_to_zero_base_tokens(streaming):
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "cache_read_tokens": 10, "cache_write_tokens": 20,
        "cached_tokens": 30, "cache_creation_tokens": 20}
    aggregator, events = Aggregator(), Events()
    accounting = TokenAccountingMiddleware(events, usage_aggregator=aggregator)
    proxy = LLMProxy(Script([LLMResponse(content="", usage=usage, usage_source="provider")]), chain(accounting))
    with scoped():
        await consume(proxy, streaming)
    assert accounting.get_usage("task")["llm_calls"] == 1
    [record] = aggregator.records
    assert record["cache_read_tokens"] == 10 and record["cache_write_tokens"] == 20
    assert len(events.records) == 1


async def test_reported_usage_from_failed_stream_is_billed_without_loop_history():
    cost, aggregator = Cost(), Aggregator()
    budget = BudgetState()
    accounting = TokenAccountingMiddleware(cost_sink=cost, usage_aggregator=aggregator)
    detection = LoopDetectionMiddleware(trigger_count=2)
    client = Script([[StreamDelta(model="model", usage_source="provider", usage={"prompt_tokens": 10, "completion_tokens": 5},
        tool_call_deltas=[{"index": 0, "id": "call", "name": "echo", "arguments": "{}"}]), RuntimeError("reset")]])
    proxy = LLMProxy(client, chain(accounting, detection), role_id="role")
    with scoped(budget), pytest.raises(RuntimeError, match="reset"):
        await consume(proxy, True)
    assert cost.records == [("task", "model", 10, 5)]
    assert budget.tokens_used == 15 and budget.llm_calls_used == 1
    assert len(aggregator.records) == 1
    assert not detection._histories


async def test_legacy_rate_context_correction_respects_the_reservation_cap():
    rate = RateLimitMiddleware(tokens_per_min=1000)
    rate._bucket._refill = lambda: None
    await rate._bucket.acquire(5000)
    ctx = LLMCallContext(metadata={"_rate_limit_estimated_tokens": 5000})
    await rate.after_llm(ctx, LLMResponse(usage={"total_tokens": 10}, usage_source="provider"))
    assert rate._bucket._token_tokens == 990


@pytest.mark.parametrize("estimate,expected", [(None, 1000), ("bad", 1000), (0, 900)])
async def test_legacy_rate_context_distinguishes_unknown_from_zero(estimate, expected):
    rate = RateLimitMiddleware(tokens_per_min=1000)
    ctx = LLMCallContext(metadata={"_rate_limit_estimated_tokens": estimate})
    await rate.after_llm(ctx, LLMResponse(usage={"total_tokens": 100}, usage_source="provider"))
    assert rate._bucket._token_tokens == expected
