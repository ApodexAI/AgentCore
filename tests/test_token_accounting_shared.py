"""Tests for TokenAccountingMiddleware — cumulative tracking, budget charging, SSE emission."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from agent_core.components.middleware.llm.base import LLMCallContext
from agent_core.components.middleware.llm.token_accounting import (
    TokenAccountingMiddleware,
)
from agent_core.llm import LLMResponse

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_response(
    content: str = "hello",
    input_tokens: int = 0,
    output_tokens: int = 0,
    format: str = "langchain",
) -> LLMResponse:
    """Create an LLMResponse with token usage metadata in various provider formats.

    ``TokenAccountingMiddleware._extract_usage`` reads the native, normalised
    ``LLMResponse.usage`` dict. The infra clients (OpenAI + Anthropic) flatten
    every provider's raw usage into the OpenAI-wire shape
    ``{prompt_tokens, completion_tokens, total_tokens, cached_tokens}`` before
    it ever reaches the middleware, so the ``format`` arg here is purely a
    label — all formats land in the same canonical ``usage`` field.
    """
    resp = LLMResponse(content=content)

    if format in ("langchain", "openai", "anthropic"):
        resp.usage = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
    elif format == "none":
        pass  # No usage metadata
    return resp


# ── Token Extraction ─────────────────────────────────────────────────────


class TestTokenExtraction:
    def test_extract_langchain_format(self):
        mw = TokenAccountingMiddleware()
        resp = _make_response(input_tokens=100, output_tokens=50, format="langchain")
        inp, out, cr, cc = mw._extract_usage(resp)
        assert inp == 100
        assert out == 50
        assert cr == 0
        assert cc == 0

    def test_extract_openai_format(self):
        mw = TokenAccountingMiddleware()
        resp = _make_response(input_tokens=200, output_tokens=80, format="openai")
        inp, out, _cr, _cc = mw._extract_usage(resp)
        assert inp == 200
        assert out == 80

    def test_extract_anthropic_format(self):
        mw = TokenAccountingMiddleware()
        resp = _make_response(input_tokens=300, output_tokens=120, format="anthropic")
        inp, out, _cr, _cc = mw._extract_usage(resp)
        assert inp == 300
        assert out == 120

    def test_extract_no_usage(self):
        mw = TokenAccountingMiddleware()
        resp = _make_response(format="none")
        inp, out, cr, cc = mw._extract_usage(resp)
        assert inp == 0
        assert out == 0
        assert cr == 0
        assert cc == 0


# ── Cumulative Tracking ──────────────────────────────────────────────────


class TestCumulativeTracking:
    def test_single_call_accumulates(self):
        mw = TokenAccountingMiddleware()
        ctx = LLMCallContext(task_id="t1", role_id="solver")
        resp = _make_response(input_tokens=100, output_tokens=50)

        loop = asyncio.new_event_loop()
        loop.run_until_complete(mw.after_llm(ctx, resp))
        loop.close()

        usage = mw.get_usage("t1")
        assert usage["input"] == 100
        assert usage["output"] == 50
        assert usage["total"] == 150
        assert usage["llm_calls"] == 1

    def test_multiple_calls_accumulate(self):
        mw = TokenAccountingMiddleware()
        ctx = LLMCallContext(task_id="t1", role_id="solver")

        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            mw.after_llm(ctx, _make_response(input_tokens=100, output_tokens=50))
        )
        loop.run_until_complete(
            mw.after_llm(ctx, _make_response(input_tokens=200, output_tokens=80))
        )
        loop.close()

        usage = mw.get_usage("t1")
        assert usage["input"] == 300
        assert usage["output"] == 130
        assert usage["total"] == 430
        assert usage["llm_calls"] == 2

    def test_different_tasks_isolated(self):
        mw = TokenAccountingMiddleware()

        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            mw.after_llm(
                LLMCallContext(task_id="t1"),
                _make_response(input_tokens=100, output_tokens=50),
            )
        )
        loop.run_until_complete(
            mw.after_llm(
                LLMCallContext(task_id="t2"),
                _make_response(input_tokens=200, output_tokens=80),
            )
        )
        loop.close()

        assert mw.get_usage("t1")["total"] == 150
        assert mw.get_usage("t2")["total"] == 280

    def test_zero_tokens_skipped(self):
        mw = TokenAccountingMiddleware()
        ctx = LLMCallContext(task_id="t1")
        resp = _make_response(format="none")

        loop = asyncio.new_event_loop()
        loop.run_until_complete(mw.after_llm(ctx, resp))
        loop.close()

        usage = mw.get_usage("t1")
        assert usage["total"] == 0
        assert usage["llm_calls"] == 0

    def test_context_metadata_populated(self):
        mw = TokenAccountingMiddleware()
        ctx = LLMCallContext(task_id="t1", role_id="solver")
        resp = _make_response(input_tokens=100, output_tokens=50)

        loop = asyncio.new_event_loop()
        loop.run_until_complete(mw.after_llm(ctx, resp))
        loop.close()

        assert "token_usage" in ctx.metadata
        assert ctx.metadata["token_usage"]["this_call"]["total"] == 150
        assert ctx.metadata["token_usage"]["cumulative"]["total"] == 150

    def test_reset(self):
        mw = TokenAccountingMiddleware()
        ctx = LLMCallContext(task_id="t1")

        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            mw.after_llm(ctx, _make_response(input_tokens=100, output_tokens=50))
        )
        loop.close()

        assert mw.get_usage("t1")["total"] == 150
        mw.reset("t1")
        assert mw.get_usage("t1")["total"] == 0


# ── Budget Charging ──────────────────────────────────────────────────────


class TestBudgetCharging:
    def test_charges_budget_state(self):
        from agent_core.execution_context import (
            ExecutionScope,
            reset_current_execution_scope,
            set_current_execution_scope,
        )
        from agent_core.models.task_budget import BudgetState, TaskBudget

        budget_state = BudgetState(allocated=TaskBudget(max_tokens=10000))
        scope = ExecutionScope(
            task_id="t1", phase_id="react_solve", role_id="solver",
            metadata={"budget_state": budget_state},
        )
        token = set_current_execution_scope(scope)

        try:
            mw = TokenAccountingMiddleware()
            ctx = LLMCallContext(task_id="t1", role_id="solver")
            resp = _make_response(input_tokens=500, output_tokens=200)

            loop = asyncio.new_event_loop()
            loop.run_until_complete(mw.after_llm(ctx, resp))
            loop.close()

            assert budget_state.tokens_used == 700
            assert budget_state.llm_calls_used == 1
            assert budget_state.exhausted is False
        finally:
            reset_current_execution_scope(token)

    def test_budget_exhaustion_detected(self):
        from agent_core.execution_context import (
            ExecutionScope,
            reset_current_execution_scope,
            set_current_execution_scope,
        )
        from agent_core.models.task_budget import BudgetState, TaskBudget

        budget_state = BudgetState(allocated=TaskBudget(max_tokens=500))
        scope = ExecutionScope(
            task_id="t1", phase_id="react_solve", role_id="solver",
            metadata={"budget_state": budget_state},
        )
        token = set_current_execution_scope(scope)

        try:
            mw = TokenAccountingMiddleware()
            ctx = LLMCallContext(task_id="t1", role_id="solver")
            resp = _make_response(input_tokens=300, output_tokens=300)

            loop = asyncio.new_event_loop()
            loop.run_until_complete(mw.after_llm(ctx, resp))
            loop.close()

            assert budget_state.tokens_used == 600
            assert budget_state.exhausted is True
        finally:
            reset_current_execution_scope(token)


# ── SSE Event Emission ───────────────────────────────────────────────────


class TestSSEEmission:
    def test_emits_event_to_event_store(self):
        event_store = AsyncMock()
        mw = TokenAccountingMiddleware(event_store=event_store)
        ctx = LLMCallContext(task_id="t1", role_id="solver")
        resp = _make_response(input_tokens=100, output_tokens=50)

        loop = asyncio.new_event_loop()
        loop.run_until_complete(mw.after_llm(ctx, resp))
        loop.close()

        event_store.append.assert_called_once()
        call_kwargs = event_store.append.call_args.kwargs
        assert call_kwargs["task_id"] == "t1"
        payload = call_kwargs["payload"]
        assert payload["trace_type"] == "token_usage"
        assert payload["this_call"]["input"] == 100
        assert payload["cumulative"]["total"] == 150

    def test_no_event_without_event_store(self):
        mw = TokenAccountingMiddleware()  # no event_store
        ctx = LLMCallContext(task_id="t1", role_id="solver")
        resp = _make_response(input_tokens=100, output_tokens=50)

        loop = asyncio.new_event_loop()
        loop.run_until_complete(mw.after_llm(ctx, resp))
        loop.close()

        # No error, just no SSE
        assert mw.get_usage("t1")["total"] == 150

    def test_event_store_error_does_not_propagate(self):
        event_store = AsyncMock()
        event_store.append.side_effect = RuntimeError("DB error")
        mw = TokenAccountingMiddleware(event_store=event_store)
        ctx = LLMCallContext(task_id="t1", role_id="solver")
        resp = _make_response(input_tokens=100, output_tokens=50)

        loop = asyncio.new_event_loop()
        loop.run_until_complete(mw.after_llm(ctx, resp))
        loop.close()

        # Should still track tokens despite SSE failure
        assert mw.get_usage("t1")["total"] == 150


# ── Middleware Properties ────────────────────────────────────────────────


class TestMiddlewareProperties:
    def test_name(self):
        mw = TokenAccountingMiddleware()
        assert mw.name == "token_accounting"

    def test_enabled_by_default(self):
        mw = TokenAccountingMiddleware()
        assert mw.enabled is True

    def test_get_usage_unknown_task(self):
        mw = TokenAccountingMiddleware()
        usage = mw.get_usage("nonexistent")
        assert usage["total"] == 0
        assert usage["llm_calls"] == 0
