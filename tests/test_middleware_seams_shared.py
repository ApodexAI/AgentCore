"""Coverage for the three middleware seams that arrived here untested.

``LLMRetryMiddleware`` and ``LLMTracingMiddleware`` had no tests in the product
they came from. ``TokenAccountingMiddleware.persist_cost`` had none either, and
it is the one piece whose shape changed on the way in: a SQLAlchemy session
factory plus an inline table write became the injected ``CostPersister``
Protocol, so the seam needs a test that pins the contract rather than the
former implementation.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agent_core.components.middleware.llm.base import LLMCallContext
from agent_core.components.middleware.llm.retry import LLMRetryMiddleware
from agent_core.components.middleware.llm.token_accounting import (
    TokenAccountingMiddleware,
)
from agent_core.components.middleware.llm.tracing import LLMTracingMiddleware
from agent_core.llm import LLMResponse
from agent_core.messages import user_msg
from agent_core.protocols import CostSink

# ── LLMRetryMiddleware ───────────────────────────────────────────────────


def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record requested back-off delays without spending them."""
    slept: list[float] = []

    async def _fake(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr(asyncio, "sleep", _fake)
    return slept


@pytest.mark.asyncio
async def test_retry_asks_for_another_attempt_on_a_retryable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slept = _no_sleep(monkeypatch)
    mw = LLMRetryMiddleware(max_retries=3, backoff_base=0.5, backoff_max=8.0)

    assert await mw.on_llm_error(LLMCallContext(), TimeoutError("timed out"), 0) is True
    assert len(slept) == 1


@pytest.mark.asyncio
async def test_retry_declines_a_non_retryable_error_without_sleeping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A permanent failure must not burn wall-clock before giving up."""
    slept = _no_sleep(monkeypatch)
    mw = LLMRetryMiddleware(max_retries=3)

    error = ValueError("malformed request: unknown field")
    assert await mw.on_llm_error(LLMCallContext(), error, 0) is False
    assert slept == []


@pytest.mark.asyncio
async def test_retry_stops_at_max_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    slept = _no_sleep(monkeypatch)
    mw = LLMRetryMiddleware(max_retries=2)
    error = TimeoutError("timed out")

    assert await mw.on_llm_error(LLMCallContext(), error, 1) is True
    assert await mw.on_llm_error(LLMCallContext(), error, 2) is False
    assert len(slept) == 1, "the refused attempt must not sleep"


@pytest.mark.asyncio
async def test_retry_backoff_grows_and_is_capped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exponential with jitter, clamped — the cap is the point.

    Without ``backoff_max`` a long retry chain reaches minutes per attempt,
    which reads as a hang rather than a retry.
    """
    slept = _no_sleep(monkeypatch)
    mw = LLMRetryMiddleware(max_retries=10, backoff_base=1.0, backoff_max=4.0)
    error = TimeoutError("timed out")

    for attempt in range(5):
        assert await mw.on_llm_error(LLMCallContext(), error, attempt) is True

    # base * 2**attempt = 1, 2, 4, 8, 16 -> clamped to 4, plus <=0.25 jitter.
    assert slept[0] == pytest.approx(1.0, abs=0.25)
    assert slept[1] == pytest.approx(2.0, abs=0.25)
    assert all(d <= 4.25 for d in slept), slept
    assert slept[3] == pytest.approx(slept[4], abs=0.5), "both clamped"


# ── LLMTracingMiddleware ─────────────────────────────────────────────────


class _RecordingTrace:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def log_llm_call(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


class _RaisingTrace:
    async def log_llm_call(self, **kwargs: Any) -> None:
        raise RuntimeError("trace backend down")


@pytest.mark.asyncio
async def test_tracing_records_duration_role_and_usage() -> None:
    trace = _RecordingTrace()
    mw = LLMTracingMiddleware(trace_logger=trace)
    ctx = LLMCallContext(task_id="t1", role_id="writer", call_index=3)

    await mw.before_llm(ctx, [user_msg("hi")])
    await mw.after_llm(
        ctx,
        LLMResponse(content="ok", usage={"prompt_tokens": 11, "completion_tokens": 7}),
    )

    assert len(trace.calls) == 1
    call = trace.calls[0]
    assert call["task_id"] == "t1"
    assert call["agent_role_id"] == "writer"
    assert call["action"] == "llm_call:3"
    metadata = call["metadata"]
    assert metadata["usage"] == {"prompt_tokens": 11, "completion_tokens": 7}
    assert metadata["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_tracing_prefers_a_metadata_duration_over_the_wall_clock() -> None:
    """A streamed call's real duration is stamped by the caller.

    The middleware's own timer only covers the wrapper, so for streams it would
    under-report; ``ctx.metadata`` wins when present.
    """
    trace = _RecordingTrace()
    mw = LLMTracingMiddleware(trace_logger=trace)
    ctx = LLMCallContext(task_id="t1", metadata={"duration_ms": 4321})

    await mw.before_llm(ctx, [user_msg("hi")])
    await mw.after_llm(ctx, LLMResponse(content="ok"))

    assert trace.calls[0]["metadata"]["duration_ms"] == 4321


@pytest.mark.asyncio
async def test_tracing_surfaces_fallback_markers_when_a_chain_failed_over() -> None:
    trace = _RecordingTrace()
    mw = LLMTracingMiddleware(trace_logger=trace)
    ctx = LLMCallContext(task_id="t1", metadata={"correlation_id": "corr-9"})

    await mw.before_llm(ctx, [user_msg("hi")])
    await mw.after_llm(
        ctx,
        LLMResponse(
            content="ok",
            response_metadata={
                "fallback_used": True,
                "model_actually_used": "backup-model",
            },
        ),
    )

    metadata = trace.calls[0]["metadata"]
    assert metadata["fallback_used"] is True
    assert metadata["model_actually_used"] == "backup-model"
    assert metadata["correlation_id"] == "corr-9"


@pytest.mark.asyncio
async def test_tracing_returns_the_response_even_when_the_backend_raises() -> None:
    """Tracing is observability: it must not be able to fail the call."""
    mw = LLMTracingMiddleware(trace_logger=_RaisingTrace())
    ctx = LLMCallContext(task_id="t1")
    response = LLMResponse(content="ok")

    await mw.before_llm(ctx, [user_msg("hi")])
    assert await mw.after_llm(ctx, response) is response


@pytest.mark.asyncio
async def test_tracing_without_a_backend_is_a_no_op() -> None:
    mw = LLMTracingMiddleware()
    ctx = LLMCallContext(task_id="t1")
    response = LLMResponse(content="ok")

    await mw.before_llm(ctx, [user_msg("hi")])
    assert await mw.after_llm(ctx, response) is response


@pytest.mark.asyncio
async def test_tracing_retains_no_instance_state_when_chat_fails() -> None:
    """A terminal chat error must not leave a timer on the shared middleware."""
    mw = LLMTracingMiddleware()
    ctx = LLMCallContext(task_id="t1")

    await mw.before_llm(ctx, [user_msg("hi")])

    assert set(vars(mw)) == {"_trace"}
    assert any(key.startswith("_llm_tracing_start") for key in ctx.metadata)


# ── TokenAccountingMiddleware.persist_cost / CostPersister ───────────────


class _Tracker:
    """Minimal ``CostSink`` that also answers ``get_summary``."""

    def __init__(self, summary: dict[str, Any] | None = None) -> None:
        self.summary = summary or {
            "total_cost_usd": 1.25,
            "total_input_tokens": 100,
            "total_output_tokens": 50,
        }

    def record(
        self,
        task_id: str,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        return 0.01

    def get_summary(self, task_id: str) -> dict[str, Any]:
        return self.summary


class _Persister:
    def __init__(self) -> None:
        self.persisted: list[tuple[str, dict[str, Any], str]] = []

    async def persist(
        self, task_id: str, summary: Any, model: str,
    ) -> None:
        self.persisted.append((task_id, dict(summary), model))


class _RaisingPersister:
    async def persist(self, task_id: str, summary: Any, model: str) -> None:
        raise RuntimeError("database unreachable")


@pytest.mark.asyncio
async def test_persist_cost_hands_the_summary_and_model_to_the_persister() -> None:
    tracker = _Tracker()
    persister = _Persister()
    mw = TokenAccountingMiddleware(cost_sink=tracker, cost_persister=persister)

    ctx = LLMCallContext(task_id="task-7")
    await mw.after_llm(
        ctx,
        LLMResponse(
            content="ok",
            model="gpt-x",
            usage={"prompt_tokens": 100, "completion_tokens": 50},
        ),
    )
    await mw.persist_cost("task-7")

    assert len(persister.persisted) == 1
    task_id, summary, model = persister.persisted[0]
    assert task_id == "task-7"
    assert summary == tracker.summary
    assert model == "gpt-x", "the model observed on the call, not a re-derivation"


@pytest.mark.asyncio
async def test_persist_cost_is_a_no_op_without_a_persister() -> None:
    """The stateless path injects neither seam, and that is not an error."""
    mw = TokenAccountingMiddleware(cost_sink=_Tracker())
    await mw.persist_cost("task-7")  # must not raise

    mw_no_sink = TokenAccountingMiddleware(cost_persister=_Persister())
    await mw_no_sink.persist_cost("task-7")  # must not raise


@pytest.mark.asyncio
async def test_persist_cost_swallows_a_failing_persister() -> None:
    """Accounting is observability: a dead database must not fail the task."""
    mw = TokenAccountingMiddleware(
        cost_sink=_Tracker(), cost_persister=_RaisingPersister(),
    )
    await mw.persist_cost("task-7")  # must not raise


@pytest.mark.asyncio
async def test_persist_cost_rejects_a_sink_that_cannot_summarise() -> None:
    """Configuring persistence with an incomplete sink must fail loudly."""

    class _RecordOnly:
        def record(
            self,
            task_id: str,
            model_name: str,
            input_tokens: int,
            output_tokens: int,
        ) -> float:
            return 0.0

    persister = _Persister()
    assert not isinstance(_RecordOnly(), CostSink)
    with pytest.raises(TypeError, match=r"CostSink\.get_summary"):
        TokenAccountingMiddleware(
            cost_sink=_RecordOnly(),  # type: ignore[arg-type]
            cost_persister=persister,
        )
