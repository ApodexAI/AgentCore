"""``APIKeyRotationMiddleware`` — placeholder, structural tests only.

The provider-specific credential swap (``_rotate_client_credentials``)
is a stub that the operator fills in before production use. These tests
lock the surrounding structure so the operator's edit can land without
worrying that the rotation cursor or retriable classifier silently
regressed.

What we DO test:
- enabled-property gates on key-count
- on_llm_error rotates the cursor when the error is retriable
- non-retriable errors don't rotate
- exhausted keys yield to the outer fallback
- max_total_rotations cap fires
- placeholder _rotate_client_credentials raises NotImplementedError
- per-call_index state isolation

What we do NOT test:
- actual mid-stream HTTP retry (needs a live 529-on-demand target)
- provider-client mutation (placeholder; operator fills in)
"""

from __future__ import annotations

import pytest

from agent_core.components.middleware.llm.api_key_rotation import (
    APIKeyRotationMiddleware,
)
from agent_core.components.middleware.llm.base import LLMCallContext


class _MockRotation(APIKeyRotationMiddleware):
    """Subclass that stubs the placeholder so we can test the framing."""

    def __init__(self, *, api_keys, **kw) -> None:
        super().__init__(api_keys=api_keys, **kw)
        self.rotated_to: list[str] = []
        self.rotate_should_raise: Exception | None = None

    def _rotate_client_credentials(self, new_api_key: str) -> None:
        if self.rotate_should_raise is not None:
            raise self.rotate_should_raise
        self.rotated_to.append(new_api_key)


# ── Constructor + enabled gating ────────────────────────────────────


def test_constructor_rejects_empty_key_list() -> None:
    """Locks the contract: at least one key required. An empty list
    would silently disable rotation while looking configured."""
    with pytest.raises(ValueError):
        APIKeyRotationMiddleware(api_keys=[])


def test_single_key_disables_middleware() -> None:
    """One key = nothing to rotate to. The middleware short-circuits
    via ``enabled = False`` so the proxy skips it without per-call
    overhead. Common case: a profile with no fallback keys configured."""
    mw = APIKeyRotationMiddleware(api_keys=["only-key"])
    assert mw.enabled is False


def test_multiple_keys_enables_middleware() -> None:
    mw = APIKeyRotationMiddleware(api_keys=["a", "b", "c"])
    assert mw.enabled is True


# ── Rotation behaviour ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retriable_error_rotates_to_next_key() -> None:
    """Retriable error → cursor advances + rotation callback fires →
    middleware asks proxy to retry."""
    mw = _MockRotation(api_keys=["k0", "k1", "k2"])
    ctx = LLMCallContext(call_index=0)
    retry = await mw.on_llm_error(ctx, TimeoutError("upstream 529"), attempt=0)
    assert retry is True
    assert mw.rotated_to == ["k1"]
    assert ctx.metadata["api_key_rotation_idx"] == 1


@pytest.mark.asyncio
async def test_non_retriable_error_does_not_rotate() -> None:
    """ValueError isn't in the retriable keyword set → middleware
    yields immediately. Lets structural bugs propagate fast instead
    of wasting keys."""
    mw = _MockRotation(api_keys=["k0", "k1", "k2"])
    ctx = LLMCallContext(call_index=0)
    retry = await mw.on_llm_error(ctx, ValueError("bad request"), attempt=0)
    assert retry is False
    assert mw.rotated_to == []


@pytest.mark.asyncio
async def test_exhausted_keys_yields_to_outer_fallback() -> None:
    """After the last key fails, return False so V3 cross-provider
    fallback (in heavy_reporter._run_with_fallback_keys) can fire."""
    mw = _MockRotation(api_keys=["k0", "k1"])
    ctx = LLMCallContext(call_index=0)

    # First failure rotates to k1.
    assert await mw.on_llm_error(ctx, TimeoutError("529"), 0) is True
    # Second failure: nothing left, yield.
    assert await mw.on_llm_error(ctx, TimeoutError("529"), 1) is False
    assert mw.rotated_to == ["k1"]


@pytest.mark.asyncio
async def test_max_total_rotations_caps_runaway_loops() -> None:
    """Across many concurrent calls, the middleware refuses to rotate
    forever if the underlying error pattern isn't actually auth-related.
    Prevents burning through quota chasing a structural bug."""
    mw = _MockRotation(api_keys=["k0", "k1", "k2", "k3", "k4"], max_total_rotations=2)
    err = TimeoutError("rate limit")

    # Two rotations across two distinct calls — both succeed.
    ctx_a = LLMCallContext(call_index=0)
    assert await mw.on_llm_error(ctx_a, err, 0) is True
    ctx_b = LLMCallContext(call_index=1)
    assert await mw.on_llm_error(ctx_b, err, 0) is True

    # Third rotation hits the cap; middleware yields.
    ctx_c = LLMCallContext(call_index=2)
    assert await mw.on_llm_error(ctx_c, err, 0) is False
    assert len(mw.rotated_to) == 2  # capped


@pytest.mark.asyncio
async def test_rotation_callback_raising_aborts_rotation_attempt() -> None:
    """If the operator's ``_rotate_client_credentials`` blows up, the
    middleware must NOT ask the proxy to retry into a half-mutated
    client. Yield to outer fallback instead."""
    mw = _MockRotation(api_keys=["k0", "k1"])
    mw.rotate_should_raise = RuntimeError("provider client broke")
    ctx = LLMCallContext(call_index=0)
    retry = await mw.on_llm_error(ctx, TimeoutError("529"), 0)
    assert retry is False


@pytest.mark.asyncio
async def test_state_is_per_call_index() -> None:
    """Concurrent streams must each start from key #1 — one call's
    cursor advancement must not leak to another's."""
    mw = _MockRotation(api_keys=["k0", "k1", "k2"])
    metadata: dict = {}
    ctx_a = LLMCallContext(call_index=0, metadata=metadata)
    ctx_b = LLMCallContext(call_index=1, metadata=metadata)

    await mw.on_llm_error(ctx_a, TimeoutError("529"), 0)
    await mw.on_llm_error(ctx_b, TimeoutError("529"), 0)

    # Both rotated to k1 — they don't share a cursor.
    assert mw.rotated_to == ["k1", "k1"]


# ── Placeholder enforcement ─────────────────────────────────────────


def test_default_rotate_implementation_raises_NotImplementedError() -> None:
    """The base class refuses to silently no-op. An operator wiring
    this middleware in production must override the method; forgetting
    is a loud failure, not a stealth one."""
    mw = APIKeyRotationMiddleware(api_keys=["k0", "k1"])
    with pytest.raises(NotImplementedError):
        mw._rotate_client_credentials("k1")


@pytest.mark.asyncio
async def test_unstubbed_on_llm_error_swallows_NotImplementedError_and_yields() -> None:
    """An operator who forgets to override the method shouldn't crash
    the LLM call — the middleware should log + yield to outer fallback.
    Production safety net for the partial-implementation state."""
    mw = APIKeyRotationMiddleware(api_keys=["k0", "k1"])
    ctx = LLMCallContext(call_index=0)
    # _rotate_client_credentials raises NotImplementedError; on_llm_error
    # catches the placeholder exception and yields.
    retry = await mw.on_llm_error(ctx, TimeoutError("529"), 0)
    assert retry is False
