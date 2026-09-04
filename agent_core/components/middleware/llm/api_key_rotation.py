"""In-stream API-key rotation — future placeholder for L1 same-provider
mid-stream optimization.

**Status: scaffolded, NOT used.** As of the heavy_mode provider chain
landing (docs/superpowers/specs/2026-05-12-heavy-mode-provider-chain-design.md
§10), all production fallback work — L1 same-provider key rotation
included — happens between-call at the workflow layer via
:func:`workflows.heavy_mode.utils.provider_chain.run_with_chain`.

This module remains as the explicit future home for the *mid-stream*
optimization within L1: when an in-flight LLM stream hits a retriable
error (e.g. Anthropic 529), swap the underlying httpx client's
Authorization header to the next key without tearing down the request
and losing accumulated chunks. The between-call path already handles
this correctly (it just costs one ReAct turn); mid-stream is a future
latency optimization, not a correctness fix.

Implementation pre-conditions (not met yet):

1. Sufficient production telemetry showing that between-call L1
   rotation costs measurable user-visible latency (>5s p95 added
   per rotation).
2. Validation that langchain's stream graph survives mid-stream
   credential swap without rebuild (the open question that scared
   this module off in the first place).

Until those are met: do NOT use this middleware. ``LLMProxy`` chains
in production should not include ``APIKeyRotationMiddleware``. The
NotImplementedError in :meth:`_rotate_client_credentials` is
intentional — it ensures accidental enablement fails loudly rather
than silently passing through the chain.
"""

from __future__ import annotations

import logging
from typing import Any

from agent_core.components.middleware.llm.base import (
    LLMCallContext,
    LLMMiddleware,
)
from agent_core.retry_policy import legacy_retryable

logger = logging.getLogger(__name__)

__all__ = ["APIKeyRotationMiddleware"]


_STATE_KEY = "_api_key_rotation_state"


class APIKeyRotationMiddleware(LLMMiddleware):
    """Rotate API keys on retriable errors without restarting the ReAct loop.

    Args:
        api_keys: ordered list of API keys to try. The first is the
            "primary" — used until it fails with a retriable error.
            The middleware is keyless and pass-through when the list
            has < 2 entries.
        llm: the inner langchain LLM whose credentials get mutated.
            **Currently unused** — see :meth:`_rotate_client_credentials`.
            Held so the operator filling in the placeholder has the
            client handle at hand without threading it through every
            hook call.
        max_total_rotations: cap across all calls handled by this
            middleware instance to prevent runaway retry loops if the
            error pattern is something other than auth/rate-limit.

    Lifecycle (placeholder semantics):

    - :meth:`before_llm` is a no-op. The first call uses
      ``api_keys[0]`` — whatever the LLM was constructed with.
    - :meth:`on_llm_error` checks if the error is retriable (via
      :func:`~agent_core.retry_policy.legacy_retryable`). If yes AND keys remain, calls
      :meth:`_rotate_client_credentials` to swap to the next key, then
      returns ``True`` to ask the proxy to retry. If no keys remain or
      the error is non-retriable, returns ``False`` so the outer
      cross-provider fallback (V3) can take over.
    """

    def __init__(
        self,
        *,
        api_keys: list[str],
        llm: Any = None,
        max_total_rotations: int = 8,
    ) -> None:
        if not api_keys:
            raise ValueError("api_keys must contain at least one key")
        self._api_keys: list[str] = list(api_keys)
        self._llm = llm
        self._max_total_rotations = max_total_rotations
        # Process-wide rotation counter so a runaway error pattern
        # can't burn through max_total_rotations × n_concurrent_calls.
        self._total_rotations = 0

    @property
    def name(self) -> str:
        return "api_key_rotation"

    @property
    def enabled(self) -> bool:
        # Single-key configs short-circuit the whole middleware so the
        # common case (no rotation configured) costs zero per call.
        return len(self._api_keys) >= 2

    async def on_llm_error(
        self,
        ctx: LLMCallContext,
        error: Exception,
        attempt: int,
    ) -> bool:
        """Return True to ask :class:`LLMProxy` to retry with rotated creds."""
        if not legacy_retryable(error):
            return False

        state = self._get_state(ctx)
        if state["next_idx"] >= len(self._api_keys):
            logger.warning(
                "APIKeyRotation: all %d keys exhausted (call_index=%d, "
                "attempt=%d) — yielding to outer fallback",
                len(self._api_keys), ctx.call_index, attempt,
            )
            return False

        if self._total_rotations >= self._max_total_rotations:
            logger.warning(
                "APIKeyRotation: hit max_total_rotations=%d — refusing to "
                "rotate further; outer retry/fallback should pick up",
                self._max_total_rotations,
            )
            return False

        next_idx = int(state["next_idx"])
        next_key = self._api_keys[next_idx]
        try:
            self._rotate_client_credentials(next_key)
        except Exception:
            # Rotation itself blew up — bail rather than retry into
            # a half-mutated client. Surface as warning so the operator
            # filling in the placeholder sees their stub failed.
            logger.exception(
                "APIKeyRotation: _rotate_client_credentials raised — "
                "aborting rotation, error will propagate",
            )
            return False

        state["next_idx"] = next_idx + 1
        self._total_rotations += 1
        ctx.metadata["api_key_rotation_idx"] = next_idx
        logger.warning(
            "APIKeyRotation: swapping to key #%d after retriable %s "
            "(call_index=%d, attempt=%d, total_rotations=%d)",
            next_idx, type(error).__name__, ctx.call_index, attempt,
            self._total_rotations,
        )
        return True

    # ── Placeholder — operator fills in ─────────────────────────────

    def _rotate_client_credentials(self, new_api_key: str) -> None:
        """Swap ``new_api_key`` into the underlying provider client.

        **PLACEHOLDER.** This is the one piece that depends on which
        provider wrapper we're rotating against; the operator wiring
        this middleware into production fills it in.

        OpenAI (langchain_openai.ChatOpenAI) — sketch::

            inner = self._llm
            # The httpx clients hold the Authorization header. Mutating
            # both keeps sync + async paths consistent.
            inner.openai_api_key = SecretStr(new_api_key)
            if inner.client is not None:
                inner.client.api_key = new_api_key
            if inner.async_client is not None:
                inner.async_client.api_key = new_api_key

        Anthropic — different attribute path. Provider-specific code
        lives here; the rest of the middleware is provider-agnostic.

        Raises:
            Whatever the provider client raises on a bad key handle.
            The caller turns this into "rotation aborted" rather than
            propagating into the retry loop.
        """
        del new_api_key
        # TODO(operator): wire the provider-specific credential swap.
        # Until then, raising NotImplementedError makes accidental
        # production usage loud rather than silent.
        raise NotImplementedError(
            "APIKeyRotationMiddleware._rotate_client_credentials is a "
            "placeholder. Fill in the provider-specific client mutation "
            "(see docstring) before enabling this middleware in a "
            "production profile.",
        )

    # ── State scoping ───────────────────────────────────────────────

    def _get_state(self, ctx: LLMCallContext) -> dict[str, Any]:
        """Per-call rotation cursor, keyed by ``ctx.call_index``.

        Fresh calls start from key #1 (next after primary). A single
        call may rotate up to ``len(api_keys) - 1`` times before
        falling through to the outer fallback.
        """
        bag = ctx.metadata.setdefault(_STATE_KEY, {})
        key = ctx.call_index
        if key not in bag:
            bag[key] = {"next_idx": 1}
        return bag[key]
