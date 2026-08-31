"""``extract_usage`` normalizes both usage shapes to OpenAI raw form.

The downstream observers (``ProtocolStreamObserver`` for SSE §4.4,
``WorkerTraceFileObserver`` for the trace file §6.0) read ``ctx.usage``
with keys ``provider`` / ``model`` / ``prompt_tokens`` /
``completion_tokens`` / ``cache_read_tokens`` / ``cache_write_tokens`` /
``cached_tokens`` / ``cache_creation_tokens`` / ``reasoning_tokens``.
``provider`` comes from ``response_metadata.provider_actually_used``
(stamped by a provider-chain wrapper per attempt) — empty string when no
chain wrapper stamped it. Both the legacy canonical ``usage_metadata``
(``input_tokens`` / ``output_tokens``, no ``model``) and OpenAI raw
``response_metadata.token_usage`` must fold to that target shape.

2026-05-28 cache schema split:

The schema previously exposed only ``cached_tokens`` (cache READ only)
and ``cache_creation_tokens`` (cache WRITE). Cost boards using a single
``cached_tokens`` field with a single rate would under-attribute
Anthropic cache write spend (~12× rate difference between read and
write). Post-split the schema carries four fields:

- ``cache_read_tokens`` — explicit cache READ count
- ``cache_write_tokens`` — explicit cache WRITE count (incl. Anthropic
  1h-TTL extension summed into write)
- ``cached_tokens`` — backward-compat sum (``read + write``);
  **semantic change** from pre-split (was read-only)
- ``cache_creation_tokens`` — backward-compat alias of
  ``cache_write_tokens`` (deprecated; prefer the new name)
"""
from __future__ import annotations

from types import SimpleNamespace

from agent_core.llm import LLMResponse
from agent_core.runtime.loop._response import _pick_int
from agent_core.runtime.loop.llm_client import extract_usage


def _resp(*, usage_metadata=None, response_metadata=None):
    return SimpleNamespace(
        usage_metadata=usage_metadata,
        response_metadata=response_metadata,
    )


def test_returns_none_when_no_usage_anywhere() -> None:
    assert extract_usage(_resp()) is None
    assert extract_usage(_resp(response_metadata={})) is None
    # Both empty / 0 → also None (no signal to record).
    assert extract_usage(_resp(usage_metadata={"input_tokens": 0,
                                               "output_tokens": 0})) is None


def test_native_response_zero_fills_reasoning_tokens() -> None:
    usage = extract_usage(LLMResponse(
        model="native-model",
        usage={"prompt_tokens": 7, "completion_tokens": 3},
    ))

    assert usage == {
        "provider": "",
        "model": "native-model",
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "cached_tokens": 0,
        "cache_creation_tokens": 0,
        "reasoning_tokens": 0,
    }


def test_canonical_usage_metadata_shape() -> None:
    # Canonical input/output token fields; model lives in response_metadata.
    resp = _resp(
        usage_metadata={
            "input_tokens": 120,
            "output_tokens": 45,
            "input_token_details": {"cache_read": 30},
        },
        response_metadata={"model_name": "qwen35_397b_a17b"},
    )
    u = extract_usage(resp)
    assert u == {
        "model": "qwen35_397b_a17b",
        "prompt_tokens": 120,
        "completion_tokens": 45,
        "cache_read_tokens": 30,
        "cache_write_tokens": 0,
        "cached_tokens": 30,  # = read + write = 30 + 0
        "cache_creation_tokens": 0,  # alias of cache_write_tokens
        "reasoning_tokens": 0,
        "provider": "",
    }


def test_openai_raw_token_usage_shape() -> None:
    # response_metadata.token_usage with prompt_tokens / completion_tokens.
    resp = _resp(
        usage_metadata=None,
        response_metadata={
            "model_name": "gpt-4o",
            "token_usage": {
                "prompt_tokens": 200,
                "completion_tokens": 60,
                "prompt_tokens_details": {"cached_tokens": 50},
            },
        },
    )
    u = extract_usage(resp)
    assert u == {
        "model": "gpt-4o",
        "prompt_tokens": 200,
        "completion_tokens": 60,
        "cache_read_tokens": 50,
        "cache_write_tokens": 0,
        "cached_tokens": 50,  # = read + write
        "cache_creation_tokens": 0,
        "reasoning_tokens": 0,
        "provider": "",
    }


def test_usage_metadata_takes_precedence_over_token_usage() -> None:
    """When both shapes are present, prefer the canonical metadata one."""
    resp = _resp(
        usage_metadata={"input_tokens": 1, "output_tokens": 2},
        response_metadata={
            "model_name": "m",
            "token_usage": {"prompt_tokens": 999, "completion_tokens": 999},
        },
    )
    u = extract_usage(resp)
    assert u == {
        "model": "m",
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "cached_tokens": 0,
        "cache_creation_tokens": 0,
        "reasoning_tokens": 0,
        "provider": "",
    }


def test_apodex_nonstream_cached_falls_through_to_raw_shape() -> None:
    """Apodex on ``ainvoke`` fills the canonical ``input_tokens`` /
    ``output_tokens`` but leaves ``input_token_details`` empty — cached
    counts only land on the raw ``prompt_tokens_details.cached_tokens``.
    Without the raw-shape fallback, cached tokens get silently dropped
    on every non-streaming call (DAG analyzer, decision_llm, synth_llm).
    """
    resp = _resp(
        usage_metadata={"input_tokens": 5000, "output_tokens": 1000},
        response_metadata={
            "model_name": "mirothinker_v20_397b",
            "provider_actually_used": "apodex",
            "token_usage": {
                "prompt_tokens": 5000,
                "completion_tokens": 1000,
                "prompt_tokens_details": {"cached_tokens": 2000},
            },
        },
    )
    u = extract_usage(resp)
    assert u is not None
    assert u["prompt_tokens"] == 5000
    assert u["completion_tokens"] == 1000
    assert u["cached_tokens"] == 2000
    assert u["provider"] == "apodex"


def test_response_metadata_alt_keys() -> None:
    """``response_metadata.usage`` is a valid alias for ``token_usage``;
    ``model`` is a valid alias for ``model_name``."""
    resp = _resp(
        usage_metadata=None,
        response_metadata={
            "model": "alt-name",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )
    assert extract_usage(resp) == {
        "model": "alt-name",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "cached_tokens": 0,
        "cache_creation_tokens": 0,
        "reasoning_tokens": 0,
        "provider": "",
    }


def test_canonical_cache_creation_and_reasoning_tokens() -> None:
    """Canonical cache creation and reasoning token detail fields."""
    resp = _resp(
        usage_metadata={
            "input_tokens": 500,
            "output_tokens": 200,
            "input_token_details": {"cache_read": 50, "cache_creation": 400},
            "output_token_details": {"reasoning": 120},
        },
        response_metadata={"model_name": "claude-opus-4-7"},
    )
    u = extract_usage(resp)
    assert u == {
        "model": "claude-opus-4-7",
        "prompt_tokens": 500,
        "completion_tokens": 200,
        "cache_read_tokens": 50,
        "cache_write_tokens": 400,
        "cached_tokens": 450,  # = 50 read + 400 write
        "cache_creation_tokens": 400,  # alias of cache_write_tokens
        "reasoning_tokens": 120,
        "provider": "",
    }


def test_openai_raw_cache_creation_and_reasoning_tokens() -> None:
    """Anthropic raw exposes ``cache_creation_input_tokens`` directly on
    the usage dict; OpenAI o-series nests reasoning under
    ``completion_tokens_details.reasoning_tokens``."""
    resp = _resp(
        usage_metadata=None,
        response_metadata={
            "model_name": "o1-preview",
            "token_usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "prompt_tokens_details": {"cached_tokens": 200},
                "completion_tokens_details": {"reasoning_tokens": 300},
                "cache_creation_input_tokens": 600,
            },
        },
    )
    u = extract_usage(resp)
    assert u == {
        "model": "o1-preview",
        "prompt_tokens": 1000,
        "completion_tokens": 500,
        "cache_read_tokens": 200,
        "cache_write_tokens": 600,
        "cached_tokens": 800,  # = 200 read + 600 write
        "cache_creation_tokens": 600,
        "reasoning_tokens": 300,
        "provider": "",
    }


def test_openrouter_cache_write_tokens_alias() -> None:
    """Claude routed through OpenRouter's OpenAI-compatible wrap exposes
    cache-creation tokens under ``prompt_tokens_details.cache_write_tokens``
    instead of the OpenAI-standard ``cache_creation_tokens``. Observed live
    on ``api.miromind.site/v1 → openrouter`` for ``claude-sonnet-4.6``.
    The fallback chain must pick this alias up so cache_create lands in the
    rollup when prompt caching is eventually enabled."""
    resp = _resp(
        usage_metadata=None,
        response_metadata={
            "model_name": "anthropic/claude-sonnet-4.6",
            "token_usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 50,
                "prompt_tokens_details": {
                    "cached_tokens": 120,
                    "cache_write_tokens": 480,
                },
            },
        },
    )
    u = extract_usage(resp)
    assert u == {
        "model": "anthropic/claude-sonnet-4.6",
        "prompt_tokens": 1000,
        "completion_tokens": 50,
        "cache_read_tokens": 120,
        "cache_write_tokens": 480,
        "cached_tokens": 600,  # = 120 read + 480 write
        "cache_creation_tokens": 480,
        "reasoning_tokens": 0,
        "provider": "",
    }


def test_qwen_null_cached_tokens_normalises_to_zero() -> None:
    """Qwen3.5-397b via apodex gateway returns
    ``prompt_tokens_details.cached_tokens = null`` (the gateway does not
    surface prefix-cache info even when the SGLang/vLLM backend has it).
    ``dict.get`` returns None for that key, the ``or 0`` chain must
    normalise it without crashing on ``int(None)``."""
    resp = _resp(
        usage_metadata=None,
        response_metadata={
            "model_name": "qwen3.5-397b-a17b",
            "token_usage": {
                "prompt_tokens": 1616,
                "completion_tokens": 505,
                "prompt_tokens_details": {
                    "audio_tokens": None,
                    "cached_tokens": None,
                    "text_tokens": 1616,
                },
                "completion_tokens_details": {
                    "reasoning_tokens": 499,
                },
            },
        },
    )
    u = extract_usage(resp)
    assert u == {
        "model": "qwen3.5-397b-a17b",
        "prompt_tokens": 1616,
        "completion_tokens": 505,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "cached_tokens": 0,
        "cache_creation_tokens": 0,
        "reasoning_tokens": 499,
        "provider": "",
    }


def test_provider_actually_used_stamped_by_chain() -> None:
    """When a provider-chain wrapper writes ``provider_actually_used``,
    extract_usage surfaces it as ``provider``
    so downstream billing can attribute the call per vendor."""
    resp = _resp(
        usage_metadata={"input_tokens": 100, "output_tokens": 20},
        response_metadata={
            "model_name": "claude-opus-4-7",
            "provider_actually_used": "anthropic",
            "model_actually_used": "claude-opus-4-7",
            "fallback_used": 1,
        },
    )
    u = extract_usage(resp)
    assert u is not None
    assert u["provider"] == "anthropic"
    assert u["model"] == "claude-opus-4-7"


def test_anthropic_direct_cache_read_input_tokens() -> None:
    """Anthropic Messages API exposes ``cache_read_input_tokens`` at the
    ``usage`` root (not nested under ``prompt_tokens_details`` the way
    OpenAI does). When Claude is hit *directly* (not via openrouter,
    which strips the standard names — see
    ``test_openrouter_cache_write_tokens_alias``) the read count lands
    there. Symmetric with ``cache_creation_input_tokens`` which we
    already accept at the same level."""
    resp = _resp(
        usage_metadata=None,
        response_metadata={
            "model_name": "claude-sonnet-4-6",
            "token_usage": {
                "prompt_tokens": 1500,
                "completion_tokens": 200,
                "cache_read_input_tokens": 800,
                "cache_creation_input_tokens": 450,
            },
        },
    )
    assert extract_usage(resp) == {
        "model": "claude-sonnet-4-6",
        "prompt_tokens": 1500,
        "completion_tokens": 200,
        "cache_read_tokens": 800,
        "cache_write_tokens": 450,
        "cached_tokens": 1250,  # = 800 read + 450 write
        "cache_creation_tokens": 450,
        "reasoning_tokens": 0,
        "provider": "",
    }


def test_apodex_qwen_nests_cache_creation_under_prompt_tokens_details() -> None:
    """Apodex / qwen3.5 gateway shape: BOTH cache fields nested under
    ``prompt_tokens_details`` instead of split (read nested, write at
    root) as Anthropic-direct does.

    Observed shape (2026-05) — note the write key is the Anthropic name
    ``cache_creation_input_tokens`` BUT lives under ``prompt_tokens_details``,
    not at the usage root::

        usage:
          prompt_tokens: 12000
          completion_tokens: 500
          prompt_tokens_details:
            cached_tokens: 8000           # cache read
            cache_creation_input_tokens: 3500  # cache write

    Without the ``ptd.cache_creation_input_tokens`` fallback the write
    count would silently drop on every apodex non-streaming call — same
    class of regression as the apodex cached-tokens cross-check fix
    that already exists for ``cached_tokens``.
    """
    resp = _resp(
        usage_metadata=None,
        response_metadata={
            "model_name": "qwen3.5-397b-a17b",
            "provider_actually_used": "apodex",
            "token_usage": {
                "prompt_tokens": 12000,
                "completion_tokens": 500,
                "prompt_tokens_details": {
                    "cached_tokens": 8000,
                    "cache_creation_input_tokens": 3500,
                },
            },
        },
    )
    u = extract_usage(resp)
    assert u is not None
    assert u["cache_read_tokens"] == 8000
    assert u["cache_write_tokens"] == 3500, (
        "apodex/qwen nest cache_creation_input_tokens under "
        "prompt_tokens_details — the extractor must accept this alias"
    )
    assert u["cached_tokens"] == 11500  # 8000 read + 3500 write


def test_bedrock_via_openai_gateway_nests_anthropic_cache_read_under_ptd() -> None:
    """Bedrock served through an OpenAI-compatible gateway (provider_class
    ``OpenAIClient``) nests BOTH Anthropic cache keys under
    ``prompt_tokens_details`` — the WRITE key
    ``cache_creation_input_tokens`` AND the READ key
    ``cache_read_input_tokens``.

    Regression for 2026-05-29: the read candidate list lacked the nested
    ``cache_read_input_tokens`` (while the write list had its nested
    counterpart), so every bedrock-via-gateway call recorded
    ``cache_read_tokens == 0`` even on a warm second call. The reported
    symptom was ``read=0 / write=24163`` in ``usage_summary``.
    """
    body = {
        "model_name": "global.anthropic.claude-sonnet-4-6",
        "provider_actually_used": "bedrock",
        "token_usage": {
            "prompt_tokens": 31538,
            "completion_tokens": 1993,
            "prompt_tokens_details": {
                "cache_read_input_tokens": 24000,
                "cache_creation_input_tokens": 24163,
            },
        },
    }
    # Raw-only shape (streaming usage chunk: no usage_metadata).
    u = extract_usage(_resp(usage_metadata=None, response_metadata=body))
    assert u is not None
    assert u["cache_read_tokens"] == 24000, (
        "nested Anthropic cache_read_input_tokens must be picked up — "
        "mirror of the write path's nested candidate"
    )
    assert u["cache_write_tokens"] == 24163

    # Canonical-metadata-present shape (ainvoke: usage_metadata populated
    # but input_token_details empty — the cross-check must still find the
    # nested read key).
    u2 = extract_usage(_resp(
        usage_metadata={"input_tokens": 31538, "output_tokens": 1993},
        response_metadata=body,
    ))
    assert u2 is not None
    assert u2["cache_read_tokens"] == 24000
    assert u2["cache_write_tokens"] == 24163


def test_apodex_canonical_path_finds_nested_cache_creation() -> None:
    """A gateway shape with canonical metadata populated but
    empty details — fallback to ptd.cache_creation_input_tokens MUST
    still pick up the write count via the cross-check that the canonical
    path runs when its own details came back empty.
    """
    resp = _resp(
        usage_metadata={
            "input_tokens": 12000,
            "output_tokens": 500,
            # Canonical details intentionally empty — apodex gateway
            # populates only the raw shape's nested details.
        },
        response_metadata={
            "model_name": "qwen3.5-397b-a17b",
            "provider_actually_used": "apodex",
            "token_usage": {
                "prompt_tokens": 12000,
                "completion_tokens": 500,
                "prompt_tokens_details": {
                    "cached_tokens": 8000,
                    "cache_creation_input_tokens": 3500,
                },
            },
        },
    )
    u = extract_usage(resp)
    assert u is not None
    assert u["cache_read_tokens"] == 8000
    assert u["cache_write_tokens"] == 3500


def test_anthropic_1h_ttl_extension_sums_into_cache_write() -> None:
    """Anthropic's 1h-TTL prompt-cache surfaces under a nested
    ``cache_creation`` dict alongside the standard 5m count.

    Both bill as cache WRITE (5m ~1.25× base, 1h ~2× base) — sum them
    into ``cache_write_tokens`` so the schema field captures the full
    write footprint. Boards needing per-TTL breakdown should consume
    the raw provider response directly; the schema picks the practical
    "all writes" rollup for the same reason ``cached_tokens`` is now
    "all cache activity".
    """
    resp = _resp(
        usage_metadata=None,
        response_metadata={
            "model_name": "claude-opus-4-7",
            "token_usage": {
                "prompt_tokens": 5000,
                "completion_tokens": 800,
                "cache_read_input_tokens": 1200,
                "cache_creation_input_tokens": 600,  # 5m TTL
                "cache_creation": {
                    "ephemeral_1h_input_tokens": 400,  # 1h TTL
                },
            },
        },
    )
    u = extract_usage(resp)
    assert u is not None
    assert u["cache_read_tokens"] == 1200
    assert u["cache_write_tokens"] == 600 + 400  # 5m + 1h summed
    assert u["cached_tokens"] == 1200 + 600 + 400  # read + write sum
    assert u["cache_creation_tokens"] == 1000  # alias of cache_write_tokens


def test_cached_tokens_backward_compat_is_sum_not_read_only() -> None:
    """**Schema semantic change (2026-05-28)**: ``cached_tokens`` is now
    the sum of cache read + cache write, not cache-read-only.

    Pre-split this field exposed only cache-read counts; cost boards
    using a single rate on ``cached_tokens`` would have under-attributed
    Anthropic write spend by ~12× (write bills at ~1.25× base vs
    read's ~0.1×). Post-split the same name returns the intuitive
    total so boards using the shortcut over-attribute (preferable to
    under-attributing).

    This test exists explicitly to lock in the new semantics — if a
    future change reverts ``cached_tokens`` to read-only, this test
    blocks it.
    """
    resp = _resp(
        usage_metadata=None,
        response_metadata={
            "model_name": "claude-sonnet-4-6",
            "token_usage": {
                "prompt_tokens": 10_000,
                "completion_tokens": 2_000,
                "cache_read_input_tokens": 3_000,
                "cache_creation_input_tokens": 2_500,
            },
        },
    )
    u = extract_usage(resp)
    assert u is not None
    assert u["cached_tokens"] == 5_500, (
        "cached_tokens must equal cache_read + cache_write post-split"
    )
    # Explicit split fields are the same numbers from the raw response.
    assert u["cache_read_tokens"] == 3_000
    assert u["cache_write_tokens"] == 2_500


def test_pick_int_skips_none_and_invalid_falls_through_to_real_signal() -> None:
    """``_pick_int`` powers the multi-source token extraction. It must
    skip ``None`` (gateway-side ``null``), unparseable values, and
    zeros — only stopping on the first non-zero int. This is the
    contract that lets us list provider field aliases in priority
    order without each one having to special-case ``null``."""
    assert _pick_int(None, None, 42) == 42
    assert _pick_int(0, 0, 100) == 100
    assert _pick_int("not-an-int", None, 7) == 7
    assert _pick_int(None) == 0
    assert _pick_int() == 0
    # Numeric string is coerced — providers occasionally JSON-decode
    # token counts as strings on edge cases.
    assert _pick_int("123") == 123


def test_usage_metadata_object_coerces_to_dict() -> None:
    """A compatibility adapter may return an object for ``usage_metadata``;
    coerce it via ``dict(...)`` rather than requiring a concrete dict."""
    class UM:
        def keys(self):
            return ("input_tokens", "output_tokens")

        def __getitem__(self, k):
            return {"input_tokens": 7, "output_tokens": 3}[k]

    resp = _resp(usage_metadata=UM(),
                 response_metadata={"model_name": "x"})
    u = extract_usage(resp)
    assert u is not None
    assert u["prompt_tokens"] == 7
    assert u["completion_tokens"] == 3
    assert u["model"] == "x"
