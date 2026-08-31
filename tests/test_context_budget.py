"""Tests for token estimation and the shared text-truncation primitive."""

from __future__ import annotations

import pytest

from agent_core.runtime.loop.context_budget import (
    estimate_tokens,
    truncate_text_to_tokens,
)


def test_estimate_tokens_english() -> None:
    assert 5 <= estimate_tokens("Hello world, this is a test sentence.") <= 15


def test_estimate_tokens_cjk() -> None:
    assert 5 <= estimate_tokens("这是一个中文测试句子") <= 20


def test_estimate_tokens_empty() -> None:
    assert estimate_tokens("") == 0


def test_truncate_no_op_returns_original_text() -> None:
    assert truncate_text_to_tokens("short", 10, estimator=len) == "short"


def test_truncate_keeps_head_and_respects_budget() -> None:
    result = truncate_text_to_tokens("0123456789", 7, marker="...", estimator=len)
    assert result == "0123..."


def test_truncate_keeps_tail_and_respects_budget() -> None:
    result = truncate_text_to_tokens(
        "0123456789",
        7,
        marker="...",
        estimator=len,
        keep="tail",
    )
    assert result == "...6789"


def test_marker_too_wide_is_dropped() -> None:
    result = truncate_text_to_tokens(
        "0123456789",
        3,
        marker="marker",
        estimator=len,
    )
    assert result == "012"


@pytest.mark.parametrize("budget", [0, -1])
def test_non_positive_budget_returns_empty(budget: int) -> None:
    assert truncate_text_to_tokens("content", budget, estimator=len) == ""


def test_invalid_keep_fails_before_truncating() -> None:
    with pytest.raises(ValueError, match="keep must be"):
        truncate_text_to_tokens(
            "content",
            3,
            estimator=len,
            keep="middle",  # type: ignore[arg-type]
        )


def _floor_div_estimator(text: str) -> int:
    """A deliberately non-additive estimator, like the module's own fallback.

    ``//4`` discards each operand's remainder, so ``est(a) + est(b)`` can come
    in below ``est(a + b)``. Every additive-estimator test above is blind to
    the marker-concatenation overshoot that this shape exposes.
    """
    return len(text) // 4


def test_additive_estimator_never_exceeds_budget() -> None:
    result = truncate_text_to_tokens("0123456789", 7, marker="...", estimator=len)
    assert len(result) <= 7


def test_non_additive_estimator_overshoots_by_at_most_one_token() -> None:
    # Copilot's case on PR #496: the slice fits the budget on its own, but the
    # returned slice+marker does not. Documented in the docstring rather than
    # fixed, because a post-hoc shrink loop would diverge from the canonical
    # upstream implementation for a one-token error.
    result = truncate_text_to_tokens(
        "0123456789",
        1,
        marker="abc",
        estimator=_floor_div_estimator,
        keep="tail",
    )
    assert result == "abc3456789"
    assert _floor_div_estimator(result) == 2  # over the budget of 1 ...
    assert _floor_div_estimator(result) <= 1 + 1  # ... but only ever by one


@pytest.mark.parametrize("keep", ["head", "tail"])
@pytest.mark.parametrize("budget", [1, 3, 7, 20])
def test_overshoot_bound_holds_across_shapes(keep: str, budget: int) -> None:
    text = "中文 mixed content 文字 " * 8
    result = truncate_text_to_tokens(
        text,
        budget,
        marker="\n[cut]",
        estimator=estimate_tokens,
        keep=keep,  # type: ignore[arg-type]
    )
    assert estimate_tokens(result) <= budget + 1
