from __future__ import annotations

import pytest

from agent_core.llm import LLMResponse
from agent_core.runtime.loop import context_budget
from agent_core.runtime.loop.model_profile import (
    DefaultThinkingParser,
    HistoryPolicy,
    ModelProfile,
    NativeMessageNormalizer,
    resolve_history_policy,
)


def _history(
    response: LLMResponse,
    *,
    thinking_format: str = "tag",
    policy: HistoryPolicy,
) -> dict:
    profile = ModelProfile(
        model_id="test-model",
        provider="test",
        thinking_format=thinking_format,
    )
    parsed = DefaultThinkingParser().extract(response, profile)
    return NativeMessageNormalizer().to_history(
        response,
        parsed,
        policy,
        thinking_format,
    )


@pytest.mark.parametrize(
    ("config", "enabled", "cap"),
    [
        ({}, False, None),
        ({"thinking_in_history": False}, False, None),
        ({"thinking_in_history": True}, True, None),
        ({"thinking_in_history": True, "thinking_history_max_tokens": 0}, True, None),
        ({"thinking_history_max_tokens": 8192}, True, 8192),
        ({"thinking_history_max_tokens": 8192.0}, True, 8192),
        # A quoted YAML "false" must disable, not read as a truthy string.
        ({"thinking_in_history": "false"}, False, None),
        ({"thinking_in_history": "true"}, True, None),
        ({"thinking_in_history": "no", "thinking_history_max_tokens": 8192}, False, None),
        (
            {"thinking_in_history": True, "thinking_history_max_tokens": "8192"},
            True,
            8192,
        ),
        (
            {"thinking_in_history": False, "thinking_history_max_tokens": 8192},
            False,
            None,
        ),
    ],
)
def test_resolve_history_policy(
    config: dict,
    enabled: bool,
    cap: int | None,
) -> None:
    policy = resolve_history_policy(config)

    assert policy.thinking_in_history is enabled
    assert policy.thinking_history_max_tokens == cap


@pytest.mark.parametrize("cap", [-1, "many", [], True, False, 8192.5, {}])
def test_resolve_history_policy_rejects_invalid_cap(cap: object) -> None:
    with pytest.raises(ValueError, match="thinking_history_max_tokens"):
        resolve_history_policy({"thinking_history_max_tokens": cap})


@pytest.mark.parametrize("flag", ["maybe", 3, []])
def test_resolve_history_policy_rejects_invalid_flag(flag: object) -> None:
    with pytest.raises(ValueError, match="thinking_in_history"):
        resolve_history_policy({"thinking_in_history": flag})


def test_disabled_policy_omits_tag_reasoning() -> None:
    history = _history(
        LLMResponse(content="answer", reasoning_content="private reasoning"),
        policy=HistoryPolicy(thinking_in_history=False),
    )

    assert history == {"content": "answer", "role": "assistant"}


def test_enabled_policy_keeps_full_tag_reasoning() -> None:
    history = _history(
        LLMResponse(content="answer", reasoning_content="full reasoning"),
        policy=HistoryPolicy(thinking_in_history=True),
    )

    assert history == {
        "content": "<think>full reasoning</think>\nanswer",
        "role": "assistant",
    }


def test_capped_policy_keeps_reasoning_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(context_budget, "estimate_tokens", len)

    history = _history(
        LLMResponse(content="answer", reasoning_content="discard-KEEP"),
        policy=HistoryPolicy(
            thinking_in_history=True,
            thinking_history_max_tokens=4,
        ),
    )

    assert history == {
        "content": "<think>KEEP</think>\nanswer",
        "role": "assistant",
    }


def test_inline_tag_reasoning_is_not_duplicated() -> None:
    history = _history(
        LLMResponse(content="<think>reasoning</think>\nanswer"),
        policy=HistoryPolicy(thinking_in_history=True),
    )

    assert history == {
        "content": "<think>reasoning</think>\nanswer",
        "role": "assistant",
    }


def test_disabled_policy_omits_reasoning_content_field() -> None:
    history = _history(
        LLMResponse(content="answer", reasoning_content="private reasoning"),
        thinking_format="reasoning_content",
        policy=HistoryPolicy(thinking_in_history=False),
    )

    assert history == {"content": "answer", "role": "assistant"}


def test_signed_content_blocks_ignore_text_history_policy() -> None:
    blocks = [
        {"type": "thinking", "thinking": "reasoning", "signature": "signed"},
        {"type": "text", "text": "answer"},
    ]
    history = _history(
        LLMResponse(content=blocks),
        thinking_format="content_block",
        policy=HistoryPolicy(
            thinking_in_history=False,
            thinking_history_max_tokens=1,
        ),
    )

    assert history == {"content": blocks, "role": "assistant"}


def test_multiple_inline_think_blocks_are_all_retained() -> None:
    """A turn that reopens <think> must not lose the later blocks.

    ``to_history`` rebuilds the message from the parsed result, so any block the
    parser drops is reasoning gone from history. The visible text around a block
    must also stay separated ("a<think>x</think>\nb" must not become "ab").
    """
    history = _history(
        LLMResponse(content="<think>A</think>\nstep1<think>B</think>\nstep2"),
        policy=HistoryPolicy(thinking_in_history=True),
    )

    assert history == {
        "content": "<think>A\nB</think>\nstep1\nstep2",
        "role": "assistant",
    }


def test_multiple_inline_think_blocks_are_omitted_when_disabled() -> None:
    history = _history(
        LLMResponse(content="<think>A</think>\nstep1<think>B</think>\nstep2"),
        policy=HistoryPolicy(thinking_in_history=False),
    )

    assert history == {"content": "step1\nstep2", "role": "assistant"}


def test_capped_policy_applies_to_reasoning_content_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(context_budget, "estimate_tokens", len)

    history = _history(
        LLMResponse(content="answer", reasoning_content="discard-KEEP"),
        thinking_format="reasoning_content",
        policy=HistoryPolicy(
            thinking_in_history=True,
            thinking_history_max_tokens=4,
        ),
    )

    assert history == {
        "content": "answer",
        "role": "assistant",
        "reasoning_content": "KEEP",
    }




def test_typed_reasoning_and_inline_tags_do_not_duplicate_reasoning() -> None:
    """SGLang qwen3 can fill the typed channel AND leave inline tags.

    Keeping the raw text as visible content replayed the same reasoning twice
    in history, and left inline reasoning there even when the policy disabled
    it.
    """
    response = LLMResponse(
        content="<think>inline</think>\nanswer",
        reasoning_content="typed rc",
    )
    history = _history(
        response, policy=HistoryPolicy(thinking_in_history=True)
    )
    content = str(history.get("content") or "")

    assert content.count("<think>") == 1
    assert "typed rc" in content
    assert "inline" in content
    assert "answer" in content


def test_disabled_thinking_history_strips_inline_tags_from_the_typed_path() -> None:
    response = LLMResponse(
        content="<think>inline</think>\nanswer",
        reasoning_content="typed rc",
    )
    history = _history(
        response, policy=HistoryPolicy(thinking_in_history=False)
    )
    content = str(history.get("content") or "")

    assert "<think>" not in content
    assert "inline" not in content
    assert "typed rc" not in content
    assert "answer" in content
