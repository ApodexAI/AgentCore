from __future__ import annotations

from agent_core.messages import (
    assistant_msg,
    assistant_msg_with_reasoning,
    for_wire,
    system_msg,
    text_of,
    tool_msg,
    user_msg,
)


def test_builders_preserve_wire_key_order() -> None:
    assert list(system_msg("system")) == ["content", "role"]
    assert list(user_msg("user")) == ["content", "role"]
    assert list(assistant_msg("answer")) == ["content", "role"]
    assert list(tool_msg("result", "call-1")) == [
        "content",
        "role",
        "tool_call_id",
    ]


def test_for_wire_removes_only_process_metadata_without_mutating_input() -> None:
    message = assistant_msg("answer")
    message["duration_ms"] = 12
    message["spill_refs"] = ["/tmp/spill"]

    wire = for_wire([message])

    assert wire == [{"content": "answer", "role": "assistant"}]
    assert message["duration_ms"] == 12


def test_for_wire_returns_same_list_when_already_clean() -> None:
    messages = [user_msg("hello"), assistant_msg("hi")]
    assert for_wire(messages) is messages


def test_reasoning_policy_is_explicit() -> None:
    tagged = assistant_msg_with_reasoning("answer", "private", thinking_format="tag")
    assert tagged["content"] == "<think>private</think>\nanswer"
    assert "reasoning_content" not in tagged

    native = assistant_msg_with_reasoning(
        "answer",
        "private",
        thinking_format="reasoning_content",
    )
    assert native["reasoning_content"] == "private"


def test_text_of_flattens_content_blocks() -> None:
    assert (
        text_of(
            [
                {"type": "text", "text": "visible"},
                "tail",
            ]
        )
        == "visible\ntail"
    )
