"""Every runaway recovery path must preserve the Anthropic request boundary."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from agent_core.messages import Message, assistant_msg, for_wire, system_msg, tool_msg, user_msg
from agent_core.providers.anthropic import AnthropicClient
from agent_core.runtime.loop import _call
from agent_core.runtime.loop._bind import bind_max_tokens


@pytest.mark.parametrize("retry_path", [
    "completed_chat", "completed_stream", "early_stream", "overflow_chat", "overflow_stream",
])
@pytest.mark.parametrize("history_kind", ["plain_user", "parallel_tools"])
@pytest.mark.parametrize("with_addendum", [False, True])
@pytest.mark.parametrize("cache_enabled", [False, True])
@pytest.mark.asyncio
async def test_all_runaway_retries_fold_without_caching_or_persisting_reminders(
    monkeypatch: pytest.MonkeyPatch,
    retry_path: str,
    history_kind: str,
    with_addendum: bool,
    cache_enabled: bool,
) -> None:
    monkeypatch.setenv("ANTHROPIC_PROMPT_CACHE", "1" if cache_enabled else "0")
    monkeypatch.setattr(_call, "_RUNAWAY_MAX_RETRIES", 3)
    monkeypatch.setattr(_call, "_RUNAWAY_EXPAND_ENABLED", True)
    monkeypatch.setattr(_call, "_RUNAWAY_BACKOFF_S", 0.0)
    client = AnthropicClient("claude-test", api_key="test-key")
    messages = [system_msg("system"), user_msg("question")]
    if history_kind == "parallel_tools":
        messages += [
            assistant_msg("", tool_calls=[{
                "id": call_id, "type": "function",
                "function": {"name": "bash", "arguments": "{}"},
            } for call_id in ("a", "b")]),
            tool_msg("result a", "a"), tool_msg("result b", "b"),
        ]
    if with_addendum:
        messages.append({**user_msg("[env]"), "transient": True})
    original = deepcopy(messages)
    projections: list[list[Message]] = []
    build_kwargs = client._build_kwargs

    def capture_projection(request: list[Message], **kwargs: Any) -> dict[str, Any]:
        projections.append(deepcopy(request))
        return build_kwargs(request, **kwargs)

    monkeypatch.setattr(client, "_build_kwargs", capture_projection)
    wire_requests: list[dict[str, Any]] = []

    async def create(**kwargs: Any) -> Any:
        wire_requests.append(deepcopy(kwargs))
        attempt = len(wire_requests)
        if retry_path.startswith("overflow") and attempt == 2:
            raise RuntimeError("maximum context length exceeded")
        done = attempt == 4
        if not kwargs.get("stream"):
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="done")] if done else [
                    SimpleNamespace(type="thinking", thinking="x" * 800, signature="sig"),
                ],
                stop_reason="end_turn" if done else "max_tokens",
                model="claude-test", id=f"response-{attempt}",
                usage=SimpleNamespace(input_tokens=10, output_tokens=2 if done else 2048),
            )

        async def events() -> Any:
            block_type = "text" if done else "thinking"
            yield SimpleNamespace(
                type="content_block_start", index=0,
                content_block=SimpleNamespace(type=block_type),
            )
            yield SimpleNamespace(
                type="content_block_delta", index=0,
                delta=SimpleNamespace(
                    type="text_delta" if done else "thinking_delta",
                    text="done" if done else "", thinking="" if done else "x" * 800,
                ),
            )
            yield SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="end_turn" if done else "max_tokens"),
                usage=SimpleNamespace(output_tokens=2 if done else 2048),
            )

        return events()

    client._client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(side_effect=create)))
    attempts: list[dict[str, Any]] = []

    async def on_attempt(event: dict[str, Any]) -> None:
        attempts.append(event)

    async def on_delta(*_args: Any, **_kwargs: Any) -> None:
        pass

    response = await _call.call_llm(
        bind_max_tokens(client, 2048), messages, timeout=30, max_retries=4, turn=1,
        on_delta=on_delta if retry_path.endswith("stream") else None,
        reasoning_only_max_tokens=100 if retry_path == "early_stream" else None,
        on_attempt=on_attempt,
    )
    assert response is not None and response.content == "done"
    assert len(wire_requests) == len(projections) == 4
    # Exercise all three retry phases, including the context-overflow downgrade.
    started = [event for event in attempts if event["phase"] == "started"]
    assert [event["thinking_mode"] for event in started] == [
        "profile_default", "expanded", "reduced", "disabled",
    ]
    finished = [event for event in attempts if event["phase"] == "finished"]
    if retry_path == "early_stream":
        expected_reasons = ["reasoning_runaway_early"] * 3 + [""]
    elif retry_path.startswith("overflow"):
        expected_reasons = ["reasoning_runaway", "context_length", "reasoning_runaway", ""]
    else:
        expected_reasons = ["reasoning_runaway"] * 3 + [""]
    assert [event["reason"] for event in finished] == expected_reasons
    assert projections[0] == original
    assert messages == original
    for projection, request in zip(projections[1:], wire_requests[1:], strict=True):
        # Each retry replaces the reminder; neither the addendum nor reminders
        # accumulate in history or leak their bookkeeping onto the wire.
        assert projection[:-1] == original
        reminder = projection[-1]
        assert reminder["role"] == "user" and reminder["transient"] is True
        assert str(reminder["content"]).startswith("[system reminder]")
        assert "transient" not in for_wire([reminder])[0]
        wire = request["messages"]
        assert all("transient" not in message for message in wire)
        assert [message["role"] for message in wire] == (
            ["user", "assistant", "user"] if history_kind == "parallel_tools" else ["user"]
        )
        content = wire[-1]["content"]
        persistent = [
            {"type": "tool_result", "tool_use_id": "a", "content": "result a"},
            {"type": "tool_result", "tool_use_id": "b", "content": "result b"},
        ] if history_kind == "parallel_tools" else [{"type": "text", "text": "question"}]
        if cache_enabled:
            persistent[-1]["cache_control"] = {"type": "ephemeral"}
        assert content == [
            *persistent,
            *([{"type": "text", "text": "[env]"}] if with_addendum else []),
            {"type": "text", "text": reminder["content"]},
        ]
        assert [i for i, block in enumerate(content) if "cache_control" in block] == (
            [len(persistent) - 1] if cache_enabled else []
        )
    # Rebuilding the original history for another logical call drops every reminder.
    next_call = build_kwargs(
        messages, tools=None, temperature=None, max_tokens=None, extra_headers=None, timeout=None,
    )
    assert next_call["messages"] == wire_requests[0]["messages"]
