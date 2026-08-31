from __future__ import annotations

from collections.abc import AsyncIterator

from agent_core.llm import LLMClient, LLMResponse, StreamDelta
from agent_core.messages import Message


def test_response_containers_do_not_share_mutable_defaults() -> None:
    first = LLMResponse()
    second = LLMResponse()
    first.tool_calls.append(
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "search", "arguments": "{}"},
        }
    )
    first.usage["input_tokens"] = 1

    assert second.tool_calls == []
    assert second.usage == {}
    assert second.response_metadata == {}


def test_stream_delta_carries_terminal_metadata() -> None:
    delta = StreamDelta(
        content="done",
        usage={"output_tokens": 3},
        finish_reason="stop",
        model="test-model",
        provider="test-provider",
    )
    assert delta.usage == {"output_tokens": 3}
    assert delta.finish_reason == "stop"
    assert delta.provider == "test-provider"


def test_structural_client_satisfies_runtime_protocol() -> None:
    class Client:
        model = "test-model"

        async def chat(
            self,
            messages: list[Message],
            **kwargs: object,
        ) -> LLMResponse:
            return LLMResponse(content="ok")

        async def stream_impl(self) -> AsyncIterator[StreamDelta]:
            yield StreamDelta(content="ok")

        def stream(
            self,
            messages: list[Message],
            **kwargs: object,
        ) -> AsyncIterator[StreamDelta]:
            return self.stream_impl()

    assert isinstance(Client(), LLMClient)
