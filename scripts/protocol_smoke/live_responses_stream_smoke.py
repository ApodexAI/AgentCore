"""Live smoke test: a real Responses-API turn, streamed vs non-streamed.

Drives ``OpenAIResponsesClient`` through the real agent loop against an actual
llm-hub endpoint, once with ``stream_llm_tokens=False`` and once with
``True``, over a question that forces one tool call. Prints both final
answers and diffs the resulting message histories — the thing
``test_responses_stream_replay.py`` checks against a fake SDK, here checked
against the real wire format.

Usage:
    OPENAI_API_KEY=... uv run python scripts/protocol_smoke/live_responses_stream_smoke.py [model]

Model defaults to gpt-4o (the only llm-hub model found that both supports
native ``/v1/responses`` and isn't blocked upstream as of 2026-10-14 — every
gpt-5.x/6.x reasoning model on that account 400s or returns a policy-violation
failure regardless of this fix; see temp/2026-10-14_responses-stream-live-smoke.md).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

from agent_core.loop_types import LoopConfig, LoopPolicy
from agent_core.providers.openai_responses import OpenAIResponsesClient
from agent_core.runtime.loop.agent_loop import run_agent_loop
from agent_core.runtime.loop.model_profile import ModelProfile


class Weather:
    name = "get_weather"

    async def ainvoke(self, args: dict) -> str:
        city = args.get("city", "")
        return f"{city}: 14C, light rain"

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Get the current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }


async def _run(model: str, *, streaming: bool) -> tuple[str, list[dict]]:
    base_url = (os.environ.get("LLM_HUB_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
               or "https://llm-hub.apodex.app/v1").removesuffix("/chat/completions")
    api_key = os.environ.get("LLM_HUB_API_KEY") or os.environ.get("OPENAI_API_KEY")
    client = OpenAIResponsesClient(model, api_key=api_key, base_url=base_url, timeout=90)
    deltas: list[str] = []

    class PrintObserver:
        wants_llm_delta = streaming

        async def on_llm_delta(self, ctx) -> None:
            text = getattr(ctx, "delta", "") or ""
            if text:
                deltas.append(text)

    result = await run_agent_loop(
        system_prompt="You are a terse assistant. Use the weather tool when asked about weather.",
        user_message="What's the weather in Berlin right now? Use the tool, then answer in one sentence.",
        llm=client, tools=[Weather()],
        config=LoopConfig(
            max_turns=4, max_llm_retries=1, stream_llm_tokens=streaming,
            loop_policy=LoopPolicy(no_tool_behavior="stop"),
        ),
        model_profile=ModelProfile(
            model_id=model, provider="openai", protocol="responses",
            thinking_format="content_block",
        ),
        observers=[PrintObserver()],
    )
    print(f"  [{'stream' if streaming else 'batch '}] deltas received: {len(deltas)}, "
          f"joined len={sum(len(d) for d in deltas)}")
    return result.final_content, result.messages


async def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt-4o"
    print(f"model={model}")

    print("-- non-streaming --")
    batch_answer, batch_messages = await _run(model, streaming=False)
    print("batch answer:", batch_answer)

    print("-- streaming --")
    stream_answer, stream_messages = await _run(model, streaming=True)
    print("stream answer:", stream_answer)

    print()
    print("tool calls seen (batch):   ",
          [m.get("tool_calls") for m in batch_messages if m.get("role") == "assistant" and m.get("tool_calls")])
    print("tool calls seen (stream):  ",
          [m.get("tool_calls") for m in stream_messages if m.get("role") == "assistant" and m.get("tool_calls")])
    print()
    same_shape = (
        len(batch_messages) == len(stream_messages)
        and [m.get("role") for m in batch_messages] == [m.get("role") for m in stream_messages]
    )
    print("same role sequence:", same_shape)
    print()
    print("batch messages:")
    print(json.dumps(batch_messages, indent=2)[:3000])
    print()
    print("stream messages:")
    print(json.dumps(stream_messages, indent=2)[:3000])


if __name__ == "__main__":
    asyncio.run(main())
