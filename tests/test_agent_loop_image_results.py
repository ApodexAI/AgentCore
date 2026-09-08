"""End-to-end: a tool that attaches an image, driven through ``run_agent_loop``.

The unit tests in ``test_tool_content_images`` cover the pieces. These cover the
wiring -- that a tool's envelope actually survives execution, rendering, the
recovery-handle step and the history append, and that the model profile is what
decides whether the pixels go on the wire.
"""

from __future__ import annotations

import struct
import zlib
from typing import Any

import pytest

from agent_core.llm import LLMResponse
from agent_core.loop_types import LoopConfig, LoopPolicy
from agent_core.messages import for_wire
from agent_core.runtime.loop.agent_loop import run_agent_loop
from agent_core.runtime.loop.model_profile import HistoryPolicy, ModelProfile
from agent_core.tool_content import image_attachment, tool_content


def _png(width: int, height: int) -> bytes:
    def chunk(tag: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body))
            + tag
            + body
            + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\x00\x00\x00" * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


class SequenceLLM:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = responses
        self.calls: list[list[dict[str, Any]]] = []

    async def chat(self, messages, **_kwargs) -> LLMResponse:
        self.calls.append([dict(message) for message in messages])
        return self.responses.pop(0)

    def stream(self, messages, **_kwargs):
        raise AssertionError("streaming was not requested")


class ShotTool:
    """A view_image-shaped tool: a caption plus the bytes themselves."""

    name = "view_image"

    def __init__(self, count: int = 1) -> None:
        self.count = count

    async def ainvoke(self, args: dict[str, Any]) -> Any:
        path = str(args.get("path") or "/tmp/shot.png")
        return tool_content(
            f"Image {path} (64x32) is attached below.",
            images=[
                image_attachment(_png(64, 32), "image/png", label=path)
                for _ in range(self.count)
            ],
        )

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "read an image",
                "parameters": {"type": "object"},
            },
        }


def _call(index: int, path: str) -> dict[str, Any]:
    return {
        "id": f"tc{index}",
        "type": "function",
        "function": {"name": "view_image", "arguments": f'{{"path":"{path}"}}'},
    }


def _config() -> LoopConfig:
    return LoopConfig(
        max_turns=6, loop_policy=LoopPolicy(no_tool_behavior="stop"), max_llm_retries=1,
    )


async def _run(
    profile: ModelProfile,
    *,
    paths: list[str],
    policy: HistoryPolicy | None = None,
    images_per_call: int = 1,
):
    responses = [
        LLMResponse(content="", tool_calls=[_call(index, path)])
        for index, path in enumerate(paths)
    ]
    responses.append(LLMResponse(content="done"))
    llm = SequenceLLM(responses)
    result = await run_agent_loop(
        system_prompt="system",
        user_message="look",
        llm=llm,
        tools=[ShotTool(images_per_call)],
        config=_config(),
        model_profile=profile,
        history_policy=policy or HistoryPolicy(),
    )
    return llm, result


def _tool_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [m for m in messages if m.get("role") == "tool"]


@pytest.mark.asyncio
async def test_image_reaches_history_as_a_content_part() -> None:
    _, result = await _run(
        ModelProfile(model_id="apodex-1.1-mini", provider="apodex", supports_images=True),
        paths=["/tmp/a.png"],
    )
    (message,) = _tool_messages(result.messages)
    content = message["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert "/tmp/a.png" in content[0]["text"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_the_image_is_still_there_on_the_next_request() -> None:
    """A tool result is only useful if it survives into the following turn."""
    llm, _ = await _run(
        ModelProfile(model_id="apodex-1.1-mini", provider="apodex", supports_images=True),
        paths=["/tmp/a.png"],
    )
    sent = llm.calls[-1]
    images = [
        block
        for message in sent
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == "image_url"
    ]
    assert len(images) == 1
    # And nothing in-process rode along with it.
    assert all("image_meta" not in message for message in for_wire(sent))


@pytest.mark.asyncio
async def test_a_text_only_model_gets_the_warning_instead_of_the_bytes() -> None:
    _, result = await _run(
        ModelProfile(model_id="text-only", provider="openai"), paths=["/tmp/a.png"],
    )
    (message,) = _tool_messages(result.messages)
    content = message["content"]
    assert isinstance(content, str)
    assert "base64" not in content
    assert "have NOT" in content


@pytest.mark.asyncio
async def test_history_is_bounded_by_max_images_in_history() -> None:
    profile = ModelProfile(
        model_id="apodex-1.1-mini", provider="apodex", supports_images=True,
    )
    _, result = await _run(
        profile,
        paths=[f"/tmp/{index}.png" for index in range(4)],
        policy=HistoryPolicy(max_images_in_history=2),
    )
    messages = _tool_messages(result.messages)
    assert len(messages) == 4
    live = [
        message for message in messages if isinstance(message.get("content"), list)
    ]
    assert len(live) == 2
    # The two that went are named, not silently missing.
    gone = [m for m in messages if isinstance(m.get("content"), str)]
    assert all("no longer see it" in m["content"] for m in gone)
    assert "/tmp/0.png" in gone[0]["content"]


@pytest.mark.asyncio
async def test_one_result_with_several_images_is_not_self_evicting() -> None:
    """A multi-page read must not push out its own earlier pages first."""
    profile = ModelProfile(
        model_id="apodex-1.1-mini", provider="apodex", supports_images=True,
    )
    _, result = await _run(
        profile,
        paths=["/tmp/doc.png"],
        policy=HistoryPolicy(max_images_in_history=5),
        images_per_call=3,
    )
    (message,) = _tool_messages(result.messages)
    urls = [
        block
        for block in message["content"]
        if isinstance(block, dict) and block.get("type") == "image_url"
    ]
    assert len(urls) == 3
