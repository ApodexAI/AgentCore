"""A thinking-only block list must not swallow the model's visible answer."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from agent_core.llm import StreamDelta
from agent_core.messages import Message, user_msg
from agent_core.runtime.loop._streaming import _stream_llm_response


class _Client:
    """Emits text deltas plus a block list that omits the text block.

    Mirrors a gateway that renames/omits the ``content_block_start`` of type
    ``text`` while thinking blocks arrive normally: ``AnthropicClient.stream``
    only records a text block on that event, so the delta text reaches
    ``accumulated`` but never the block list.
    """

    def __init__(self, blocks: list[dict[str, Any]]) -> None:
        self._blocks = blocks

    async def stream(
        self,
        _messages: list[Message],
        **_kwargs: Any,
    ) -> AsyncIterator[StreamDelta]:
        yield StreamDelta(content="The answer ")
        yield StreamDelta(content="is 42.")
        yield StreamDelta(
            reasoning_blocks=self._blocks,
            finish_reason="end_turn",
            model="claude-x",
        )


async def _noop(*_args: Any, **_kwargs: Any) -> None:
    return None


def _run(blocks: list[dict[str, Any]]) -> Any:
    return asyncio.run(
        _stream_llm_response(
            _Client(blocks),
            [user_msg("q")],
            timeout=5.0,
            on_delta=_noop,
        ),
    )


THINKING = {"type": "thinking", "thinking": "hmm", "signature": "sig-abc"}


def test_thinking_only_blocks_keep_the_visible_answer() -> None:
    response = _run([THINKING])
    assert isinstance(response.content, list)
    # Thinking block preserved byte-exact for replay...
    assert response.content[0] == THINKING
    # ...and the answer is no longer lost.
    assert response.content[-1] == {"type": "text", "text": "The answer is 42."}


def test_blocks_that_already_carry_text_are_untouched() -> None:
    blocks = [THINKING, {"type": "text", "text": "The answer is 42."}]
    response = _run(blocks)
    assert response.content == blocks
    # No duplicated text block.
    assert sum(b.get("type") == "text" for b in response.content) == 1


def test_empty_text_block_does_not_count_as_carrying_text() -> None:
    response = _run([THINKING, {"type": "text", "text": "   "}])
    assert response.content[-1] == {"type": "text", "text": "The answer is 42."}


def test_no_blocks_still_yields_the_flat_string() -> None:
    response = _run([])
    assert response.content == "The answer is 42."
