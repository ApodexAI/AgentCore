"""Put a tool's image attachments into the message the provider sees.

Two decisions live here and nowhere else: whether an attachment is allowed onto
the wire at all, and how many stay in history as the run goes on. Both resolve
the same way when the answer is no -- the image is replaced by a sentence saying
an image was there and is not any more.

That is the point of the module rather than an incidental nicety. The
calibration run behind this feature (MiroHarness
``internal-docs/designs/2026-09-08-native-image-in-tool-result-calibration.md``,
case E) took a working request and deleted only the image block. The model was instructed to
answer ``NO_IMAGE`` if it could not see an image. It instead reported a
four-digit code and three shapes, all confidently wrong. A model handed a tool
result that reads like an image was delivered will describe the image it
expects; only text that contradicts that expectation stops it.

Per-image bookkeeping (label, estimated tokens) rides on the message under
``image_meta``, positionally aligned with the ``image_url`` blocks in
``content``. It is deliberately message-level and not inside the blocks: a
content part with an unrecognised key is rejected by the served endpoint's
pydantic union, while a message-level key outside ``WIRE_MESSAGE_KEYS`` is
dropped by :func:`agent_core.messages.for_wire` before the request is built.
Keeping the token estimate here is also what stops the context guard from
having to base64-decode the whole history on every turn to find out how big it
is.
"""

from __future__ import annotations

import logging
from typing import Any, cast

from agent_core.messages import Message
from agent_core.runtime.loop.model_profile import ModelProfile
from agent_core.tool_content import (
    image_tokens,
    message_image_meta,
    message_image_tokens,
)

logger = logging.getLogger(__name__)

__all__ = [
    "attach_images",
    "evict_old_images",
    "image_blocks_in",
    "message_image_tokens",
]

# Wire protocols whose inline-image block shape is the one built below.
# Anthropic Messages and the OpenAI Responses API both accept images and both
# spell the block differently (``{"type": "image", "source": {...}}`` /
# ``{"type": "input_image", ...}``). This is not a shape a server tolerates and
# ignores when wrong: the calibration run sent the Anthropic spelling to the
# OpenAI-compatible endpoint and got HTTP 400 with a pydantic union error, which
# fails the entire turn including the other tool results in it. So the protocol
# is checked rather than assumed, and an unhandled one withholds with a note.
_INLINE_IMAGE_PROTOCOLS = frozenset({"chat_completions"})


def _as_block(block: Any) -> dict[str, Any] | None:
    """A content entry as a keyed block, or ``None`` if it is not one."""
    return cast("dict[str, Any]", block) if isinstance(block, dict) else None


def _is_image(block: Any) -> bool:
    entry = _as_block(block)
    return entry is not None and entry.get("type") == "image_url"


def attach_images(
    message: Message,
    images: list[dict[str, Any]],
    *,
    profile: ModelProfile,
) -> None:
    """Add *images* to a tool *message*, in place.

    When the model cannot take them, the message keeps plain-string content and
    gains an explicit note. Callers never branch on capability: they always call
    this, and the message is correct either way.
    """
    if not images:
        return

    why = ""
    if not profile.supports_images:
        why = f"the model in use ({profile.model_id}) cannot accept images"
    elif profile.protocol not in _INLINE_IMAGE_PROTOCOLS:
        why = (
            f"inline images are not implemented for the {profile.protocol} "
            "wire protocol"
        )
    if why:
        logger.info("withholding %d image(s) from tool message: %s", len(images), why)
        text = _text_content(message)
        note = _withheld_note(images, why)
        message["content"] = f"{text}\n\n{note}" if text else note
        return

    text = _text_content(message)
    blocks: list[dict[str, Any]] = []
    if text:
        blocks.append({"type": "text", "text": text})
    meta: list[dict[str, Any]] = []
    for image in images:
        blocks.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{image['mime_type']};base64,{image['data']}",
            },
        })
        meta.append({
            "label": str(image.get("label") or ""),
            "tokens": image_tokens(image),
        })
    message["content"] = blocks
    cast("dict[str, Any]", message)["image_meta"] = meta


def evict_old_images(messages: list[Message], max_images: int) -> int:
    """Keep only the newest *max_images* inline images; return how many went.

    Walks newest-first, so what survives is what the model is most likely to
    still be working from. An evicted image leaves a sentence behind, and a
    message left with no images goes back to plain-string content -- there is no
    reason to keep a block list, and a string is what every checkpoint, replay
    and text-flattening path handles most cheaply.
    """
    if max_images < 0:
        return 0

    kept = 0
    evicted = 0
    for message in messages[::-1]:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        meta = message_image_meta(message)
        blocks = cast("list[Any]", content)
        image_positions = [
            index for index, block in enumerate(blocks) if _is_image(block)
        ]
        if not image_positions:
            continue
        # Newest-first WITHIN the message too, not just across messages. A
        # single tool result can return several images (a PDF rendered page by
        # page), and walking its blocks forward while walking the history
        # backward keeps the wrong end of that result.
        drop_at: set[int] = set()
        for index in reversed(image_positions):
            if kept < max_images:
                kept += 1
                continue
            drop_at.add(index)
            evicted += 1
        if not drop_at:
            continue
        at_image = set(image_positions)
        rebuilt: list[Any] = []
        surviving: list[dict[str, Any]] = []
        seen = 0
        for index, block in enumerate(blocks):
            if index not in at_image:
                rebuilt.append(block)
                continue
            entry = meta[seen] if seen < len(meta) else {}
            seen += 1
            if index in drop_at:
                rebuilt.append({"type": "text", "text": _evicted_note(entry)})
            else:
                rebuilt.append(block)
                surviving.append(entry)
        if surviving:
            message["content"] = rebuilt
            cast("dict[str, Any]", message)["image_meta"] = surviving
        else:
            message["content"] = "\n".join(
                str(entry.get("text") or "")
                for entry in (_as_block(block) for block in rebuilt)
                if entry is not None and entry.get("text")
            )
            cast("dict[str, Any]", message).pop("image_meta", None)
    if evicted:
        logger.info(
            "evicted %d image(s) from history, keeping the newest %d",
            evicted, max_images,
        )
    return evicted


def image_blocks_in(message: Message) -> int:
    """How many inline images this message currently carries."""
    content = message.get("content")
    if not isinstance(content, list):
        return 0
    return sum(1 for block in cast("list[Any]", content) if _is_image(block))


def _withheld_note(images: list[dict[str, Any]], why: str) -> str:
    named = ", ".join(
        label for label in (str(i.get("label") or "") for i in images) if label
    )
    subject = "1 image" if len(images) == 1 else f"{len(images)} images"
    where = f" ({named})" if named else ""
    return (
        f"[{subject}{where} could not be shown to you: {why}. You have NOT "
        "seen this image. Do not describe, transcribe, or draw any conclusion "
        "from its contents.]"
    )


def _evicted_note(entry: dict[str, Any]) -> str:
    label = str(entry.get("label") or "")
    where = f" {label}" if label else ""
    return (
        f"[An image{where} was here and has been dropped from context to make "
        "room. You can no longer see it. Read it again if you still need it, "
        "and do not rely on remembering what it showed.]"
    )


def _text_content(message: Message) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(entry.get("text") or "")
            for entry in (_as_block(block) for block in cast("list[Any]", content))
            if entry is not None and entry.get("type") == "text"
        ]
        return "\n".join(part for part in parts if part)
    return "" if content is None else str(content)
