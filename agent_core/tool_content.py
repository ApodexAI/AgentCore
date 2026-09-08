"""Structured tool output: text plus image attachments the model itself reads.

A tool normally returns a string and that string becomes the whole tool message.
Some tools have something to hand back that is not text — an image. The
alternative to this module is what the products did before it: call a separate
vision model, and put ITS prose transcript into the history. That is a lossy
round trip (the main model never sees the pixels, only another model's summary
of them, and cannot go back and look again when a later turn raises a new
question about the same picture) and it is not cheaper — a transcript runs
2-6K tokens where the image it describes is a few hundred.

The wire shape a tool returns is a plain JSON dict, never a dataclass, because
sandbox-native tools are executed in a child process and their return value
crosses a ``json.dumps(default=str)`` boundary on the way back
(``plugins/tool_runtime/server.py`` in MiroHarness). A dataclass survives that
trip as its repr. Build it with :func:`tool_content`::

    return tool_content(
        "Screenshot of the failing dialog:",
        images=[image_attachment(png_bytes, "image/png", label="/tmp/shot.png")],
    )

Nothing downstream is obliged to honour it. Whether the attachment reaches the
provider is decided in the loop, from ``ModelProfile.supports_images`` and the
wire protocol — a model with no vision gets the text and an explicit note that
an image was withheld. That note is not politeness. In the calibration run for
this feature the control case removed the image block and left everything else
identical: the model did not report a missing image, it invented a confident,
wrong reading of one. An image that silently vanishes from a tool result is
therefore a correctness bug, not a degraded capability, and every path here that
declines to attach says so in the text instead.

The measurements behind the constants in this module — the pixels-per-token
figure, the caps, the block shape the server actually accepts — are in
MiroHarness ``internal-docs/designs/2026-09-08-native-image-in-tool-result-calibration.md``.
Read it before changing any of them.
"""

from __future__ import annotations

import base64
import binascii
import logging
from typing import Any, cast

logger = logging.getLogger(__name__)

__all__ = [
    "IMAGE_MIME_TYPES",
    "MAX_IMAGES_PER_RESULT",
    "MAX_IMAGE_BYTES",
    "TOOL_CONTENT_MARKER",
    "image_attachment",
    "image_tokens",
    "message_image_meta",
    "message_image_tokens",
    "parse_tool_content",
    "redacted_for_trace",
    "sniff_image_size",
    "tool_content",
]

# Present and truthy on a dict that means "structured tool output". A marker key
# rather than duck-typing on ``{"text", "images"}``: tools are free to return
# ordinary dicts, and one that happens to carry those two keys must keep
# stringifying the way it always did.
TOOL_CONTENT_MARKER = "__tool_content__"

# What the OpenAI content-part schema accepts as an inline image. Enforced here
# because an unsupported type is a 400 from the server, and a 400 fails the whole
# turn rather than just the attachment.
IMAGE_MIME_TYPES = frozenset(
    {"image/png", "image/jpeg", "image/webp", "image/gif"}
)

# Per-image and per-result ceilings. Both are about the context window, not the
# HTTP body: a 4K screenshot measured 8.5K prompt tokens against apodex-1.1-mini,
# so a handful of them at full resolution is the whole budget. Producers are
# expected to downscale; these are the backstop for producers that did not.
MAX_IMAGE_BYTES = 6 * 1024 * 1024
MAX_IMAGES_PER_RESULT = 8

# Pixels per token, fitted against apodex-1.1-mini prompt_tokens over the same
# image at 640x360 / 1280x720 / 1920x1080 / 3840x2160 (227 / 908 / 2043 / 8170
# image tokens, r² > 0.999 on a straight px term). Providers differ, and a
# provider that tiles differently will be off by a constant factor — that is
# acceptable for a budget estimate, and catastrophically better than the zero
# this used to contribute, which let a history full of images look empty to the
# context guard.
_PIXELS_PER_TOKEN = 1024
# Charged when dimensions cannot be read. Deliberately not "0" and not the 4K
# figure: an unknown image is assumed to be roughly a 1080p screenshot.
_UNKNOWN_IMAGE_TOKENS = 2400


def tool_content(text: str, *, images: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Build the structured return value for a tool with attachments.

    ``text`` is what the tool would have returned on its own, and stays the
    tool's result string everywhere in the loop — observers, spill/recovery,
    repeat detection and the trajectory all keep seeing a plain string. The
    images ride alongside it.
    """
    return {
        TOOL_CONTENT_MARKER: 1,
        "text": text,
        "images": list(images or []),
    }


def image_attachment(
    data: bytes | str,
    mime_type: str,
    *,
    label: str = "",
    width: int = 0,
    height: int = 0,
) -> dict[str, Any]:
    """One image attachment. ``data`` may be raw bytes or an existing base64 str.

    ``label`` is what the model is told the image IS (usually the path it asked
    for). It is carried separately from the text so a placeholder can name the
    image after the bytes are gone.
    """
    b64 = base64.b64encode(data).decode("ascii") if isinstance(data, bytes) else data
    attachment: dict[str, Any] = {"mime_type": mime_type, "data": b64}
    if label:
        attachment["label"] = label
    if width > 0 and height > 0:
        attachment["width"] = int(width)
        attachment["height"] = int(height)
    return attachment


def parse_tool_content(raw: Any) -> tuple[str, list[dict[str, Any]]] | None:
    """Split a tool return value into ``(text, images)``, or ``None``.

    ``None`` means "this is not structured output" and the caller should keep
    its existing ``str(raw)`` behaviour. A malformed attachment inside a
    well-formed envelope is dropped and reported in the text, never silently:
    see the module docstring on why a vanishing image is worse than no image.
    """
    if not isinstance(raw, dict):
        return None
    envelope = cast("dict[str, Any]", raw)
    if not envelope.get(TOOL_CONTENT_MARKER):
        return None
    text = envelope.get("text")
    text = text if isinstance(text, str) else ("" if text is None else str(text))

    raw_images = envelope.get("images")
    candidates: list[Any] = (
        list(cast("list[Any]", raw_images)) if isinstance(raw_images, list) else []
    )
    images: list[dict[str, Any]] = []
    rejected: list[str] = []
    for index, candidate in enumerate(candidates):
        if len(images) >= MAX_IMAGES_PER_RESULT:
            rejected.append(
                f"image {index + 1} and the ones after it (more than "
                f"{MAX_IMAGES_PER_RESULT} images in one result)"
            )
            break
        accepted, why = _validated_image(candidate)
        if accepted is None:
            rejected.append(f"image {index + 1} ({why})")
            continue
        images.append(accepted)

    if rejected:
        note = "not attached: " + "; ".join(rejected)
        logger.warning("tool content dropped %s", note)
        text = f"{text}\n\n[{note}]" if text else f"[{note}]"
    return text, images


def _validated_image(candidate: Any) -> tuple[dict[str, Any] | None, str]:
    if not isinstance(candidate, dict):
        return None, "not an object"
    image = cast("dict[str, Any]", candidate)
    mime = str(image.get("mime_type") or "")
    if mime not in IMAGE_MIME_TYPES:
        return None, f"unsupported type {mime or 'unset'!r}"
    data = image.get("data")
    if not isinstance(data, str) or not data:
        return None, "no base64 payload"
    # Validated here rather than at the provider: a bad payload is an HTTP 400
    # that takes the whole turn down, and the turn's other tool results with it.
    try:
        decoded_size = len(base64.b64decode(data, validate=True))
    except (binascii.Error, ValueError):
        return None, "payload is not valid base64"
    if decoded_size == 0:
        return None, "payload is empty"
    if decoded_size > MAX_IMAGE_BYTES:
        return None, (
            f"{decoded_size // 1024} KB exceeds the "
            f"{MAX_IMAGE_BYTES // (1024 * 1024)} MB per-image cap"
        )

    accepted: dict[str, Any] = {"mime_type": mime, "data": data}
    label = image.get("label")
    if isinstance(label, str) and label:
        accepted["label"] = label
    width, height = _declared_size(image)
    if width <= 0 or height <= 0:
        width, height = sniff_image_size(base64.b64decode(data, validate=True))
    if width > 0 and height > 0:
        accepted["width"] = width
        accepted["height"] = height
    return accepted, ""


def _declared_size(image: dict[str, Any]) -> tuple[int, int]:
    try:
        return int(image.get("width") or 0), int(image.get("height") or 0)
    except (TypeError, ValueError):
        return 0, 0


def sniff_image_size(data: bytes) -> tuple[int, int]:
    """Pixel dimensions from an image header, or ``(0, 0)``.

    Header parsing rather than a decode: the only consumer is the token
    estimate, this runs on every attachment, and AgentCore has no imaging
    dependency to decode with.
    """
    try:
        if data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
            return (
                int.from_bytes(data[16:20], "big"),
                int.from_bytes(data[20:24], "big"),
            )
        if data[:3] == b"GIF":
            return (
                int.from_bytes(data[6:8], "little"),
                int.from_bytes(data[8:10], "little"),
            )
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return _webp_size(data)
        if data[:2] == b"\xff\xd8":
            return _jpeg_size(data)
    except (IndexError, ValueError):
        return 0, 0
    return 0, 0


def _webp_size(data: bytes) -> tuple[int, int]:
    chunk = data[12:16]
    if chunk == b"VP8X":
        return (
            int.from_bytes(data[24:27], "little") + 1,
            int.from_bytes(data[27:30], "little") + 1,
        )
    if chunk == b"VP8 ":
        return (
            int.from_bytes(data[26:28], "little") & 0x3FFF,
            int.from_bytes(data[28:30], "little") & 0x3FFF,
        )
    if chunk == b"VP8L":
        bits = int.from_bytes(data[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    return 0, 0


def _jpeg_size(data: bytes) -> tuple[int, int]:
    # Walk the marker chain to the frame header. SOF0/1/2/3/5/6/7/9-11/13-15 all
    # carry the dimensions at the same offset; DHT/DAC/RST/SOS do not and are
    # skipped by length like any other segment.
    offset = 2
    end = len(data)
    while offset + 9 < end:
        if data[offset] != 0xFF:
            offset += 1
            continue
        marker = data[offset + 1]
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            offset += 2
            continue
        if marker == 0xDA:  # start of scan — no frame header past here
            return 0, 0
        segment_length = int.from_bytes(data[offset + 2:offset + 4], "big")
        if segment_length < 2:
            return 0, 0
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            return (
                int.from_bytes(data[offset + 7:offset + 9], "big"),
                int.from_bytes(data[offset + 5:offset + 7], "big"),
            )
        offset += 2 + segment_length
    return 0, 0


def image_tokens(image: dict[str, Any]) -> int:
    """Estimated prompt tokens an attachment costs. Never zero."""
    width, height = _declared_size(image)
    if width <= 0 or height <= 0:
        return _UNKNOWN_IMAGE_TOKENS
    return max(1, (width * height) // _PIXELS_PER_TOKEN)


def message_image_meta(message: Any) -> list[dict[str, Any]]:
    """The ``image_meta`` list on a message, positionally aligned with images."""
    if not isinstance(message, dict):
        return []
    raw = cast("dict[str, Any]", message).get("image_meta")
    if not isinstance(raw, list):
        return []
    return [
        cast("dict[str, Any]", entry)
        for entry in cast("list[Any]", raw)
        if isinstance(entry, dict)
    ]


def message_image_tokens(message: Any) -> int:
    """Estimated prompt tokens the inline images on *message* cost.

    Counted from the ``image_url`` blocks actually present, priced from
    ``image_meta``. Driven by the content rather than by the metadata because a
    compactor is free to rewrite ``content`` to a plain string -- which drops
    the images -- without knowing that a bookkeeping key describes them. Trusting
    the metadata there would keep charging for pixels no longer in the request.

    The prices are read off ``image_meta`` rather than measured: the bytes are a
    data URI by this point, and base64-decoding the whole history on every
    estimate would make the context guard cost more than the turn it guards. An
    image with no usable price still charges something -- an unknown image is
    not a free one.

    Lives here rather than beside the rest of the message-image handling in
    ``runtime.loop.image_attach`` for one reason: ``tokens`` needs it, and this
    module imports nothing from ``agent_core``, so there is no import cycle to
    get wrong later.
    """
    if not isinstance(message, dict):
        return 0
    content = cast("dict[str, Any]", message).get("content")
    if not isinstance(content, list):
        return 0
    blocks = sum(
        1
        for block in cast("list[Any]", content)
        if isinstance(block, dict) and cast("dict[str, Any]", block).get("type") == "image_url"
    )
    if not blocks:
        return 0
    meta = message_image_meta(message)
    total = 0
    for index in range(blocks):
        entry = meta[index] if index < len(meta) else {}
        try:
            priced = max(0, int(entry.get("tokens") or 0))
        except (TypeError, ValueError):
            priced = 0
        total += priced or _UNKNOWN_IMAGE_TOKENS
    return total


def redacted_for_trace(message: Any) -> Any:
    """A copy of *message* with inline image payloads replaced by a marker.

    For anything that writes a message somewhere other than the provider: a
    trajectory file, a log line, an event record. The base64 of a single 1080p
    screenshot is ~137 KB, and a trace that copies messages verbatim writes that
    again for every turn the image survives in history -- a few screenshots turn
    a readable trajectory into tens of megabytes of unreadable one.

    The block KEEPS its ``image_url`` type and gains a stated size, so a reader
    can still see that an image was in the request and how big it was. That
    matters for the same reason the loop narrates evictions: a trace that shows
    no image where the model saw one misrepresents what the model was answering.

    Returns the message unchanged (not a copy) when it carries no inline image,
    which is nearly every message.
    """
    if not isinstance(message, dict):
        return message
    typed = cast("dict[str, Any]", message)
    content = typed.get("content")
    if not isinstance(content, list):
        return typed
    blocks = cast("list[Any]", content)
    if not any(
        isinstance(block, dict)
        and cast("dict[str, Any]", block).get("type") == "image_url"
        for block in blocks
    ):
        return typed

    redacted: list[Any] = []
    for block in blocks:
        entry = cast("dict[str, Any]", block) if isinstance(block, dict) else None
        if entry is None or entry.get("type") != "image_url":
            redacted.append(block)
            continue
        url = entry.get("image_url")
        raw = str(cast("dict[str, Any]", url).get("url") or "") if isinstance(url, dict) else ""
        redacted.append({
            "type": "image_url",
            "image_url": {"url": _elided_data_uri(raw)},
        })
    return {**typed, "content": redacted}


def _elided_data_uri(url: str) -> str:
    """``data:image/png;base64,<...>`` → a same-shaped string stating the size."""
    if not url.startswith("data:") or ";base64," not in url:
        return url
    prefix, payload = url.split(";base64,", 1)
    approx_kb = max(1, (len(payload) * 3 // 4) // 1024)
    return f"{prefix};base64,[{approx_kb} KB of image data elided from trace]"
