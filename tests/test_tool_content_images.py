"""Images returned by a tool: envelope parsing, attachment, eviction, cost.

The behavioural anchor for the whole feature is
``test_withheld_image_says_so_in_the_text`` and its eviction twin. Everything
else here is plumbing; those two encode why the plumbing is shaped this way.
See ``agent_core/runtime/loop/image_attach.py`` for the calibration run.
"""

from __future__ import annotations

import base64
import struct
import zlib
from typing import Any

import pytest

from agent_core.messages import for_wire, text_of, tool_msg
from agent_core.runtime.loop.image_attach import (
    attach_images,
    evict_old_images,
    image_blocks_in,
)
from agent_core.runtime.loop.model_profile import ModelProfile
from agent_core.tokens import estimate_message_tokens
from agent_core.tool_content import (
    MAX_IMAGE_BYTES,
    MAX_IMAGES_PER_RESULT,
    image_attachment,
    image_tokens,
    parse_tool_content,
    sniff_image_size,
    tool_content,
)


def _png(width: int, height: int) -> bytes:
    """A real, minimal PNG of the requested size (no imaging dependency)."""
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


def _vision_profile() -> ModelProfile:
    return ModelProfile(
        model_id="apodex-1.1-mini", provider="apodex", supports_images=True,
    )


def _blind_profile() -> ModelProfile:
    return ModelProfile(model_id="text-only", provider="openai")


def _attached(text: str, images: list[dict[str, Any]], profile: ModelProfile):
    message = tool_msg(text, "call_1")
    attach_images(message, images, profile=profile)
    return message


# ── envelope ──────────────────────────────────────────────────────────────


def test_plain_returns_are_not_envelopes() -> None:
    assert parse_tool_content("just text") is None
    assert parse_tool_content(None) is None
    # A tool returning an ordinary dict that happens to have these keys keeps
    # its existing stringify behaviour -- the marker is what opts in.
    assert parse_tool_content({"text": "hi", "images": []}) is None


def test_envelope_splits_text_from_images() -> None:
    envelope = tool_content(
        "screenshot:",
        images=[image_attachment(_png(64, 32), "image/png", label="/tmp/a.png")],
    )
    parsed = parse_tool_content(envelope)
    assert parsed is not None
    text, images = parsed
    assert text == "screenshot:"
    assert len(images) == 1
    assert images[0]["label"] == "/tmp/a.png"
    assert (images[0]["width"], images[0]["height"]) == (64, 32)


@pytest.mark.parametrize(
    ("image", "reason"),
    [
        ({"mime_type": "image/tiff", "data": "aGk="}, "unsupported type"),
        ({"mime_type": "image/png", "data": "not base64!!"}, "not valid base64"),
        ({"mime_type": "image/png", "data": ""}, "no base64 payload"),
        ({"mime_type": "image/png"}, "no base64 payload"),
        ("not-a-dict", "not an object"),
    ],
)
def test_a_rejected_image_is_reported_not_dropped(image: Any, reason: str) -> None:
    parsed = parse_tool_content(tool_content("body", images=[image]))
    assert parsed is not None
    text, images = parsed
    assert images == []
    assert "not attached" in text
    assert reason in text


def test_oversized_image_is_rejected_with_its_size() -> None:
    payload = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * MAX_IMAGE_BYTES)
    parsed = parse_tool_content(
        tool_content("body", images=[{"mime_type": "image/png", "data": payload.decode()}])
    )
    assert parsed is not None
    text, images = parsed
    assert images == []
    assert "per-image cap" in text


def test_image_count_is_capped_and_the_overflow_is_named() -> None:
    one = image_attachment(_png(8, 8), "image/png")
    parsed = parse_tool_content(
        tool_content("body", images=[dict(one) for _ in range(MAX_IMAGES_PER_RESULT + 3)])
    )
    assert parsed is not None
    text, images = parsed
    assert len(images) == MAX_IMAGES_PER_RESULT
    assert "more than" in text


# ── header sniffing and cost ──────────────────────────────────────────────


def test_png_dimensions_come_off_the_header() -> None:
    assert sniff_image_size(_png(1920, 4)) == (1920, 4)


def test_unreadable_header_still_costs_something() -> None:
    assert sniff_image_size(b"not an image") == (0, 0)
    # An image whose size cannot be read must not be free; a zero here is how a
    # history of images measures as empty to the context guard.
    assert image_tokens({}) > 1000


def test_token_estimate_tracks_pixels() -> None:
    small = image_tokens({"width": 640, "height": 360})
    large = image_tokens({"width": 1920, "height": 1080})
    assert small == pytest.approx(225, abs=30)
    assert large == pytest.approx(2043, abs=200)


def test_estimate_counts_attached_images() -> None:
    caption = tool_msg("screenshot:", "call_1")
    text_only = estimate_message_tokens(caption)
    withimage = _attached(
        "screenshot:",
        [image_attachment(_png(1920, 1080), "image/png")],
        _vision_profile(),
    )
    assert estimate_message_tokens(withimage) - text_only > 1500


# ── attachment ────────────────────────────────────────────────────────────


def test_attached_image_becomes_an_openai_content_part() -> None:
    message = _attached(
        "screenshot:",
        [image_attachment(_png(64, 32), "image/png", label="/tmp/a.png")],
        _vision_profile(),
    )
    content = message["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "screenshot:"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    # The block itself carries nothing extra: the served endpoint's pydantic
    # union rejects an unknown key inside a content part outright.
    assert set(content[1]) == {"type", "image_url"}
    assert set(content[1]["image_url"]) == {"url"}


def test_bookkeeping_never_reaches_the_wire() -> None:
    message = _attached(
        "screenshot:", [image_attachment(_png(64, 32), "image/png", label="/tmp/a.png")],
        _vision_profile(),
    )
    assert message["image_meta"][0]["label"] == "/tmp/a.png"
    assert "image_meta" not in for_wire([message])[0]


def test_withheld_image_says_so_in_the_text() -> None:
    """The anchor case: a model that cannot see the image must be told.

    Removing the image block and changing nothing else is what made the
    calibration model report a confident, entirely invented reading of the
    picture. Silence here is a correctness bug, not a missing feature.
    """
    message = _attached(
        "screenshot:",
        [image_attachment(_png(64, 32), "image/png", label="/tmp/a.png")],
        _blind_profile(),
    )
    content = message["content"]
    assert isinstance(content, str)
    assert "screenshot:" in content
    assert "have NOT" in content
    assert "/tmp/a.png" in content
    assert "text-only" in content


def test_unsupported_protocol_withholds_rather_than_guessing_a_block_shape() -> None:
    profile = ModelProfile(
        model_id="claude", provider="anthropic",
        supports_images=True, protocol="anthropic",
    )
    message = _attached("shot:", [image_attachment(_png(8, 8), "image/png")], profile)
    assert isinstance(message["content"], str)
    assert "anthropic" in message["content"]


def test_no_images_leaves_the_message_exactly_as_it_was() -> None:
    message = tool_msg("plain result", "call_1")
    attach_images(message, [], profile=_vision_profile())
    assert message == {"content": "plain result", "role": "tool", "tool_call_id": "call_1"}


def test_flattening_to_text_leaves_a_mark() -> None:
    message = _attached(
        "shot:", [image_attachment(_png(8, 8), "image/png")], _vision_profile(),
    )
    # Anthropic translation, compaction summaries and the trajectory all read
    # content through text_of; an image that flattens to nothing produces a
    # transcript claiming a picture was never there.
    assert "[image" in text_of(message["content"])


# ── eviction ──────────────────────────────────────────────────────────────


def _history(count: int) -> list[Any]:
    history: list[Any] = []
    for index in range(count):
        history.append(
            _attached(
                f"shot {index}:",
                [image_attachment(_png(16, 16), "image/png", label=f"/tmp/{index}.png")],
                _vision_profile(),
            )
        )
    return history


def test_eviction_keeps_the_newest_images() -> None:
    history = _history(5)
    assert evict_old_images(history, 2) == 3
    assert [image_blocks_in(m) for m in history] == [0, 0, 0, 1, 1]


def test_evicted_image_leaves_a_sentence_naming_it() -> None:
    history = _history(2)
    evict_old_images(history, 1)
    oldest = history[0]["content"]
    assert isinstance(oldest, str)
    assert "shot 0:" in oldest
    assert "/tmp/0.png" in oldest
    assert "no longer see it" in oldest


def test_eviction_drops_the_bookkeeping_it_no_longer_describes() -> None:
    history = _history(3)
    evict_old_images(history, 1)
    assert "image_meta" not in history[0]
    assert len(history[-1]["image_meta"]) == 1


def test_partial_eviction_within_one_message_keeps_the_rest_aligned() -> None:
    message = _attached(
        "pages:",
        [
            image_attachment(_png(16, 16), "image/png", label="/tmp/p1.png"),
            image_attachment(_png(16, 16), "image/png", label="/tmp/p2.png"),
        ],
        _vision_profile(),
    )
    assert evict_old_images([message], 1) == 1
    assert image_blocks_in(message) == 1
    # The surviving entry must be the one still present -- p2, the newer.
    assert [entry["label"] for entry in message["image_meta"]] == ["/tmp/p2.png"]
    assert "/tmp/p1.png" in text_of(message["content"])


def test_eviction_is_idempotent_and_ignores_text_messages() -> None:
    history = _history(2)
    history.insert(0, tool_msg("no images here", "call_x"))
    assert evict_old_images(history, 1) == 1
    assert evict_old_images(history, 1) == 0
    assert history[0]["content"] == "no images here"


def test_negative_budget_disables_eviction() -> None:
    history = _history(3)
    assert evict_old_images(history, -1) == 0
    assert sum(image_blocks_in(m) for m in history) == 3


def test_zero_budget_evicts_everything_but_says_so_each_time() -> None:
    history = _history(2)
    assert evict_old_images(history, 0) == 2
    assert all(isinstance(m["content"], str) for m in history)
    assert all("no longer see it" in m["content"] for m in history)


# ── interaction with compaction ───────────────────────────────────────────


def test_stale_bookkeeping_stops_charging_once_the_images_are_gone() -> None:
    """A compactor may rewrite content to a string without knowing about images.

    ``compress_tool_results`` flattens tool content through ``text_of`` and
    assigns a plain string. The images are gone from the request at that point,
    so the estimate must stop charging for them even though ``image_meta`` is
    still on the message.
    """
    message = _attached(
        "shot:", [image_attachment(_png(1920, 1080), "image/png")], _vision_profile(),
    )
    assert estimate_message_tokens(message) > 1500
    message["content"] = "…condensed by a compactor…"
    assert "image_meta" in message
    assert estimate_message_tokens(message) < 100


def test_images_without_bookkeeping_are_still_charged() -> None:
    """History restored from a checkpoint written before ``image_meta`` existed."""
    message = _attached(
        "shot:", [image_attachment(_png(1920, 1080), "image/png")], _vision_profile(),
    )
    del message["image_meta"]
    assert estimate_message_tokens(message) > 1000


# ── trace redaction ───────────────────────────────────────────────────────


def test_trace_redaction_keeps_the_shape_and_states_the_size() -> None:
    from agent_core.tool_content import redacted_for_trace

    message = _attached(
        "shot:", [image_attachment(_png(1920, 1080), "image/png")], _vision_profile(),
    )
    traced = redacted_for_trace(message)
    url = traced["content"][1]["image_url"]["url"]
    assert "base64" in url
    assert "elided from trace" in url
    assert "KB" in url
    # Still visibly an image, so a trace does not misrepresent what the model saw.
    assert traced["content"][1]["type"] == "image_url"
    assert traced["content"][0] == {"type": "text", "text": "shot:"}
    # And the real message is untouched.
    assert message["content"][1]["image_url"]["url"].startswith("data:image/png;base64,i")


def test_trace_redaction_is_a_no_op_for_ordinary_messages() -> None:
    from agent_core.tool_content import redacted_for_trace

    plain = tool_msg("no images", "call_1")
    assert redacted_for_trace(plain) is plain
    assert redacted_for_trace("not a message") == "not a message"


def test_the_trajectory_observer_does_not_write_base64() -> None:
    import json
    import tempfile
    from pathlib import Path

    from agent_core.components.observers.trajectory import TrajectoryFileObserver

    message = _attached(
        "shot:", [image_attachment(_png(1920, 1080), "image/png")], _vision_profile(),
    )
    with tempfile.TemporaryDirectory() as tmp:
        observer = TrajectoryFileObserver(Path(tmp))
        rendered = observer._message_to_dict(message)
    assert rendered is not None
    body = json.dumps(rendered)
    assert "elided from trace" in body
    # A 1080p PNG is ~137 KB of base64; the trace entry must not carry it.
    assert len(body) < 2000
