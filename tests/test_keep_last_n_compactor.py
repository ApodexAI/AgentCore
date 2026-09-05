"""Tier 1 elided-result mini card: what survives a blanked tool body.

The card carries the call's arguments and the source URLs from the body being
discarded — the two things a later turn needs in order not to re-issue a query
it already ran. These tests pin its budget, its idempotency against the existing
placeholder check, and the one case where the card is skipped in favour of the
body it would have replaced.
"""

from __future__ import annotations

from agent_core.messages import system_msg, tool_msg
from agent_core.runtime.loop.compact import (
    _LEGACY_OMITTED_TOOL_RESULT_PLACEHOLDERS,
    _MINI_CARD_ARGS_MAX_CHARS,
    _MINI_CARD_BODY_MAX_CHARS,
    _MINI_CARD_MAX_URLS,
    OMITTED_TOOL_RESULT_PLACEHOLDER,
    KeepLastNToolResultsCompactor,
    _args_preview,
    default_recovery_footer,
)


def _ai(tid: str, name: str, arguments: str = "{}") -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": tid, "function": {"name": name, "arguments": arguments}}],
    }


def _one_call(name: str, arguments: str, body: str, **tool_fields: object) -> list[dict]:
    """A minimal history whose single tool result is old enough to be blanked."""
    msg = tool_msg(body, "c1")
    msg.update(tool_fields)  # type: ignore[arg-type]
    return [system_msg("S"), _ai("c1", name, arguments), msg]


def _blanked(messages: list[dict], **kwargs: object) -> str:
    compactor = KeepLastNToolResultsCompactor(keep_tool_result=0, **kwargs)  # type: ignore[arg-type]
    out = compactor.compact(messages, 0)
    bodies = [m["content"] for m in out if m.get("role") == "tool"]
    assert len(bodies) == 1
    return bodies[0]


def _card_of(content: str) -> str:
    """The card lines only: everything after the placeholder's first line."""
    assert content.startswith(OMITTED_TOOL_RESULT_PLACEHOLDER)
    return content[len(OMITTED_TOOL_RESULT_PLACEHOLDER) :].lstrip("\n")


# --- what the card carries -------------------------------------------------


def test_card_names_the_call_and_its_arguments():
    body = "RESULT " + "x" * 2_000
    args = '{"query": "NVIDIA H100 market share 2025"}'
    content = _blanked(_one_call("web_search", args, body))
    assert "[Called: web_search(" in content
    assert "NVIDIA H100 market share 2025" in content


def test_card_carries_source_urls_from_the_discarded_body():
    body = "see https://nvidianews.nvidia.com/q3 and https://tomshardware.com/h100 " + "x" * 2_000
    content = _blanked(_one_call("web_search", '{"query": "h100"}', body))
    assert "[Source URLs]" in content
    assert "https://nvidianews.nvidia.com/q3" in content
    assert "https://tomshardware.com/h100" in content


def test_url_already_in_the_arguments_is_not_repeated():
    url = "https://example.com/report"
    body = f"fetched {url}\n" + "x" * 2_000
    content = _blanked(_one_call("web_fetch", f'{{"url": "{url}"}}', body))
    assert content.count(url) == 1


def test_url_heavy_body_stays_within_the_card_budget():
    urls = [f"https://example{i}.com/{'p' * 60}" for i in range(20)]
    body = " ".join(urls) + " " + "x" * 5_000
    content = _blanked(_one_call("web_search", '{"query": "many"}', body))
    card = _card_of(content)
    assert len(card) <= _MINI_CARD_BODY_MAX_CHARS
    assert sum(card.count(u) for u in urls) <= _MINI_CARD_MAX_URLS


def test_exact_rendered_card_stays_within_budget():
    # The header and newline are part of the model-visible card too. Choose URL
    # lengths that made the old per-URL accounting overshoot the cap.
    urls = [f"https://e.com/{letter * 68}" for letter in "abc"]
    args = "x" * _MINI_CARD_ARGS_MAX_CHARS
    content = _blanked(_one_call("web_search", args, " ".join(urls) + " " + "x" * 2_000))
    assert len(_card_of(content)) <= _MINI_CARD_BODY_MAX_CHARS


def test_overlong_arguments_are_truncated():
    args = '{"command": "' + "a" * 500 + '"}'
    content = _blanked(_one_call("bash", args, "OUT " + "x" * 2_000))
    call_line = _card_of(content).splitlines()[0]
    assert "…" in call_line
    assert len(call_line) < _MINI_CARD_ARGS_MAX_CHARS + 60
    assert len(_args_preview(args)) == _MINI_CARD_ARGS_MAX_CHARS


def test_multiline_arguments_are_flattened_to_one_line():
    args = '{"command": "cat <<EOF\\nline one\\nline two\\nEOF"}'
    content = _blanked(_one_call("bash", args, "OUT " + "x" * 2_000))
    card = _card_of(content)
    assert card.splitlines()[0].startswith("[Called: bash(")
    # No URLs in this body, so the whole card must be the single call line.
    assert len(card.splitlines()) == 1


# --- when the card is skipped ---------------------------------------------


def test_body_shorter_than_the_card_is_kept_verbatim():
    # No spill configured: replacing destroys the body, so a card that is not
    # even shorter is a pure loss.
    content = _blanked(_one_call("web_search", '{"query": "x"}', "ok"))
    assert content == "ok"


def test_declined_spill_keeps_the_body_at_any_size():
    # Pins the pre-existing branch the comment above the length check relies on:
    # a configured callback that declines returns the body verbatim regardless of
    # size, which is why the length check needs no minimum-size threshold.
    body = "RESULT " + "x" * 5_000
    content = _blanked(
        _one_call("web_search", '{"query": "x"}', body), spill=lambda _n, _c: None
    )
    assert content == body


def test_small_body_with_a_recovery_pointer_is_still_replaced():
    # Such a body is already an upstream-truncated preview, and the pointer only
    # reaches the model through spill_refs → the Tier 2 recovery index. Keeping
    # the body would strand the spilled full text as unrecoverable, so we replace
    # even though the card is longer.
    compactor = KeepLastNToolResultsCompactor(keep_tool_result=0)
    out = compactor.compact(
        _one_call("web_search", '{"query": "x"}', "tiny", result_store_ref="/spill/abc"),
        0,
    )
    tool_out = next(m for m in out if m.get("role") == "tool")
    assert tool_out["content"].startswith(OMITTED_TOOL_RESULT_PLACEHOLDER)
    assert tool_out["spill_refs"] == ["/spill/abc"]


# --- ordering and idempotency ---------------------------------------------


def test_recovery_pointer_stays_on_the_last_line():
    body = "see https://example.com/a " + "x" * 2_000
    content = _blanked(
        _one_call("web_search", '{"query": "x"}', body),
        spill=lambda _n, _c: "/spill/xyz",
    )
    assert content.splitlines()[-1] == "[Full text] /spill/xyz"
    assert content.splitlines()[0] == OMITTED_TOOL_RESULT_PLACEHOLDER


def test_second_pass_does_not_nest_a_card_inside_a_card():
    body = "see https://example.com/a " + "x" * 2_000
    messages = _one_call("web_search", '{"query": "x"}', body)
    compactor = KeepLastNToolResultsCompactor(keep_tool_result=0)
    once = compactor.compact(messages, 0)
    twice = compactor.compact(once, 0)
    first = next(m["content"] for m in once if m.get("role") == "tool")
    second = next(m["content"] for m in twice if m.get("role") == "tool")
    assert first == second
    assert second.count("[Called:") == 1


def test_legacy_placeholder_text_is_still_recognised():
    legacy = _LEGACY_OMITTED_TOOL_RESULT_PLACEHOLDERS[0] + "\n[Called: web_search({})]"
    content = _blanked(_one_call("web_search", '{"query": "x"}', legacy))
    assert content == legacy


# --- which existing handle backs a body --------------------------------------


def test_existing_spill_ref_wins_over_the_loop_cap_handle():
    # ``spill_refs`` describes the content still on the message; the loop-cap
    # ``result_store_ref`` describes the pre-truncation body upstream shed.
    # Reading the latter first would pin the wrong handle into the index.
    calls: list[str] = []
    compactor = KeepLastNToolResultsCompactor(
        keep_tool_result=0, spill=lambda _n, c: calls.append(c) or "/spill/fresh"
    )
    out = compactor.compact(
        _one_call(
            "web_search",
            '{"query": "x"}',
            "RESULT " + "x" * 2_000,
            spill_refs=["/spill/pinned"],
            result_store_ref="/spill/loop-cap",
        ),
        0,
    )
    tool_out = next(m for m in out if m.get("role") == "tool")
    assert tool_out["spill_refs"] == ["/spill/pinned"]
    assert calls == []  # already stored — must not spill a second copy


def test_loop_cap_handle_is_used_when_no_spill_ref_is_pinned():
    compactor = KeepLastNToolResultsCompactor(keep_tool_result=0)
    out = compactor.compact(
        _one_call(
            "web_search",
            '{"query": "x"}',
            "RESULT " + "x" * 2_000,
            result_store_ref="/spill/loop-cap",
        ),
        0,
    )
    tool_out = next(m for m in out if m.get("role") == "tool")
    assert tool_out["spill_refs"] == ["/spill/loop-cap"]


# --- the recovery footer hook ----------------------------------------------


def test_the_default_footer_names_no_tool():
    """A card is the whole message, so a wrong tool name here has no antidote.

    This module cannot know what a host calls its recovery tool, or whether that
    tool is bound for the agent whose history this is. Carrying only the handle
    is the one rendering that is correct in every host, which is why it is the
    default rather than a guess at the common case.
    """
    body = "see https://example.com/a " + "x" * 2_000
    content = _blanked(
        _one_call("web_search", '{"query": "x"}', body),
        spill=lambda _n, _c: "/spill/xyz",
    )
    assert content.splitlines()[-1] == default_recovery_footer("/spill/xyz")
    assert "recover" not in content.splitlines()[-1].lower()


def test_a_host_footer_replaces_the_last_line_and_nothing_else():
    body = "see https://example.com/a " + "x" * 2_000
    messages = _one_call("web_search", '{"query": "x"}', body)
    default = _blanked(messages, spill=lambda _n, _c: "/spill/xyz")
    hosted = _blanked(
        messages,
        spill=lambda _n, _c: "/spill/xyz",
        recovery_footer=lambda ref: f"[Saved. Fetch it with fetch_body id {ref}.]",
    )

    assert hosted.splitlines()[-1] == "[Saved. Fetch it with fetch_body id /spill/xyz.]"
    # The card above the footer — call line and source URLs — is untouched, so a
    # host swapping the footer cannot silently change the card's token budget.
    assert hosted.splitlines()[:-1] == default.splitlines()[:-1]


def test_a_host_footer_still_reaches_the_model_only_when_a_body_spilled():
    """No handle, no footer — a host renderer must not invent one.

    The card is emitted for unspilled bodies too (that is the whole point of the
    args + URLs lines). Calling the footer there would have it render a pointer
    to nothing.
    """
    calls: list[str] = []

    def footer(ref: str) -> str:
        calls.append(ref)
        return f"[Saved: {ref}]"

    content = _blanked(
        _one_call("web_search", '{"query": "x"}', "see https://example.com/a " + "x" * 2_000),
        recovery_footer=footer,
    )
    assert calls == []
    assert "[Saved:" not in content


def test_a_host_footer_survives_a_second_pass_unnested():
    """Idempotency is anchored on the placeholder, not the footer's wording.

    A host footer is free to be longer than the default, which is exactly the
    case where a second pass re-carding the message would compound. The
    already-placeheld check has to catch it regardless of what the last line says.
    """
    body = "see https://example.com/a " + "x" * 2_000
    messages = _one_call("web_search", '{"query": "x"}', body)
    compactor = KeepLastNToolResultsCompactor(
        keep_tool_result=0,
        spill=lambda _n, _c: "/spill/xyz",
        recovery_footer=lambda ref: (
            "[Full text saved. Recovery id: " + ref + " — use the recover_result tool "
            "(a tool call, not a shell command) with that spill id.]"
        ),
    )
    once = compactor.compact(messages, 0)
    twice = compactor.compact(once, 0)
    first = next(m["content"] for m in once if m.get("role") == "tool")
    second = next(m["content"] for m in twice if m.get("role") == "tool")
    assert first == second
    assert second.count("Recovery id:") == 1
