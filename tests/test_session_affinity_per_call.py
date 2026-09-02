"""A scoped client must pin a replica even when the caller sends no headers."""

from __future__ import annotations

from typing import Any

from agent_core.providers.openai_chat import OpenAIClient


def _client(scope: str) -> OpenAIClient:
    return OpenAIClient(
        "m",
        api_key="k",
        base_url="https://host/v1",
        default_headers={"x-session-id": "sess-1"},
        # A scoped resolver makes __init__ withhold ``default_query`` from the
        # SDK, so the per-call path is the only source of affinity.
        session_query_resolver=lambda headers: (
            {"session": headers["x-session-id"]}
            if headers and "x-session-id" in headers
            else {}
        ),
        session_scope_resolver=lambda: scope,
    )


def test_scoped_client_withholds_sdk_default_query() -> None:
    client = _client("task-1")
    assert client._client._custom_query in (None, {})


def test_affinity_is_supplied_without_extra_headers() -> None:
    """The regression: gating on ``extra_headers`` pinned no replica at all."""
    client = _client("task-1")
    query = client._session_query(None)
    assert query == {"session": "sess-1"}


def test_per_call_headers_win_over_construction_time() -> None:
    client = _client("task-1")
    query = client._session_query({"x-session-id": "sess-2"})
    assert query == {"session": "sess-2"}


def test_stale_scope_drops_construction_time_affinity() -> None:
    client = _client("task-1")
    # The task that built the client is gone; the cached client outlived it.
    client._session_scope_resolver = lambda: "task-2"
    assert client._session_query(None) == {}


def test_kwargs_carry_extra_query_with_no_extra_headers(
    monkeypatch: Any,
) -> None:
    """End-to-end: the request kwargs actually receive ``extra_query``."""
    client = _client("task-1")
    captured: dict[str, Any] = {}

    async def fake_create(**kwargs: Any) -> Any:
        captured.update(kwargs)
        raise RuntimeError("stop after capture")

    monkeypatch.setattr(
        client._client.chat.completions,
        "create",
        fake_create,
    )
    import asyncio

    import pytest

    from agent_core.messages import user_msg

    with pytest.raises(Exception, match="stop after capture"):
        asyncio.run(client.chat([user_msg("hi")]))
    # The caller passed no session header, yet the request still pins a replica.
    assert captured["extra_query"] == {"session": "sess-1"}
    assert "x-session-id" not in captured.get("extra_headers", {})
