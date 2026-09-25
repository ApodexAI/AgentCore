"""An empty ``api_key`` must fall back to ``OPENAI_API_KEY`` like ``None``."""

from __future__ import annotations

import pytest

from agent_core.providers.openai_chat import OpenAIClient
from agent_core.providers.openai_responses import OpenAIResponsesClient


@pytest.mark.parametrize("client_cls", [OpenAIClient, OpenAIResponsesClient])
def test_empty_api_key_uses_the_environment(
    client_cls: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")

    client = client_cls("gpt-test", api_key="")

    assert client._client.api_key == "from-env"


@pytest.mark.parametrize("client_cls", [OpenAIClient, OpenAIResponsesClient])
def test_explicit_api_key_wins(client_cls: type, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")

    client = client_cls("gpt-test", api_key="explicit")

    assert client._client.api_key == "explicit"
