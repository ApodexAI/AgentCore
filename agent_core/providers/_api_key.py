"""API-key resolution shared by the OpenAI-SDK-backed clients."""

from __future__ import annotations

import os

#: Sent when neither the caller nor the environment supplies a key.
UNSET_OPENAI_API_KEY = "EMPTY"


def resolve_openai_api_key(api_key: str | None) -> str:
    """Return the key an ``AsyncOpenAI`` client should be constructed with.

    An empty or ``None`` key falls back to ``OPENAI_API_KEY``: an empty string
    (a config read before the environment was populated) would otherwise shadow
    the variable. With no variable either, return a placeholder rather than
    ``None``/``""``: the SDK raises on those at CONSTRUCTION (newer SDKs reject
    ``""`` too), which fails hosts that build an OpenAI client they never call,
    e.g. a default-provider client beside an Anthropic workflow LLM. A client
    that is called fails at request time with the endpoint's 401 instead;
    keyless OpenAI-compatible servers (vLLM, a proxy) accept it as-is.
    """
    return api_key or os.environ.get("OPENAI_API_KEY") or UNSET_OPENAI_API_KEY
