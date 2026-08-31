"""Non-blocking tiktoken encoder access.

``tiktoken.get_encoding()`` fetches the BPE ranks file over HTTP on a
cache miss (``openaipublic.blob.core.windows.net``) with **no timeout**.
Run inline on the asyncio loop thread inside an egress-restricted
container, that synchronous fetch freezes the *entire* event loop for
minutes — the 2026-06-05 partial3 hang (182 s at the first sub-agent
spawn) and the 2026-06-08 swarm-gv worker hang both trace to a lazy
``get_encoding()`` running on the loop.

Two layers of defense protect the loop:

1. **Image-baked vocab** (``TIKTOKEN_CACHE_DIR``) turns the fetch into a
   local file read — see ``docker/stateful-agent.Dockerfile``.
2. **This module**: the first request for an encoding kicks the
   (potentially network-fetching) init onto a daemon thread and returns
   ``None``; callers fall back to a chars/4 heuristic until the encoder
   lands. The loop thread NEVER blocks on tiktoken, cache-baked or not.

NOTE: ``llm_client.py`` keeps an equivalent private copy (with an
``o200k_base`` → ``cl100k_base`` preference) that predates this module
and is left untouched to avoid churning tested code. New callers should
use :func:`get_encoding_nonblocking`.
"""

from __future__ import annotations

import importlib
import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Sentinel for "this name has never been requested" — distinct from the
# ``None`` we store to mark a load that is in flight.
_MISSING = object()

# Per-encoding cache. State machine for a given name:
#   absent (==_MISSING) → never requested
#   None                → background init in flight; use the heuristic for now
#   False               → tiktoken unavailable / bad name; terminal, heuristic forever
#   <Encoding object>   → ready
_encoders: dict[str, Any] = {}
_lock = threading.Lock()


def _load(name: str) -> None:
    """Blocking tiktoken init — only ever runs on a daemon thread."""
    enc: Any
    try:
        tiktoken: Any = importlib.import_module("tiktoken")
        enc = tiktoken.get_encoding(name)
    except Exception:  # not installed / fetch failure / bad name
        enc = False
        logger.debug("tiktoken encoding %r unavailable; using heuristic", name)
    with _lock:
        _encoders[name] = enc


def get_encoding_nonblocking(name: str = "cl100k_base") -> Any | None:
    """Return the cached tiktoken encoder for ``name`` without ever blocking.

    The first call schedules a daemon-thread init and returns ``None``;
    later calls return the encoder once it has loaded, or ``None`` while
    it is still loading. Returns ``None`` permanently when tiktoken is
    unavailable — callers MUST fall back to a heuristic on ``None``.
    """
    enc = _encoders.get(name, _MISSING)
    if enc is not _MISSING:
        return enc or None  # None (loading) and False (failed) both collapse to None
    with _lock:
        if _encoders.get(name, _MISSING) is _MISSING:  # still unclaimed under the lock
            _encoders[name] = None  # mark loading so concurrent callers don't re-spawn
            threading.Thread(
                target=_load,
                args=(name,),
                name=f"tiktoken-init-{name}",
                daemon=True,
            ).start()
    return None
