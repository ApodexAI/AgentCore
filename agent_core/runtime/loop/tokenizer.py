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

What is deliberately NOT on that daemon thread is ``import tiktoken``.
``tiktoken._tiktoken`` is a Rust extension, so the import ``dlopen``s a
shared object; CPython kills daemon threads mid-flight during
finalization (``pthread_exit`` at the next GIL acquisition), and being
killed inside the dynamic linker is not survivable — losing glibc's
``_dl_load_lock`` shows up as a later SIGSEGV, and unwinding through a
Rust / ``extern "C"`` frame calls ``abort()``. The 2026-09-05
investigation found the thread parked inside that import at
interpreter-exit on *every* short run, with two observed deaths after a
fully correct protocol stream (``-11`` in
``test_stateless_across_invocations``, ``-6`` in
``test_serve_subprocess_e2e``). The import is a local dlopen — 24-29 ms,
no network, nothing the docstring above is defending against — so it
belongs on the caller thread. Only ``get_encoding()`` (140 ms+, and
unbounded on a cache miss) needs the thread.

Belt and braces: an ``atexit`` hook joins any in-flight init, following
``providers/nonblocking_stream.py``. CPython runs ``atexit`` callbacks
before it starts killing daemon threads, so that is the last point at
which the load can be drained cleanly.
"""

from __future__ import annotations

import atexit
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

# In-flight init threads, keyed by encoding name, so ``_join_pending`` can
# drain them at exit. Each thread removes its own entry when it finishes.
_threads: dict[str, threading.Thread] = {}
_atexit_registered = False

# Long enough for a cache-hit ``get_encoding`` (~140 ms) to land, short
# enough that a wedged network fetch cannot hold up process exit. A load
# that misses this deadline is left where it is — the thread no longer
# touches the dynamic linker, which is what made the mid-flight kill
# dangerous in the first place.
_JOIN_TIMEOUT_S = 1.0


def _join_pending() -> None:
    """atexit: drain in-flight inits before the interpreter kills them."""
    with _lock:
        pending = list(_threads.values())
    for thread in pending:
        thread.join(timeout=_JOIN_TIMEOUT_S)


def _load(name: str, tiktoken: Any) -> None:
    """Blocking ``get_encoding`` — only ever runs on a daemon thread.

    ``tiktoken`` is passed in already imported: this function must not
    import anything, see the module docstring.
    """
    enc: Any
    try:
        enc = tiktoken.get_encoding(name)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    except Exception:  # fetch failure / bad name
        enc = False
        logger.debug("tiktoken encoding %r unavailable; using heuristic", name)
    with _lock:
        _encoders[name] = enc
        _threads.pop(name, None)


def get_encoding_nonblocking(name: str = "cl100k_base") -> Any | None:
    """Return the cached tiktoken encoder for ``name`` without ever blocking.

    The first call for a name imports tiktoken on the calling thread (a
    local dlopen, no network), schedules the encoder init on a daemon
    thread and returns ``None``; later calls return the encoder once it
    has loaded, or ``None`` while it is still loading. Returns ``None``
    permanently when tiktoken is unavailable — callers MUST fall back to
    a heuristic on ``None``.
    """
    global _atexit_registered

    enc = _encoders.get(name, _MISSING)
    if enc is not _MISSING:
        return enc or None  # None (loading) and False (failed) both collapse to None

    # On the caller thread, deliberately — never on the daemon thread.
    try:
        import tiktoken  # pyright: ignore[reportMissingImports]
    except Exception:  # not installed
        with _lock:
            _encoders[name] = False
        logger.debug("tiktoken unavailable; using heuristic")
        return None

    with _lock:
        if _encoders.get(name, _MISSING) is not _MISSING:  # claimed while we imported
            return _encoders[name] or None
        _encoders[name] = None  # mark loading so concurrent callers don't re-spawn
        thread = threading.Thread(
            target=_load,
            args=(name, tiktoken),
            name=f"tiktoken-init-{name}",
            daemon=True,
        )
        _threads[name] = thread
        if not _atexit_registered:
            atexit.register(_join_pending)
            _atexit_registered = True
        thread.start()
    return None
