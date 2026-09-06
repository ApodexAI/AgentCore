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

That join has a 1 s budget, which covers a cache *hit* (~140 ms) and
nothing else. On a cache **miss** ``get_encoding`` does an unbounded
``requests.get`` (tiktoken's ``load.py`` passes no timeout) plus a BPE
parse, so it is still mid-flight when the budget expires and the thread
is killed anyway — no longer inside the dynamic linker, but plausibly
inside ``malloc``. Reproduced 2026-09-06 on
``test_serve_subprocess_e2e``: with the vocab cache guaranteed empty,
1 of 90 runs died with ``double free or corruption (fasttop)`` and
``-6`` after a fully correct protocol stream, while 60/60 warm runs and
400 warm/blocked-egress micro-probes were clean. So the previous fix
covered only the warm half.

Hence the third layer: when a load could only be served over the
network — no cache directory, or an empty one — no thread is started at
all and the name stays on the chars/4 heuristic. There is then nothing
to kill, at the cost of approximate token counts on a host that never
warmed the cache. A host that wants the fetch anyway sets
``AGENT_CORE_TIKTOKEN_FETCH=1`` and accepts the window.
"""

from __future__ import annotations

import atexit
import logging
import os
import tempfile
import threading
import time
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

# Opt back in to the unbounded network fetch (and its exit-time window).
_FETCH_ENV = "AGENT_CORE_TIKTOKEN_FETCH"
_TRUTHY = frozenset({"1", "true", "yes", "on"})

# The "no cache, staying on the heuristic" warning is worth saying once.
_warned_uncached = False


def _cache_dir() -> str:
    """Where tiktoken would look for a cached vocab.

    Mirrors ``tiktoken.load.read_file_cached``: ``TIKTOKEN_CACHE_DIR``,
    else ``DATA_GYM_CACHE_DIR``, else ``<tmp>/data-gym-cache``. An empty
    string is tiktoken's way of disabling the cache entirely.
    """
    for var in ("TIKTOKEN_CACHE_DIR", "DATA_GYM_CACHE_DIR"):
        value = os.environ.get(var)
        if value is not None:
            return value
    return os.path.join(tempfile.gettempdir(), "data-gym-cache")


def _fetch_is_certain() -> bool:
    """True when a load could only be served over the network.

    Deliberately coarse: tiktoken keys cache files by
    ``sha1(blobpath)``, and the blobpath only exists inside the
    constructor we are trying not to call, so this cannot ask about one
    encoding. "The cache holds nothing at all" is the state that
    actually occurs — a fresh checkout, an image built without the
    warm-up, a test with ``TMPDIR`` pointed somewhere empty — and it is
    answerable with one ``scandir``.
    """
    directory = _cache_dir()
    if not directory:
        return True
    try:
        with os.scandir(directory) as entries:
            return not any(entry.is_file() for entry in entries)
    except OSError:  # missing, or unreadable
        return True


def _network_fetch_allowed() -> bool:
    return os.environ.get(_FETCH_ENV, "").strip().lower() in _TRUTHY


def _join_pending() -> None:
    """atexit: drain in-flight inits before the interpreter kills them."""
    with _lock:
        pending = list(_threads.values())
    deadline = time.monotonic() + _JOIN_TIMEOUT_S
    for thread in pending:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        thread.join(timeout=remaining)


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
    permanently when tiktoken is unavailable, and permanently when the
    vocab cache is empty so the init could only be served by an
    unbounded network fetch (``AGENT_CORE_TIKTOKEN_FETCH=1`` opts back
    in) — callers MUST fall back to a heuristic on ``None``.
    """
    global _atexit_registered, _warned_uncached

    enc = _encoders.get(name, _MISSING)
    if enc is not _MISSING:
        return enc or None  # None (loading) and False (failed) both collapse to None

    # A load that can only be served over the network is not worth a
    # thread: it cannot finish inside the exit-time join budget, and
    # being killed mid-fetch is what corrupts the heap.
    if _fetch_is_certain() and not _network_fetch_allowed():
        with _lock:
            _encoders[name] = False
        if not _warned_uncached:
            _warned_uncached = True
            logger.warning(
                "tiktoken vocab cache %r is empty; token counts stay approximate "
                "(chars/4). Warm it once with "
                "`python -c 'import tiktoken; tiktoken.get_encoding(\"%s\")'` "
                "(or bake TIKTOKEN_CACHE_DIR into the image); set %s=1 to fetch "
                "it at runtime instead.",
                _cache_dir(),
                name,
                _FETCH_ENV,
            )
        return None

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
