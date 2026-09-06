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
   ``None``; callers fall back to the CJK-aware heuristic in
   ``context_budget.estimate_tokens`` until the encoder lands. The loop
   thread NEVER blocks on tiktoken, cache-baked or not.

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

Hence the third layer: no thread is started unless the requested vocab's
exact cache artifact exists and passes the hash tiktoken itself expects.
An unrelated or corrupt cache file cannot accidentally reopen the fetch
path. There is then nothing to kill, at the cost of approximate token
counts on a host that never warmed the cache. A host that wants the
fetch anyway sets ``AGENT_CORE_TIKTOKEN_FETCH=1`` and accepts the window.

Proving the load is local costs a SHA-256 over the cached vocab — ~1.7 MB
for ``cl100k_base``, single-digit milliseconds — and it runs on the
calling thread, like the import above it. Same reasoning: a bounded local
read is not what the network defense above exists for, and it happens
once per encoding per process (the ``_encoders`` entry is written either
way, so a gated name is never re-hashed). Verifying with tiktoken's own
constructor instead would mean *calling the thing that may fetch*, which
is the operation being avoided.
"""

from __future__ import annotations

import atexit
import hashlib
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

# tiktoken caches the downloaded bytes under ``sha1(blobpath)`` and validates
# them against the constructor's expected SHA-256 before parsing. All AgentCore
# callers currently request cl100k_base. Keeping both values here lets us prove
# that this exact load is local without invoking tiktoken's constructor (which
# is the operation that may fetch). If tiktoken changes either value, this gate
# fails closed until the table is updated.
_CACHE_SPECS: dict[str, tuple[str, str]] = {
    "cl100k_base": (
        "9b5ad71b2ce5302211f9c61530b329a4922fc6a4",
        "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
    ),
}

# The "no valid target cache, staying on the heuristic" warning is worth once.
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


def _cache_problem(name: str) -> str | None:
    """Why ``name`` cannot be proved to load locally, or ``None`` if it can."""
    directory = _cache_dir()
    if not directory:
        return "disabled"

    spec = _CACHE_SPECS.get(name)
    if spec is None:
        return "unverified encoding"

    cache_key, expected_hash = spec
    path = os.path.join(directory, cache_key)
    try:
        with open(path, "rb") as cache_file:
            actual_hash = hashlib.file_digest(cache_file, "sha256").hexdigest()
    except FileNotFoundError:
        return "missing"
    except OSError:
        return "unreadable"
    return None if actual_hash == expected_hash else "invalid"


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
    requested vocab has no valid cache artifact so the init could issue
    an unbounded network fetch (``AGENT_CORE_TIKTOKEN_FETCH=1`` opts
    back in) — callers MUST fall back to a heuristic on ``None``.
    """
    global _atexit_registered, _warned_uncached

    enc = _encoders.get(name, _MISSING)
    if enc is not _MISSING:
        return enc or None  # None (loading) and False (failed) both collapse to None

    # A load that can only be served over the network is not worth a
    # thread: it cannot finish inside the exit-time join budget, and
    # being killed mid-fetch is what corrupts the heap.
    cache_problem = _cache_problem(name)
    if cache_problem is not None and not _network_fetch_allowed():
        with _lock:
            _encoders[name] = False
        if not _warned_uncached:
            _warned_uncached = True
            directory = _cache_dir()
            if cache_problem == "disabled":
                logger.warning(
                    "tiktoken caching is disabled by TIKTOKEN_CACHE_DIR=''; "
                    "token counts stay approximate (CJK-aware heuristic). Set "
                    "TIKTOKEN_CACHE_DIR to a writable directory, warm %s there, "
                    "or set %s=1 to fetch it at runtime.",
                    name,
                    _FETCH_ENV,
                )
            else:
                logger.warning(
                    "tiktoken vocab cache %r has no valid %s artifact (%s); "
                    "token counts stay approximate (CJK-aware heuristic). Warm it "
                    "once with "
                    "`python -c 'import tiktoken; tiktoken.get_encoding(\"%s\")'` "
                    "after ensuring TIKTOKEN_CACHE_DIR points to a readable and "
                    "writable cache; set %s=1 to fetch it at runtime instead.",
                    directory,
                    name,
                    cache_problem,
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
