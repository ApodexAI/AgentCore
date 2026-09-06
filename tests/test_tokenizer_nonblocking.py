"""The tiktoken loader must not leave a daemon thread inside an import.

``import tiktoken`` dlopens a Rust extension. A daemon thread killed
mid-import at interpreter finalization takes glibc's ``_dl_load_lock``
with it (later SIGSEGV) or unwinds through an ``extern "C"`` frame
(``abort()``); both were observed in production as a signal death
*after* a fully correct protocol stream. So the import belongs on the
caller thread and only ``get_encoding`` on the daemon thread — these
tests pin which thread runs which, and that the exit hook drains the
load.

The load itself is only started when the vocab cache can serve it. A
cache miss means an unbounded fetch that outlives the exit-time join, so
the thread gets killed mid-``malloc`` instead — reproduced as
``double free or corruption (fasttop)`` / ``-6``. The gate around that,
and its ``AGENT_CORE_TIKTOKEN_FETCH`` escape hatch, are pinned below.
Every test therefore has to be explicit about the cache state; the
fixture points ``TIKTOKEN_CACHE_DIR`` at a warm directory so the host's
own ``/tmp`` cannot decide which branch runs.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import re
import sys
import threading
import types
from typing import Any

import pytest

from agent_core.runtime.loop import tokenizer


class _RecordingLoader:
    """Stands in for the real ``tiktoken``, recording the loading thread."""

    def __init__(self, gate: threading.Event | None = None) -> None:
        self.import_thread: str | None = None
        self.encode_thread: str | None = None
        self.encoder = object()
        self._gate = gate

    # -- meta_path finder -------------------------------------------------
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname != "tiktoken":
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec: Any) -> types.ModuleType:
        return types.ModuleType(spec.name)

    def exec_module(self, module: types.ModuleType) -> None:
        self.import_thread = threading.current_thread().name
        module.get_encoding = self._get_encoding  # pyright: ignore[reportAttributeAccessIssue]

    # -- the fake tiktoken API -------------------------------------------
    def _get_encoding(self, name: str) -> object:
        self.encode_thread = threading.current_thread().name
        if self._gate is not None:
            self._gate.wait(timeout=5.0)
        return self.encoder


@pytest.fixture
def warm_cache_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """A valid target vocab cache, so the fetch gate lets loads run."""
    cache = tmp_path / "data-gym-cache"
    cache.mkdir()
    cache_key, _ = tokenizer._CACHE_SPECS["cl100k_base"]
    payload = b"ranks"
    monkeypatch.setitem(
        tokenizer._CACHE_SPECS,
        "cl100k_base",
        (cache_key, hashlib.sha256(payload).hexdigest()),
    )
    (cache / cache_key).write_bytes(payload)
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(cache))
    monkeypatch.delenv("AGENT_CORE_TIKTOKEN_FETCH", raising=False)
    return cache


@pytest.fixture
def clean_tokenizer(warm_cache_dir):
    """Reset the module cache and unhook any real/fake tiktoken."""
    saved_module = sys.modules.pop("tiktoken", None)
    saved_meta = list(sys.meta_path)
    saved_atexit_registered = tokenizer._atexit_registered
    saved_warned = tokenizer._warned_uncached
    tokenizer._warned_uncached = False  # the warning is once-per-process
    tokenizer._encoders.clear()
    tokenizer._threads.clear()
    tokenizer._atexit_registered = True  # don't leak a real atexit hook per test
    try:
        yield
    finally:
        tokenizer._join_pending()
        tokenizer._encoders.clear()
        tokenizer._threads.clear()
        tokenizer._atexit_registered = saved_atexit_registered
        tokenizer._warned_uncached = saved_warned
        sys.meta_path[:] = saved_meta
        sys.modules.pop("tiktoken", None)
        if saved_module is not None:
            sys.modules["tiktoken"] = saved_module


def test_import_runs_on_caller_thread_get_encoding_on_daemon(clean_tokenizer) -> None:
    loader = _RecordingLoader()
    sys.meta_path.insert(0, loader)

    assert tokenizer.get_encoding_nonblocking("cl100k_base") is None
    # The import must already have happened, synchronously, right here.
    assert loader.import_thread == threading.current_thread().name
    assert "tiktoken" in sys.modules

    tokenizer._join_pending()
    assert loader.encode_thread == "tiktoken-init-cl100k_base"
    assert tokenizer.get_encoding_nonblocking("cl100k_base") is loader.encoder


def test_atexit_hook_registers_once_and_drains_the_in_flight_load(
    clean_tokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = threading.Event()
    loader = _RecordingLoader(gate=gate)
    sys.meta_path.insert(0, loader)
    registered: list[Any] = []
    monkeypatch.setattr(tokenizer.atexit, "register", registered.append)
    tokenizer._atexit_registered = False

    tokenizer.get_encoding_nonblocking("cl100k_base")
    tokenizer.get_encoding_nonblocking("o200k_base")
    assert "cl100k_base" in tokenizer._threads  # in flight
    assert registered == [tokenizer._join_pending]

    gate.set()
    registered[0]()
    assert tokenizer._threads == {}  # thread finished and deregistered itself
    assert tokenizer._encoders["cl100k_base"] is loader.encoder


def test_atexit_join_timeout_is_shared_across_threads(
    clean_tokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Clock:
        now = 10.0

        def monotonic(self) -> float:
            return self.now

    class _PendingThread:
        def __init__(self, clock: _Clock) -> None:
            self.clock = clock
            self.timeouts: list[float] = []

        def join(self, timeout: float | None = None) -> None:
            assert timeout is not None
            self.timeouts.append(timeout)
            self.clock.now += 0.4

    clock = _Clock()
    pending = [_PendingThread(clock) for _ in range(4)]
    tokenizer._threads.update({str(i): thread for i, thread in enumerate(pending)})  # type: ignore[arg-type]
    monkeypatch.setattr(tokenizer.time, "monotonic", clock.monotonic)

    tokenizer._join_pending()

    assert [thread.timeouts for thread in pending] == [
        [pytest.approx(1.0)],
        [pytest.approx(0.6)],
        [pytest.approx(0.2)],
        [],
    ]


def test_caller_never_blocks_on_a_wedged_get_encoding(clean_tokenizer) -> None:
    gate = threading.Event()
    loader = _RecordingLoader(gate=gate)
    sys.meta_path.insert(0, loader)
    try:
        # Returns immediately even though get_encoding is parked.
        assert tokenizer.get_encoding_nonblocking("cl100k_base") is None
        assert tokenizer.get_encoding_nonblocking("cl100k_base") is None
    finally:
        gate.set()


def test_missing_tiktoken_is_terminal_and_spawns_no_thread(clean_tokenizer) -> None:
    class _Blocker:
        def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
            if fullname == "tiktoken":
                raise ImportError("no tiktoken")
            return None

    sys.meta_path.insert(0, _Blocker())

    assert tokenizer.get_encoding_nonblocking("cl100k_base") is None
    assert tokenizer._threads == {}
    assert tokenizer._encoders["cl100k_base"] is False
    assert tokenizer.get_encoding_nonblocking("cl100k_base") is None


def test_concurrent_callers_spawn_one_thread(clean_tokenizer) -> None:
    gate = threading.Event()
    loader = _RecordingLoader(gate=gate)
    sys.meta_path.insert(0, loader)
    try:
        start = threading.Barrier(4)

        def call() -> None:
            start.wait(timeout=5.0)
            tokenizer.get_encoding_nonblocking("cl100k_base")

        callers = [threading.Thread(target=call) for _ in range(4)]
        for t in callers:
            t.start()
        for t in callers:
            t.join(timeout=5.0)

        assert len(tokenizer._threads) == 1
    finally:
        gate.set()


# -- the cold-cache fetch gate ------------------------------------------------


def test_empty_cache_dir_stays_on_the_heuristic_and_spawns_no_thread(
    clean_tokenizer, tmp_path, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    """No thread means nothing for finalization to kill mid-fetch."""
    loader = _RecordingLoader()
    sys.meta_path.insert(0, loader)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(empty))

    with caplog.at_level("WARNING", logger=tokenizer.logger.name):
        assert tokenizer.get_encoding_nonblocking("cl100k_base") is None

    assert tokenizer._threads == {}
    assert loader.import_thread is None  # not even imported
    assert tokenizer._encoders["cl100k_base"] is False  # terminal
    assert tokenizer.get_encoding_nonblocking("cl100k_base") is None
    assert "TIKTOKEN_CACHE_DIR" in caplog.text
    assert "AGENT_CORE_TIKTOKEN_FETCH" in caplog.text


@pytest.mark.parametrize(
    "cache_dir",
    ["", "does/not/exist"],
    ids=["caching-disabled", "missing-directory"],
)
def test_cache_problem_without_a_usable_cache(
    clean_tokenizer, tmp_path, monkeypatch: pytest.MonkeyPatch, cache_dir: str
) -> None:
    target = "" if cache_dir == "" else str(tmp_path / cache_dir)
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", target)
    assert tokenizer._cache_problem("cl100k_base") is not None


def test_populated_cache_lets_the_load_run(clean_tokenizer) -> None:
    assert tokenizer._cache_problem("cl100k_base") is None

    loader = _RecordingLoader()
    sys.meta_path.insert(0, loader)
    assert tokenizer.get_encoding_nonblocking("cl100k_base") is None
    tokenizer._join_pending()
    assert tokenizer.get_encoding_nonblocking("cl100k_base") is loader.encoder


def test_unrelated_cache_file_spawns_no_thread(
    clean_tokenizer, warm_cache_dir
) -> None:
    cache_key, _ = tokenizer._CACHE_SPECS["cl100k_base"]
    (warm_cache_dir / cache_key).unlink()
    (warm_cache_dir / "another-encoding").write_bytes(b"valid for something else")
    loader = _RecordingLoader()
    sys.meta_path.insert(0, loader)

    assert tokenizer._cache_problem("cl100k_base") is not None
    assert tokenizer.get_encoding_nonblocking("cl100k_base") is None
    assert tokenizer._threads == {}
    assert loader.import_thread is None


def test_corrupt_target_cache_file_spawns_no_thread(
    clean_tokenizer, warm_cache_dir
) -> None:
    cache_key, _ = tokenizer._CACHE_SPECS["cl100k_base"]
    (warm_cache_dir / cache_key).write_bytes(b"corrupt")
    loader = _RecordingLoader()
    sys.meta_path.insert(0, loader)

    assert tokenizer._cache_problem("cl100k_base") is not None
    assert tokenizer.get_encoding_nonblocking("cl100k_base") is None
    assert tokenizer._threads == {}
    assert loader.import_thread is None


def test_unknown_encoding_fails_closed(clean_tokenizer) -> None:
    loader = _RecordingLoader()
    sys.meta_path.insert(0, loader)

    assert tokenizer._cache_problem("future_encoding") is not None
    assert tokenizer.get_encoding_nonblocking("future_encoding") is None
    assert tokenizer._threads == {}
    assert loader.import_thread is None


def test_data_gym_cache_dir_is_the_second_choice(
    clean_tokenizer, warm_cache_dir, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TIKTOKEN_CACHE_DIR", raising=False)
    monkeypatch.setenv("DATA_GYM_CACHE_DIR", str(warm_cache_dir))
    assert tokenizer._cache_dir() == str(warm_cache_dir)
    assert tokenizer._cache_problem("cl100k_base") is None


def test_disabled_cache_warning_requires_a_writable_directory(
    clean_tokenizer, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", "")

    with caplog.at_level("WARNING", logger=tokenizer.logger.name):
        assert tokenizer.get_encoding_nonblocking("cl100k_base") is None

    assert "caching is disabled" in caplog.text
    assert "writable directory" in caplog.text


@pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
def test_explicit_opt_in_restores_the_network_fetch(
    clean_tokenizer, tmp_path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    loader = _RecordingLoader()
    sys.meta_path.insert(0, loader)
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path / "gone"))
    monkeypatch.setenv("AGENT_CORE_TIKTOKEN_FETCH", value)

    assert tokenizer.get_encoding_nonblocking("cl100k_base") is None
    assert loader.import_thread == threading.current_thread().name
    tokenizer._join_pending()
    assert tokenizer._encoders["cl100k_base"] is loader.encoder


@pytest.mark.parametrize("value", ["", "0", "false", "no"])
def test_unrecognised_opt_in_values_keep_the_gate_closed(
    clean_tokenizer, tmp_path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path / "gone"))
    monkeypatch.setenv("AGENT_CORE_TIKTOKEN_FETCH", value)
    assert tokenizer._network_fetch_allowed() is False


def test_pinned_cache_metadata_matches_tiktokens_own_declaration() -> None:
    """The pinned cache key and content hash must be tiktoken's actual ones.

    ``_CACHE_SPECS`` is otherwise unfalsifiable. Every test above monkeypatches
    it and writes a payload whose hash it just computed, so a typo in either
    constant — or a tiktoken release that re-publishes a vocab — leaves the gate
    permanently closed with nothing going red. That failure is silent by
    construction: exact token counts become heuristic ones and the one WARNING
    blames the host's cache for a directory that is in fact correctly warmed.

    Read the truth out of ``tiktoken_ext.openai_public`` *without* calling the
    constructor, because calling it is the fetch this whole module exists to
    avoid. The cache key is ``sha1(blobpath)`` (tiktoken's ``read_file_cached``)
    and the content hash is the ``expected_hash`` its loader validates against.
    """
    pub = pytest.importorskip(
        "tiktoken_ext.openai_public",
        reason="tiktoken is the optional `tokenizer` extra; install it to check these pins",
    )
    for name, (cache_key, expected_hash) in tokenizer._CACHE_SPECS.items():
        constructor = getattr(pub, name, None)
        assert constructor is not None, (
            f"tiktoken no longer defines a {name!r} constructor, so _CACHE_SPECS pins a "
            "vocab upstream does not publish under that name"
        )
        source = inspect.getsource(constructor)
        blobpath = re.search(r'"(https://\S+?\.tiktoken)"', source)
        declared = re.search(r'expected_hash\s*=\s*"([0-9a-f]{64})"', source)
        assert blobpath and declared, (
            f"cannot read {name!r}'s blobpath and expected_hash out of tiktoken's source; "
            "its shape changed, so verify _CACHE_SPECS by hand and repair this test"
        )
        assert hashlib.sha1(blobpath.group(1).encode()).hexdigest() == cache_key, (
            f"{name!r} cache key is stale: tiktoken caches {blobpath.group(1)} under a "
            "different name now, so the gate can never find a warm cache"
        )
        assert declared.group(1) == expected_hash, (
            f"{name!r} content hash is stale: tiktoken expects {declared.group(1)}, so a "
            "correctly warmed cache reads as invalid and exact counts stay off"
        )
