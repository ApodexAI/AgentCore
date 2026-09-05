"""The tiktoken loader must not leave a daemon thread inside an import.

``import tiktoken`` dlopens a Rust extension. A daemon thread killed
mid-import at interpreter finalization takes glibc's ``_dl_load_lock``
with it (later SIGSEGV) or unwinds through an ``extern "C"`` frame
(``abort()``); both were observed in production as a signal death
*after* a fully correct protocol stream. So the import belongs on the
caller thread and only ``get_encoding`` on the daemon thread — these
tests pin which thread runs which, and that the exit hook drains the
load.
"""

from __future__ import annotations

import importlib.util
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
def clean_tokenizer():
    """Reset the module cache and unhook any real/fake tiktoken."""
    saved_module = sys.modules.pop("tiktoken", None)
    saved_meta = list(sys.meta_path)
    tokenizer._encoders.clear()
    tokenizer._threads.clear()
    tokenizer._atexit_registered = True  # don't leak a real atexit hook per test
    try:
        yield
    finally:
        tokenizer._join_pending()
        tokenizer._encoders.clear()
        tokenizer._threads.clear()
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


def test_atexit_hook_drains_the_in_flight_load(clean_tokenizer) -> None:
    loader = _RecordingLoader()
    sys.meta_path.insert(0, loader)

    tokenizer.get_encoding_nonblocking("cl100k_base")
    assert "cl100k_base" in tokenizer._threads  # in flight

    tokenizer._join_pending()
    assert tokenizer._threads == {}  # thread finished and deregistered itself
    assert tokenizer._encoders["cl100k_base"] is loader.encoder


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
