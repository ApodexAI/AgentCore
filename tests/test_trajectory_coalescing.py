"""``0`` disables a coalescing bound; it must not mean "never flush".

The module header states "0 disables the respective bound". The predicate read
a disabled bound as *never due* instead, so with ``COALESCE_N=0`` (and the
default ``COALESCE_MS=0``) the JSON snapshot was written once and then frozen
until ``on_loop_end`` forced a final flush. A process killed before that — an
OOM kill, a hard timeout — left a file containing only the ``start`` state,
which is exactly the situation the snapshot exists for.
"""

from __future__ import annotations

import importlib
import json

import pytest

from agent_core.loop_types import LoopConfig


def _observer(tmp_path, monkeypatch, *, coalesce_n: int, coalesce_ms: int = 0):
    """Re-import the module so its env-derived constants are re-read.

    Flushes are counted by spying on ``_write_envelope``: the coalescing
    predicate decides *whether* a rewrite happens, which is the behaviour under
    test — the document's contents are a separate concern.
    """
    monkeypatch.setenv("SWARM_TRAJECTORY_COALESCE_N", str(coalesce_n))
    monkeypatch.setenv("SWARM_TRAJECTORY_COALESCE_MS", str(coalesce_ms))
    import agent_core.components.observers.trajectory as traj

    traj = importlib.reload(traj)
    obs = traj.TrajectoryFileObserver(tmp_path, filename="t", formats=["json"])
    writes: list[int] = []
    original = obs._write_envelope
    def _spy(path):
        writes.append(1)
        return original(path)
    monkeypatch.setattr(obs, "_write_envelope", _spy)
    return obs, writes


async def _flush_n_times(obs, n: int) -> None:
    """Drive ``_flush_json`` exactly ``n`` times through the non-forced path."""
    for _ in range(n):
        obs._flush_json()


@pytest.mark.asyncio
async def test_disabled_bounds_flush_every_event(tmp_path, monkeypatch):
    obs, writes = _observer(tmp_path, monkeypatch, coalesce_n=0, coalesce_ms=0)

    await _flush_n_times(obs, 5)

    assert len(writes) == 5


@pytest.mark.asyncio
async def test_count_bound_coalesces_then_flushes(tmp_path, monkeypatch):
    obs, writes = _observer(tmp_path, monkeypatch, coalesce_n=3, coalesce_ms=0)

    await _flush_n_times(obs, 1)
    assert len(writes) == 1, "the first flush always lands"

    await _flush_n_times(obs, 2)
    assert len(writes) == 1, "held back until the batch fills"

    await _flush_n_times(obs, 1)
    assert len(writes) == 2, "batch of 3 reached"


@pytest.mark.asyncio
async def test_time_bound_alone_still_coalesces(tmp_path, monkeypatch):
    """N disabled, MS set — the remaining bound must still apply."""
    obs, writes = _observer(tmp_path, monkeypatch, coalesce_n=0, coalesce_ms=60_000)

    await _flush_n_times(obs, 5)

    assert len(writes) == 1


@pytest.mark.asyncio
async def test_force_always_writes(tmp_path, monkeypatch):
    """``on_loop_end`` relies on this regardless of the bounds."""
    obs, writes = _observer(tmp_path, monkeypatch, coalesce_n=1000, coalesce_ms=0)

    obs._flush_json()
    obs._flush_json()
    obs._flush_json(force=True)

    assert len(writes) == 2  # first (always) + forced


@pytest.mark.asyncio
async def test_snapshot_is_readable_after_every_event(tmp_path, monkeypatch):
    """The point of flushing often: a kill -9 must leave a usable file."""
    obs, _writes = _observer(tmp_path, monkeypatch, coalesce_n=0, coalesce_ms=0)
    await obs.on_loop_start(
        LoopConfig(task_id="t", role_id="r", max_turns=5),
    )

    obs._flush_json()

    doc = json.loads((tmp_path / "t.json").read_text())
    assert isinstance(doc, dict)
