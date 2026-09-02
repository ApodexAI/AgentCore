"""``make_task_pause_check`` must recognise every shape a host loader returns.

The loader is host-injected (AgentCore ships no task store), so the shape is not
under this repo's control: a status enum, a plain string, an ORM row with a
``.status`` attribute, or — very commonly — a DB row handed back as a dict. The
attribute-only lookup silently missed the mapping case: ``getattr(row, "status",
row)`` returned the dict itself, which never matches a status, so a suspended or
aborted task kept running indefinitely with nothing in the logs.
"""

from __future__ import annotations

import logging

import pytest

from agent_core.runtime.pause_check import make_task_pause_check
from agent_core.types import TaskStatus


class _Row:
    def __init__(self, status: object) -> None:
        self.status = status


@pytest.mark.parametrize(
    "value",
    [
        TaskStatus.SUSPENDED,
        "suspended",
        _Row(TaskStatus.SUSPENDED),
        _Row("suspended"),
        {"status": TaskStatus.SUSPENDED},
        {"status": "suspended"},
        {"status": "aborted"},
    ],
    ids=["enum", "str", "row-enum", "row-str", "map-enum", "map-str", "map-aborted"],
)
async def test_stop_statuses_pause_across_every_loader_shape(value):
    check = make_task_pause_check("t1", lambda _tid: value)
    assert await check() is True


@pytest.mark.parametrize(
    "value",
    [
        TaskStatus.RUNNING,
        "running",
        _Row("running"),
        {"status": "running"},
        {"status": "completed"},
        # No status anywhere — fail open rather than wedge the loop.
        {"id": "t1"},
        object(),
        None,
    ],
    ids=["enum", "str", "row", "map", "map-done", "map-no-status", "opaque", "none"],
)
async def test_active_and_unknown_shapes_do_not_pause(value):
    check = make_task_pause_check("t1", lambda _tid: value)
    assert await check() is False


async def test_mapping_pause_is_logged(caplog):
    """The old silent miss had no log line to point at the shape mismatch."""
    check = make_task_pause_check("t1", lambda _tid: {"status": "suspended"})
    with caplog.at_level(logging.INFO):
        assert await check() is True
    assert "pausing the loop" in caplog.text


async def test_async_mapping_loader_is_awaited():
    async def _load(_tid):
        return {"status": "aborted"}

    assert await make_task_pause_check("t1", _load)() is True


async def test_missing_task_exception_fails_open():
    class _Missing(Exception):
        pass

    def _load(_tid):
        raise _Missing

    check = make_task_pause_check(
        "t1", _load, missing_exceptions=(_Missing,),
    )
    assert await check() is False


async def test_backend_failure_fails_open_and_warns(caplog):
    def _load(_tid):
        raise RuntimeError("db down")

    with caplog.at_level(logging.WARNING):
        assert await make_task_pause_check("t1", _load)() is False
    assert "load_status" in caplog.text
