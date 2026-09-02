"""Portable task-status pause checks for long-running agent loops."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, cast

from agent_core.types import TaskStatus

logger = logging.getLogger(__name__)

type PauseCheckFn = Callable[[], Awaitable[bool]]
type TaskStatusLoader = Callable[[str], Any | Awaitable[Any]]
_STOP_STATUSES = frozenset({TaskStatus.SUSPENDED, TaskStatus.ABORTED})


def make_task_pause_check(
    task_id: str,
    load_status: TaskStatusLoader,
    *,
    missing_exceptions: tuple[type[BaseException], ...] = (),
) -> PauseCheckFn:
    """Build a safe loop pause hook around a host-supplied status loader.

    ``load_status`` may return a status directly, a MAPPING with a ``status``
    key (the common shape when a host hands back a DB row as a dict), or an
    object with a ``status`` attribute — synchronously or asynchronously.
    Missing synthetic sub-runs are treated as active; other backend failures are
    logged and also fail open.
    """

    async def _check() -> bool:
        try:
            loaded = load_status(task_id)
            value = await loaded if inspect.isawaitable(loaded) else loaded
        except missing_exceptions:
            return False
        except Exception as exc:
            logger.warning("pause_check: load_status(%s) failed: %s", task_id, exc)
            return False
        status = _status_of(value)
        if status is not None and status in _STOP_STATUSES:
            logger.info(
                "pause_check: task %s is %s; pausing the loop", task_id, status,
            )
            return True
        return False

    return _check


def _status_of(value: Any) -> str | None:
    """Extract the status from whatever the host's loader returned.

    The Mapping branch comes FIRST and is not optional: a loader that returns a
    row as ``{"status": "suspended"}`` used to fall through to the bare value,
    which is a dict and therefore never matches a status — so a suspended or
    aborted task kept running indefinitely, silently.

    Returns ``None`` when nothing status-like is present. Narrowing to ``str``
    also keeps the membership test total: ``TaskStatus`` is a ``StrEnum`` (so a
    plain ``"suspended"`` compares equal to the member and needs no conversion),
    while an unhashable leftover — a status-less dict, say — would otherwise
    raise ``TypeError`` from ``in`` and turn a fail-open check into a crash.
    """
    if isinstance(value, Mapping):
        candidate = cast(Mapping[str, Any], value).get("status")
    else:
        candidate = getattr(value, "status", value)
    return candidate if isinstance(candidate, str) else None


def pause_check_from_state(state: Mapping[str, Any] | None) -> PauseCheckFn | None:
    """Read a host-injected pause hook from ``state.metadata``."""
    if not state:
        return None
    metadata_value: object = state.get("metadata")
    if not isinstance(metadata_value, Mapping):
        return None
    metadata = cast(Mapping[str, object], metadata_value)
    pause_check: object = metadata.get("pause_check")
    return cast(PauseCheckFn, pause_check) if callable(pause_check) else None


__all__ = ["PauseCheckFn", "TaskStatusLoader", "make_task_pause_check", "pause_check_from_state"]
