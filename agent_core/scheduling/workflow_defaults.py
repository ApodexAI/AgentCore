"""Product-neutral workflow default registry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_WORKFLOW_DEFAULTS: dict[str, dict[str, Any]] = {}


def register_workflow_defaults(
    pipeline_ids: str | tuple[str, ...],
    defaults: Mapping[str, Any],
    *,
    merge: bool = True,
    include_none: bool = False,
) -> None:
    """Register product defaults without accidentally blanking prior values.

    Registrations merge by default because hosts commonly contribute defaults
    from several composition modules. ``None`` means "no product default" and
    is ignored unless ``include_none`` is explicitly requested.
    """
    ids = (pipeline_ids,) if isinstance(pipeline_ids, str) else pipeline_ids
    snapshot = {
        key: value
        for key, value in defaults.items()
        if include_none or value is not None
    }
    for pipeline_id in ids:
        current = _WORKFLOW_DEFAULTS.get(pipeline_id, {}) if merge else {}
        _WORKFLOW_DEFAULTS[pipeline_id] = {**current, **snapshot}


def clear_workflow_defaults() -> None:
    """Clear registrations, primarily for isolated tests and app teardown."""
    _WORKFLOW_DEFAULTS.clear()


def get_workflow_default(pipeline_id: str | None, attr: str) -> Any:
    """Return one registered product default, or ``None`` when absent."""
    if not pipeline_id:
        return None
    return _WORKFLOW_DEFAULTS.get(pipeline_id, {}).get(attr)


__all__ = [
    "clear_workflow_defaults",
    "get_workflow_default",
    "register_workflow_defaults",
]
