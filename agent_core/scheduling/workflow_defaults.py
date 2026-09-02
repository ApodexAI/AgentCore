"""Product-neutral workflow default registry."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

_WORKFLOW_DEFAULTS: dict[str, dict[str, Any]] = {}


def register_workflow_defaults(
    pipeline_ids: str | tuple[str, ...],
    defaults: Mapping[str, Any],
) -> None:
    """Register an immutable snapshot of product-owned workflow defaults."""
    ids = (pipeline_ids,) if isinstance(pipeline_ids, str) else pipeline_ids
    snapshot = dict(defaults)
    for pipeline_id in ids:
        _WORKFLOW_DEFAULTS[pipeline_id] = snapshot


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
