"""Service Registry — simplified dependency injection container.

All kernel services register themselves here. Other layers resolve
dependencies through the registry instead of importing singletons.

Two tables back these functions. The process table (``_services``) is the
historical one. A :class:`~agent_core.runtime.registries.scope.ServiceScope`
activated via ``scope.use_scope()`` shadows it for the current async
context, so two runs in one process can hold different
``ResourceManager`` / ``AgentRegistry`` instances instead of overwriting
each other. Reads consult the active scope first and fall back to the
process table (unless the scope sets ``fallback_to_process=False``);
writes go to whichever table is active. With no scope active every
function behaves exactly as it did before, so callers and composition
roots need no change.
"""

from __future__ import annotations

from typing import Any

from agent_core.errors import ServiceNotRegistered
from agent_core.runtime.registries.scope import ServiceScope, current_scope

_services: dict[type, Any] = {}


def _write_table() -> dict[type, Any]:
    """The table mutations apply to: the active scope's, else the process one."""
    scope = current_scope()
    return _services if scope is None else scope.services


def _read_tables() -> tuple[dict[type, Any], ...]:
    """Tables to consult, in precedence order."""
    scope: ServiceScope | None = current_scope()
    if scope is None:
        return (_services,)
    if not scope.fallback_to_process:
        return (scope.services,)
    return (scope.services, _services)


def _structural_lookup[T](
    table: dict[type, Any],
    service_type: type[T],
) -> T | None:
    """Runtime-checkable Protocol scan over one table."""
    for candidate in table.values():
        try:
            if isinstance(candidate, service_type):
                return candidate
        except TypeError:
            # ``service_type`` is not runtime-checkable (or not a class).
            break
    return None


def register[T](service_type: type[T], instance: T) -> None:
    """Register a service instance by its type."""
    _write_table()[service_type] = instance


def get[T](service_type: type[T]) -> T:
    """Retrieve a registered service. Raises ServiceNotRegistered if missing.

    Exact-key lookup only — no Protocol scan. ``get_optional`` is the
    entry point that also resolves structurally.
    """
    for table in _read_tables():
        instance = table.get(service_type)
        if instance is not None:
            return instance
    raise ServiceNotRegistered(service_type)


def get_optional[T](service_type: type[T]) -> T | None:
    """Retrieve a service or None if not registered."""
    # Resolve each table completely before falling back to the next one. A
    # scope-local structural Protocol implementation must shadow an exact-key
    # process registration just as a scope-local exact key does.
    for table in _read_tables():
        instance = table.get(service_type)
        if instance is not None:
            return instance
        # Protocol keys such as ``EventSink`` / ``PhaseMiddlewareChain`` are
        # intentionally registered structurally by some composition roots
        # during the migration. Scan before consulting a lower-precedence
        # table so fallback never defeats scope shadowing.
        found = _structural_lookup(table, service_type)
        if found is not None:
            return found
    return None


def get_optional_by_type_name(type_name: str) -> Any | None:
    """Return the first service registered under a class with ``type_name``.

    Compatibility helper for migration shims: callers can support an old
    concrete registration key without importing that concrete class across
    layer boundaries.
    """
    for table in _read_tables():
        for registered_type, instance in table.items():
            if getattr(registered_type, "__name__", "") == type_name:
                return instance
    return None


def clear() -> None:
    """Clear all registrations in the active table (useful for testing)."""
    _write_table().clear()


def is_registered(service_type: type) -> bool:
    return any(service_type in table for table in _read_tables())


def get_local[T](service_type: type[T]) -> T:
    """Retrieve an exact-key service from the active write table only.

    Unlike :func:`get`, this never falls back from a scope to the process
    table. Composition roots use it when deciding whether the current scope
    already owns a service or still needs an isolated instance.
    """
    instance = _write_table().get(service_type)
    if instance is None:
        raise ServiceNotRegistered(service_type)
    return instance


def is_registered_local(service_type: type) -> bool:
    """Whether ``service_type`` exists in the active write table itself."""
    return service_type in _write_table()


def snapshot() -> dict[type, Any]:
    """Return a shallow copy of the active registration map.

    Pair with ``restore`` to scope service registrations inside a
    ``with``/``async with`` block — used by ``BenchmarkSession`` so test
    runs do not leak service instances across sessions.
    """
    return dict(_write_table())


def restore(snapshot_map: dict[type, Any]) -> None:
    """Replace the active registration map with ``snapshot_map``.

    Counterpart to :func:`snapshot`; ownership of ``snapshot_map`` is
    not retained — the registry stores its own copy.
    """
    table = _write_table()
    table.clear()
    table.update(snapshot_map)
