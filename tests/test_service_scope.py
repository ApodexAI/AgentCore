"""ServiceScope — per-run isolation of the service registry.

Covers the two things M1 must guarantee: (1) with no scope active the
registry behaves exactly as before, so the ~108 existing
``services.get`` call sites need no edit; (2) two runs in one process
hold independent service tables, which is what lets a Solver and an
independent Verifier own different ``ResourceManager`` instances.
"""

from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable

import pytest

from agent_core.errors import ServiceNotRegistered
from agent_core.runtime.registries import services
from agent_core.runtime.registries.scope import (
    ServiceScope,
    current_scope,
    use_scope,
)


class Alpha:
    def __init__(self, tag: str = "") -> None:
        self.tag = tag


class Beta:
    pass


@runtime_checkable
class Capability(Protocol):
    def do(self) -> str: ...


class CapImpl:
    def __init__(self, tag: str = "") -> None:
        self.tag = tag

    def do(self) -> str:
        return self.tag


@pytest.fixture(autouse=True)
def _isolate_process_table():
    """Keep the process-global table untouched across tests."""
    saved = dict(services._services)
    services._services.clear()
    yield
    services._services.clear()
    services._services.update(saved)


# ── (1) no scope active → unchanged behaviour ────────────────────────


def test_no_scope_register_and_get():
    a = Alpha("proc")
    services.register(Alpha, a)
    assert services.get(Alpha) is a
    assert services.get_optional(Alpha) is a
    assert services.is_registered(Alpha)
    assert current_scope() is None


def test_no_scope_missing_raises():
    with pytest.raises(ServiceNotRegistered):
        services.get(Alpha)
    assert services.get_optional(Alpha) is None
    assert not services.is_registered(Beta)


def test_get_does_not_scan_structurally_but_get_optional_does():
    """Preserves the pre-existing asymmetry between the two lookups."""
    impl = CapImpl("proc")
    services.register(CapImpl, impl)
    # ``get`` is exact-key only.
    with pytest.raises(ServiceNotRegistered):
        services.get(Capability)
    # ``get_optional`` resolves the Protocol structurally.
    assert services.get_optional(Capability) is impl


def test_no_scope_snapshot_restore_and_clear():
    services.register(Alpha, Alpha("one"))
    snap = services.snapshot()
    services.register(Beta, Beta())
    services.restore(snap)
    assert services.is_registered(Alpha)
    assert not services.is_registered(Beta)
    services.clear()
    assert not services.is_registered(Alpha)


# ── (2) scoped isolation ─────────────────────────────────────────────


def test_scope_register_does_not_leak_to_process_table():
    with use_scope(name="run") as scope:
        services.register(Alpha, Alpha("scoped"))
        assert services.get(Alpha).tag == "scoped"
        assert Alpha in scope.services
    assert Alpha not in services._services
    assert services.get_optional(Alpha) is None


def test_two_scopes_hold_different_instances_for_the_same_key():
    with use_scope(name="solver"):
        services.register(Alpha, Alpha("solver"))
        solver_seen = services.get(Alpha)
    with use_scope(name="verifier"):
        services.register(Alpha, Alpha("verifier"))
        verifier_seen = services.get(Alpha)
    assert solver_seen.tag == "solver"
    assert verifier_seen.tag == "verifier"
    assert solver_seen is not verifier_seen


def test_scope_falls_back_to_process_table_by_default():
    shared = Beta()
    services.register(Beta, shared)
    with use_scope():
        services.register(Alpha, Alpha("scoped"))
        assert services.get(Beta) is shared  # inherited
        assert services.get(Alpha).tag == "scoped"  # own
        assert services.is_registered(Beta)


def test_sealed_scope_does_not_inherit():
    services.register(Beta, Beta())
    with use_scope(fallback_to_process=False):
        assert services.get_optional(Beta) is None
        assert not services.is_registered(Beta)
        with pytest.raises(ServiceNotRegistered):
            services.get(Beta)


def test_scope_shadows_process_entry_without_mutating_it():
    proc = Alpha("proc")
    services.register(Alpha, proc)
    with use_scope():
        services.register(Alpha, Alpha("scoped"))
        assert services.get(Alpha).tag == "scoped"
    assert services.get(Alpha) is proc


def test_scope_clear_leaves_process_table_intact():
    services.register(Beta, Beta())
    with use_scope():
        services.register(Alpha, Alpha("scoped"))
        services.clear()
        assert services.get_optional(Alpha) is None
        assert services.is_registered(Beta)  # process entry survives
    assert services.is_registered(Beta)


def test_nested_scopes_restore_outer_on_exit():
    with use_scope(name="outer") as outer:
        services.register(Alpha, Alpha("outer"))
        with use_scope(name="inner"):
            services.register(Alpha, Alpha("inner"))
            assert services.get(Alpha).tag == "inner"
            assert current_scope().name == "inner"
        assert current_scope() is outer
        assert services.get(Alpha).tag == "outer"
    assert current_scope() is None


def test_structural_lookup_prefers_scope_over_process():
    services.register(CapImpl, CapImpl("proc"))
    with use_scope():
        services.register(Alpha, CapImpl("scoped"))  # different key, same Protocol
        assert services.get_optional(Capability).tag == "scoped"


def test_scope_structural_lookup_beats_process_exact_protocol_key():
    services.register(Capability, CapImpl("proc"))
    with use_scope():
        services.register(Alpha, CapImpl("scoped"))
        assert services.get_optional(Capability).tag == "scoped"


def test_local_lookup_does_not_fall_back_to_process_table():
    services.register(Alpha, Alpha("proc"))
    with use_scope():
        assert services.is_registered(Alpha)
        assert not services.is_registered_local(Alpha)
        with pytest.raises(ServiceNotRegistered):
            services.get_local(Alpha)

        scoped = Alpha("scoped")
        services.register(Alpha, scoped)
        assert services.is_registered_local(Alpha)
        assert services.get_local(Alpha) is scoped


def test_explicit_scope_object_can_be_reused():
    scope = ServiceScope(name="reused")
    with use_scope(scope):
        services.register(Alpha, Alpha("first"))
    with use_scope(scope):
        assert services.get(Alpha).tag == "first"


# ── (3) concurrency: the property D2 depends on ──────────────────────


def test_concurrent_rollouts_do_not_overwrite_each_other():
    """Two rollouts in one event loop keep independent tables.

    Relies on ``asyncio.gather`` giving each coroutine its own Context
    copy — the same mechanism ``execution_context`` uses for
    ``_CURRENT_TOOL_CALL_ID``.
    """
    order: list[str] = []

    async def rollout(tag: str, first_delay: float) -> str:
        with use_scope(name=tag):
            services.register(Alpha, Alpha(tag))
            await asyncio.sleep(first_delay)  # let the sibling interleave
            order.append(tag)
            services.register(Beta, Beta())  # a second write, post-yield
            await asyncio.sleep(0)
            return services.get(Alpha).tag  # must still be its own

    async def main() -> list[str]:
        return await asyncio.gather(
            rollout("a", 0.02),
            rollout("b", 0.001),
        )

    seen = asyncio.run(main())
    assert seen == ["a", "b"]
    assert order == ["b", "a"]  # they really did interleave
    assert Alpha not in services._services
    assert Beta not in services._services


def test_child_tasks_inherit_the_parent_scope():
    async def child() -> str:
        return services.get(Alpha).tag

    async def main() -> str:
        with use_scope(name="parent"):
            services.register(Alpha, Alpha("parent"))
            return await asyncio.create_task(child())

    assert asyncio.run(main()) == "parent"
