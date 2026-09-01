"""Scoped service tables — per-run isolation for the service registry.

``services.py`` keeps one process-global table. That is correct for a
single-run CLI process, but it makes two concurrent runs in one process
share every service: registering a ``ResourceManager`` for run B
overwrites run A's. The same defect has already been patched twice in
narrower places — ``TaskContextStore``
(``models/task_context.py``) fixed it for ``TaskContext``, and
``sdk_bootstrap``'s ``api_owned`` check guards ``ResourceManager``
against a competing assembler — and is still open in
``get_or_bootstrap_sdk_runtime``, which rebuilds and re-registers
``ResourceManager`` on every call.

A :class:`ServiceScope` is that table as an object. Activate one with
:func:`use_scope` and every ``services.register`` inside the block writes
to it instead of the process table; lookups read it first and fall back
to the process table by default. Code that never activates a scope keeps
its exact previous behaviour, which is why the ~108 existing
``services.get`` call sites need no edit: none of them cache the result
in a module-level variable, so they all resolve through the active scope.

Isolation rides on ``contextvars``: ``asyncio.gather`` / ``create_task``
copy the current Context, so a task tree started inside a scope inherits
it while a sibling rollout in its own scope stays unaffected. This is the
mechanism ``core/execution_context.py`` already relies on for
``_CURRENT_TOOL_CALL_ID``.

Scope objects hold a plain ``dict`` rather than another contextvar: their
lifetime is one run, created and discarded explicitly.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any


class ServiceScope:
    """An isolated service table, activated via :func:`use_scope`.

    ``fallback_to_process`` keeps process-wide singletons (skill loaders,
    middleware chains, anything a bootstrap registered before the scope
    opened) visible to scoped code. Set it False to assert a run is fully
    self-contained — useful in tests and for a Verifier that must not
    silently inherit the Solver's ``ResourceManager``.
    """

    __slots__ = ("fallback_to_process", "name", "services")

    def __init__(
        self,
        *,
        name: str = "",
        fallback_to_process: bool = True,
        initial: dict[type, Any] | None = None,
    ) -> None:
        self.services: dict[type, Any] = dict(initial or {})
        self.fallback_to_process = fallback_to_process
        self.name = name

    def __repr__(self) -> str:
        return (
            f"ServiceScope(name={self.name!r}, entries={len(self.services)}, "
            f"fallback_to_process={self.fallback_to_process})"
        )


_CURRENT_SERVICE_SCOPE: ContextVar[ServiceScope | None] = ContextVar(
    "miroharness_service_scope", default=None
)


def current_scope() -> ServiceScope | None:
    """The scope active in this async context, or ``None``."""
    return _CURRENT_SERVICE_SCOPE.get()


def set_current_scope(scope: ServiceScope | None) -> Token[ServiceScope | None]:
    """Activate ``scope``; pass the returned token to :func:`reset_current_scope`."""
    return _CURRENT_SERVICE_SCOPE.set(scope)


def reset_current_scope(token: Token[ServiceScope | None]) -> None:
    """Restore the prior scope. Nesting-safe via the token."""
    _CURRENT_SERVICE_SCOPE.reset(token)


@contextmanager
def use_scope(
    scope: ServiceScope | None = None,
    *,
    name: str = "",
    fallback_to_process: bool = True,
) -> Generator[ServiceScope, None, None]:
    """Run a block against an isolated service table.

    Usable inside ``async def`` — a plain ``with`` is enough, because the
    contextvar set/reset pair happens in one task and child tasks inherit
    a copy of the Context::

        with use_scope(name="solver") as solver_scope:
            bootstrap_kernel_from_assembly(llm=solver_llm)
            await run_solver()          # sees solver_scope
    """
    active = (
        scope
        if scope is not None
        else ServiceScope(
            name=name,
            fallback_to_process=fallback_to_process,
        )
    )
    token = set_current_scope(active)
    try:
        yield active
    finally:
        reset_current_scope(token)
