"""Composable, fail-closed tool permission policy."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

logger = logging.getLogger(__name__)


def _normalize_names(names: Iterable[str] | None) -> frozenset[str]:
    return frozenset(name.lower() for name in names or ())


@dataclass(frozen=True)
class ToolPermissionContext:
    """Allowlist and denylist constraints for tool names.

    Names are matched case-insensitively: every constraint is lower-cased on
    construction, so building a context directly is as safe as going through
    :meth:`from_iterables`.
    """

    allow_names: frozenset[str] | None = None
    deny_names: frozenset[str] = frozenset()
    deny_prefixes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Normalizing here (rather than only in the factories) keeps direct
        # construction from silently denying every tool the caller listed with
        # mixed case, and tolerates any iterable being passed in.
        if self.allow_names is not None:
            object.__setattr__(self, "allow_names", _normalize_names(self.allow_names))
        object.__setattr__(self, "deny_names", _normalize_names(self.deny_names))
        object.__setattr__(
            self, "deny_prefixes", tuple(prefix.lower() for prefix in self.deny_prefixes or ())
        )

    def blocks(self, tool_name: str) -> bool:
        lowered = tool_name.lower()
        return lowered in self.deny_names or any(
            lowered.startswith(prefix) for prefix in self.deny_prefixes
        )

    def allows(self, tool_name: str) -> bool:
        lowered = tool_name.lower()
        if self.blocks(lowered):
            return False
        return self.allow_names is None or lowered in self.allow_names

    def filter(self, tool_names: Iterable[str]) -> set[str]:
        return {name for name in tool_names if self.allows(name)}

    def is_empty(self) -> bool:
        return self.allow_names is None and not self.deny_names and not self.deny_prefixes

    def merge(self, other: ToolPermissionContext) -> ToolPermissionContext:
        """Combine two contexts so every restriction remains effective."""
        if self.allow_names is None:
            allow = other.allow_names
        elif other.allow_names is None:
            allow = self.allow_names
        else:
            allow = self.allow_names & other.allow_names
        return ToolPermissionContext(
            allow_names=allow,
            deny_names=self.deny_names | other.deny_names,
            deny_prefixes=tuple(dict.fromkeys((*self.deny_prefixes, *other.deny_prefixes))),
        )

    @classmethod
    def from_iterables(
        cls,
        *,
        allow_names: Iterable[str] | None = None,
        deny_names: Iterable[str] | None = (),
        deny_prefixes: Iterable[str] | None = (),
    ) -> ToolPermissionContext:
        return cls(
            allow_names=None if allow_names is None else _normalize_names(allow_names),
            deny_names=_normalize_names(deny_names),
            deny_prefixes=tuple(prefix.lower() for prefix in deny_prefixes or ()),
        )


def from_config_map(mapping: Any | None) -> ToolPermissionContext:
    """Build a policy from a ``{tool_name: bool}`` configuration map.

    The two boolean values are **not** symmetric toggles:

    * ``False`` adds the tool to the denylist and leaves every unlisted tool
      alone.
    * ``True`` adds the tool to the *allowlist* — and because an allowlist is
      exhaustive by definition, a single ``True`` entry anywhere in the map
      denies every tool the map does not name.

    So ``{"bash": True}`` means "bash and nothing else", not "additionally
    enable bash". This is the fail-closed reading: to enable one tool without
    revoking the rest, list the tools to remove with ``False`` instead.
    """
    if not isinstance(mapping, dict) or not mapping:
        return ToolPermissionContext()

    allow: set[str] = set()
    deny: set[str] = set()
    entries = cast("dict[object, object]", mapping)
    for name, enabled in entries.items():
        if not isinstance(name, str):
            continue
        if enabled is True:
            allow.add(name)
        elif enabled is False:
            deny.add(name)
        else:
            logger.warning(
                "Tool policy entry %r=%r ignored: value must be a bare bool "
                "(true/false), got %s — the tool is left UNCHANGED.",
                name,
                enabled,
                type(enabled).__name__,
            )
    if allow:
        logger.debug(
            "Tool policy map enabled %d tool(s) explicitly; this is an allowlist, "
            "so every unlisted tool is now denied.",
            len(allow),
        )
    return ToolPermissionContext.from_iterables(
        allow_names=allow or None,
        deny_names=deny,
    )


def from_execution_policy(policy: Any | None) -> ToolPermissionContext:
    """Build a policy from an object exposing execution-policy attributes.

    Attributes that are absent *or* ``None`` fall back to "unconstrained",
    since ``allow_tools: tuple[str, ...] | None = None`` is a common shape.
    """
    if policy is None:
        return ToolPermissionContext()
    return ToolPermissionContext.from_iterables(
        allow_names=getattr(policy, "allow_tools", None),
        deny_names=getattr(policy, "deny_tools", None) or (),
        deny_prefixes=getattr(policy, "deny_tool_prefixes", None) or (),
    )


__all__ = ["ToolPermissionContext", "from_config_map", "from_execution_policy"]
