"""Composable, fail-closed tool permission policy."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolPermissionContext:
    """Allowlist and denylist constraints for tool names."""

    allow_names: frozenset[str] | None = None
    deny_names: frozenset[str] = frozenset()
    deny_prefixes: tuple[str, ...] = ()

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

    def filter(self, tool_names: set[str]) -> set[str]:
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
        allow_names: set[str] | list[str] | tuple[str, ...] | None = None,
        deny_names: set[str] | list[str] | tuple[str, ...] = (),
        deny_prefixes: tuple[str, ...] | list[str] = (),
    ) -> ToolPermissionContext:
        normalized_allow = None
        if allow_names is not None:
            normalized_allow = frozenset(name.lower() for name in allow_names)
        return cls(
            allow_names=normalized_allow,
            deny_names=frozenset(name.lower() for name in deny_names),
            deny_prefixes=tuple(prefix.lower() for prefix in deny_prefixes),
        )


def from_config_map(mapping: Any | None) -> ToolPermissionContext:
    """Build a policy from a ``{tool_name: bool}`` configuration map."""
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
    return ToolPermissionContext.from_iterables(
        allow_names=allow or None,
        deny_names=deny,
    )


def from_execution_policy(policy: Any | None) -> ToolPermissionContext:
    """Build a policy from an object exposing execution-policy attributes."""
    if policy is None:
        return ToolPermissionContext()
    return ToolPermissionContext.from_iterables(
        allow_names=getattr(policy, "allow_tools", None),
        deny_names=getattr(policy, "deny_tools", ()),
        deny_prefixes=getattr(policy, "deny_tool_prefixes", ()),
    )


__all__ = ["ToolPermissionContext", "from_config_map", "from_execution_policy"]
