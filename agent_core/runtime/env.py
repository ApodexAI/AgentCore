"""Shared env-variable prefix cascade.

Both ``core/runtime/loop/_runaway.py`` and this package's own
``session_context.py`` independently re-walked the same
``AGENT_CORE_ / MIROHARNESS_ / FRONTIER_AGENT_`` prefix order looking for a
configured value — one copy per module, easy to drift if a prefix is ever
added or reordered in only one of them. This is the single implementation
both converge on.
"""

from __future__ import annotations

import os

# The portable spelling wins when multiple aliases are configured, followed
# by the AgentCore and FrontierAgent compatibility names. Order matters:
# callers rely on the first configured prefix winning.
ENV_PREFIXES = ("AGENT_CORE_", "MIROHARNESS_", "FRONTIER_AGENT_")


def first_configured(suffix: str, prefixes: tuple[str, ...] = ENV_PREFIXES) -> tuple[str, str] | None:
    """Return the ``(name, value)`` of the first non-empty ``{prefix}{suffix}`` env var.

    ``None`` when none of the prefixed names are set (or all are blank).
    """
    for prefix in prefixes:
        name = f"{prefix}{suffix}"
        raw = os.environ.get(name, "").strip()
        if raw:
            return name, raw
    return None


__all__ = ["ENV_PREFIXES", "first_configured"]
