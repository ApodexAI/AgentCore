# pyright: reportUnknownVariableType=false, reportUnknownParameterType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportMissingTypeArgument=false, reportUnnecessaryIsInstance=false, reportUnusedFunction=false, reportAttributeAccessIssue=false, reportUnnecessaryComparison=false
"""Skill state configuration — persists enable/disable state for skills.

Simplified from MiroOS's MCP extensions config — only skill state management,
no MCP server configuration.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr

from agent_core.runtime.env import first_configured

logger = logging.getLogger(__name__)

_CONFIG_FILENAMES = ["extensions_config.json", "mcp_config.json"]
# Resolved through the shared prefix cascade (``AGENT_CORE_`` wins, then the
# MiroHarness / FrontierAgent compatibility spellings) rather than the single
# ``MIROHARNESS_``-prefixed name this module used to hardcode.
_ENV_SUFFIX = "EXTENSIONS_CONFIG_PATH"


def _find_config_file() -> Path | None:
    """Search for extensions config in standard locations."""
    configured = first_configured(_ENV_SUFFIX)
    if configured:
        _, env_path = configured
        p = Path(env_path)
        if p.is_file():
            return p

    for directory in [Path.cwd(), Path.cwd().parent]:
        for name in _CONFIG_FILENAMES:
            p = directory / name
            if p.is_file():
                return p

    return None


class SkillStateConfig(BaseModel):
    """Enable/disable state for a skill."""

    enabled: bool = True


class ExtensionsConfig(BaseModel):
    """Skill state configuration (loaded from extensions_config.json)."""

    skills: dict[str, SkillStateConfig] = Field(default_factory=dict)
    _file_path: Path | None = PrivateAttr(default=None)
    # Digest of the bytes this config was parsed from -- see ``has_changed``.
    _file_digest: str = PrivateAttr(default="")

    model_config = {"populate_by_name": True}

    @classmethod
    def from_file(cls, config_path: str | Path | None = None) -> ExtensionsConfig:
        """Load config from JSON file with environment variable resolution."""
        if config_path:
            resolved = Path(config_path)
            # Keep the source stable across later ``chdir`` calls.  Do not use
            # ``resolve()`` here: retaining a configured symlink path lets an
            # operator atomically repoint that symlink and have reload observe
            # the new target.
            if not resolved.is_absolute():
                resolved = Path.cwd() / resolved
        else:
            resolved = _find_config_file()

        if resolved is None or not resolved.is_file():
            logger.debug("No extensions config found — using empty defaults")
            return cls()

        try:
            raw = resolved.read_bytes()
            data = json.loads(raw.decode("utf-8"))
            _resolve_env_variables(data)
            logger.info("Loaded extensions config from %s", resolved)
            instance = cls.model_validate(data)
            instance._file_path = resolved
            # Digest the exact bytes that were parsed, so the stored fingerprint
            # and the loaded state can never describe different file contents.
            instance._file_digest = _digest(raw)
            return instance
        except Exception as e:
            logger.warning("Failed to load extensions config %s: %s", resolved, e)
            return cls()

    @property
    def source_path(self) -> Path | None:
        """File this config was loaded from; ``None`` for in-memory defaults.

        A reloader must pass this back to :meth:`from_file` — calling it with
        no argument restarts the cwd/env search, which silently returns empty
        defaults for a config that was loaded from an explicit path.
        """
        return self._file_path

    def has_changed(self) -> bool:
        """Return True if the backing file's contents differ from what we hold.

        Compares a digest of the bytes, not the modification time. Two reasons,
        both of which bit this code:

        Timestamps are far coarser than the edits they are meant to order. The
        filesystem clock here advances in 1 ms steps, and two consecutive writes
        land on an identical mtime about 92% of the time -- so a change made
        within a millisecond of the load was simply invisible, and an operator
        toggling a skill got the old state until something else touched the
        file. That was reaching the test suite as an intermittent failure whose
        rate tracked how fast the machine happened to be running.

        A strict ``>`` also cannot see a file whose timestamp moves BACKWARD,
        which is the normal outcome of restoring a backup, a ``git checkout``,
        or an ``rsync --times`` of an older revision. The content changed; the
        config went on reporting that it had not.

        The file is a small JSON document and this is called from
        ``get_enabled_skills``, which its callers cache -- reading it is cheaper
        than being wrong about it. An identical rewrite correctly reports no
        change, since nothing needs reloading.
        """
        if self._file_path is None or not self._file_path.is_file():
            return False
        try:
            return _digest(self._file_path.read_bytes()) != self._file_digest
        except OSError:
            return False

    def is_skill_enabled(self, skill_name: str) -> bool:
        """Check if a skill is enabled (default: True if not listed)."""
        state = self.skills.get(skill_name)
        return state.enabled if state else True


def _digest(raw: bytes) -> str:
    """Content fingerprint. Not a security boundary -- just change detection."""
    return hashlib.blake2b(raw, digest_size=16).hexdigest()


def _resolve_env_variables(obj: Any) -> Any:
    """Recursively replace $VAR_NAME with environment variable values."""
    if isinstance(obj, str) and obj.startswith("$"):
        var_name = obj[1:]
        value = os.getenv(var_name, "")
        if not value:
            logger.debug("Env var %s not set, using empty string", var_name)
        return value
    elif isinstance(obj, dict):
        for key in obj:
            obj[key] = _resolve_env_variables(obj[key])
        return obj
    elif isinstance(obj, list):
        return [_resolve_env_variables(item) for item in obj]
    return obj
