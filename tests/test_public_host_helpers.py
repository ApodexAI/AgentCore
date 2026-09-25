"""Helpers host facades re-export must be reachable through the public API.

Hosts alias AgentCore modules with ``from agent_core.X import *``; a name left
out of ``__all__`` (or kept private) is invisible to static checkers there.
"""

from __future__ import annotations

import agent_core.loop_types as loop_types
import agent_core.runtime.loop.model_profile as model_profile


def test_wall_deadline_remaining_s_is_exported() -> None:
    assert "wall_deadline_remaining_s" in loop_types.__all__


def test_to_openai_tool_calls_is_public_and_keeps_its_old_name() -> None:
    namespace: dict[str, object] = {}
    exec("from agent_core.runtime.loop.model_profile import *", namespace)

    assert namespace["to_openai_tool_calls"] is model_profile.to_openai_tool_calls
    assert model_profile._to_openai_tool_calls is model_profile.to_openai_tool_calls
