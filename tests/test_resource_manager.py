"""Tests for the ResourceManager kernel module."""
from unittest.mock import MagicMock

import pytest

from agent_core.errors import PermissionDenied
from agent_core.runtime.registries import services as service_registry
from agent_core.runtime.registries.agents import AgentDefinition, AgentRegistry
from agent_core.runtime.resources.manager import ResourceManager
from agent_core.runtime.tool_permission import ToolPermissionContext


@pytest.fixture
def resource_mgr_with_registry():
    """ResourceManager backed by a real AgentRegistry with known roles."""
    service_registry.clear()

    # Build a minimal AgentRegistry with the 3 canonical roles
    agent_reg = AgentRegistry()
    agent_reg.register(AgentDefinition(
        role_id="researcher", display_name="Researcher",
        system_prompt="test", allowed_tools=["web_search", "web_fetch", "bash"],
        color="#3b82f6", icon="search",
    ))
    agent_reg.register(AgentDefinition(
        role_id="summarizer", display_name="Summarizer",
        system_prompt="test", allowed_tools=["read_file"],
        color="#a855f7", icon="file",
    ))

    # Register in service_registry so ResourceManager._get_allowed_tools can find it
    service_registry.register(AgentRegistry, agent_reg)

    llm = MagicMock()
    tools = {
        "web_search": MagicMock(name="web_search"),
        "web_fetch": MagicMock(name="web_fetch"),
        "bash": MagicMock(name="bash"),
        "read_file": MagicMock(name="read_file"),
    }
    for name, tool in tools.items():
        tool.name = name
    mgr = ResourceManager(llm=llm, tools=tools)
    yield mgr
    service_registry.clear()


def test_researcher_has_all_tools(resource_mgr_with_registry):
    tools = resource_mgr_with_registry.get_tools_for_role("researcher")
    assert len(tools) == 3


def test_summarizer_only_has_read_file(resource_mgr_with_registry):
    tools = resource_mgr_with_registry.get_tools_for_role("summarizer")
    assert len(tools) == 1


def test_permission_check(resource_mgr_with_registry):
    assert resource_mgr_with_registry.check_permission("researcher", "web_search")
    assert not resource_mgr_with_registry.check_permission("summarizer", "web_search")
    assert not resource_mgr_with_registry.check_permission("unknown-role", "web_search")


def test_require_permission_raises(resource_mgr_with_registry):
    with pytest.raises(PermissionDenied):
        resource_mgr_with_registry.require_permission("summarizer", "web_search")


def test_unknown_role_has_no_tools(resource_mgr_with_registry):
    assert resource_mgr_with_registry.get_tools_for_role("unknown-role") == []
    assert resource_mgr_with_registry.get_tool_names_for_role("unknown-role") == []
    assert resource_mgr_with_registry.get_tool_for_role("unknown-role", "web_search") is None


def test_permission_context_filters_tools(resource_mgr_with_registry):
    tools = resource_mgr_with_registry.get_tools_for_role(
        "researcher",
        permission_context=ToolPermissionContext.from_iterables(
            allow_names={"bash"},
        ),
    )
    assert [t.name for t in tools] == ["bash"]


def test_get_tool_for_role_honors_permission_context(resource_mgr_with_registry):
    tool = resource_mgr_with_registry.get_tool_for_role(
        "researcher",
        "bash",
        permission_context=ToolPermissionContext.from_iterables(
            allow_names={"bash"},
        ),
    )
    assert tool is not None
    denied = resource_mgr_with_registry.get_tool_for_role(
        "researcher",
        "web_search",
        permission_context=ToolPermissionContext.from_iterables(
            allow_names={"bash"},
        ),
    )
    assert denied is None


def test_registry_unavailable_is_fail_closed():
    """Fail-closed: no AgentRegistry → no tools allowed for any role."""
    service_registry.clear()
    llm = MagicMock()
    tools = {
        "web_search": MagicMock(name="web_search"),
        "web_fetch": MagicMock(name="web_fetch"),
    }
    mgr = ResourceManager(llm=llm, tools=tools)

    assert len(mgr.get_tools_for_role("any-role")) == 0
    assert not mgr.check_permission("any-role", "web_search")
    with pytest.raises(PermissionDenied):
        mgr.require_permission("any-role", "web_search")


def test_no_permission_context_returns_all_role_tools(resource_mgr_with_registry):
    tools = resource_mgr_with_registry.get_tools_for_role("researcher")
    assert {t.name for t in tools} == {"web_search", "web_fetch", "bash"}


def test_set_role_llm_overrides_default_for_subsequent_get(resource_mgr_with_registry):
    """``set_role_llm`` lets a workflow node register a per-task LLM
    (e.g. swarm main_agent loading qwen35_397b) so peer nodes pick it
    up via ``get_llm(role_id)`` instead of falling back to the
    ResourceManager's default LLM."""
    profile_llm = MagicMock(name="qwen-397b")
    resource_mgr_with_registry.set_role_llm("swarm_main", profile_llm)
    assert resource_mgr_with_registry.get_llm("swarm_main") is profile_llm


def test_set_role_llm_does_not_leak_into_other_roles(resource_mgr_with_registry):
    """Registering one role's LLM doesn't change resolution for other
    roles — they still hit the default LLM."""
    profile_llm = MagicMock(name="qwen-397b")
    resource_mgr_with_registry.set_role_llm("swarm_main", profile_llm)
    other = resource_mgr_with_registry.get_llm("researcher")
    assert other is not profile_llm
    assert other is resource_mgr_with_registry.llm


def test_set_role_llm_idempotent_last_writer_wins(resource_mgr_with_registry):
    """Repeated registration (e.g. heavy_mode K parallel main_agents
    all loading the same profile) is harmless: the latest registration
    wins, no exceptions raised."""
    first = MagicMock(name="qwen-397b-v1")
    second = MagicMock(name="qwen-397b-v2")
    resource_mgr_with_registry.set_role_llm("swarm_main", first)
    resource_mgr_with_registry.set_role_llm("swarm_main", second)
    assert resource_mgr_with_registry.get_llm("swarm_main") is second


# ── Global tool policy (config injection) ─────────────────────────────────

def test_global_deny_removes_tool_for_all_roles(resource_mgr_with_registry):
    """``{web_search: false}`` config → web_search disappears from every
    role that would otherwise have it."""
    from agent_core.runtime.tool_permission import (
        from_config_map,
    )

    resource_mgr_with_registry.set_global_tool_policy(
        from_config_map({"web_search": False}),
    )
    names = {t.name for t in resource_mgr_with_registry.get_tools_for_role("researcher")}
    assert names == {"web_fetch", "bash"}
    assert not resource_mgr_with_registry.check_permission("researcher", "web_search")


def test_global_allowlist_intersects_role_tools(resource_mgr_with_registry):
    """``{bash: true}`` config → only bash survives for any role."""
    from agent_core.runtime.tool_permission import (
        from_config_map,
    )

    resource_mgr_with_registry.set_global_tool_policy(
        from_config_map({"bash": True}),
    )
    assert [t.name for t in resource_mgr_with_registry.get_tools_for_role("researcher")] == ["bash"]
    # A role without bash ends up with nothing.
    assert resource_mgr_with_registry.get_tools_for_role("summarizer") == []


def test_global_policy_layers_under_per_call_context(resource_mgr_with_registry):
    """A global deny and a per-call allowlist both apply — deny wins."""
    from agent_core.runtime.tool_permission import (
        from_config_map,
    )

    resource_mgr_with_registry.set_global_tool_policy(
        from_config_map({"web_search": False}),
    )
    # Per-call asks for web_search + bash; global deny strips web_search.
    tools = resource_mgr_with_registry.get_tools_for_role(
        "researcher",
        permission_context=ToolPermissionContext.from_iterables(
            allow_names={"web_search", "bash"},
        ),
    )
    assert [t.name for t in tools] == ["bash"]


def test_clearing_global_policy_restores_tools(resource_mgr_with_registry):
    """Setting an empty/None policy clears it so the next run is unaffected."""
    from agent_core.runtime.tool_permission import (
        from_config_map,
    )

    resource_mgr_with_registry.set_global_tool_policy(
        from_config_map({"web_search": False}),
    )
    # Empty config map → no-op context → policy cleared.
    resource_mgr_with_registry.set_global_tool_policy(from_config_map({}))
    assert resource_mgr_with_registry.global_tool_policy is None
    names = {t.name for t in resource_mgr_with_registry.get_tools_for_role("researcher")}
    assert names == {"web_search", "web_fetch", "bash"}


def test_set_global_policy_replaces_not_accumulates(resource_mgr_with_registry):
    """``set_global_tool_policy`` is last-writer-wins, not additive — this is
    the cross-run safety contract the SDK driver relies on (it sets the policy
    unconditionally at the start of every run, so run N-1's deny never leaks
    into run N)."""
    from agent_core.runtime.tool_permission import (
        from_config_map,
    )

    # Run 1 denies web_search.
    resource_mgr_with_registry.set_global_tool_policy(
        from_config_map({"web_search": False}),
    )
    # Run 2 denies web_fetch instead — must NOT also keep run 1's web_search deny.
    resource_mgr_with_registry.set_global_tool_policy(
        from_config_map({"web_fetch": False}),
    )
    names = {t.name for t in resource_mgr_with_registry.get_tools_for_role("researcher")}
    assert names == {"web_search", "bash"}   # web_search back; only web_fetch denied
