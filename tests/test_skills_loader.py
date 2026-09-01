"""Unit tests for SkillLoader, SkillConfig, and _parse_frontmatter.

Tests skill discovery, frontmatter parsing, toggle, reload,
and SkillConfig model behavior.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_core.components.skills import FileSystemSkillLoader, SkillConfig
from agent_core.components.skills.file_system_loader import _parse_frontmatter

# ── _parse_frontmatter ────────────────────────────────────────────────────


def test_parse_frontmatter_basic():
    text = '---\nname: Test\ndescription: "A skill"\nversion: "2.0"\n---\n# Body\nHello'
    meta, body = _parse_frontmatter(text)
    assert meta["name"] == "Test"
    assert meta["description"] == "A skill"
    assert meta["version"] == "2.0"
    assert body == "# Body\nHello"


def test_parse_frontmatter_list_values():
    text = "---\ntags:\n  - alpha\n  - beta\nallowed-tools:\n  - bash\n  - read_text\n---\nBody"
    meta, body = _parse_frontmatter(text)
    assert meta["tags"] == ["alpha", "beta"]
    assert meta["allowed-tools"] == ["bash", "read_text"]
    assert body == "Body"


def test_parse_frontmatter_empty():
    text = "---\n\n---\nJust body"
    meta, body = _parse_frontmatter(text)
    assert meta == {}
    assert body == "Just body"


def test_parse_frontmatter_no_frontmatter():
    text = "No frontmatter here\nJust markdown"
    meta, body = _parse_frontmatter(text)
    assert meta == {}
    assert body == text


def test_parse_frontmatter_quoted_values():
    text = "---\nname: \"Quoted Name\"\nauthor: 'Single Quoted'\n---\nBody"
    meta, _body = _parse_frontmatter(text)
    assert meta["name"] == "Quoted Name"
    assert meta["author"] == "Single Quoted"


def test_parse_frontmatter_comments_ignored():
    text = '---\nname: Test\n# This is a comment\nversion: "1.0"\n---\nBody'
    meta, _body = _parse_frontmatter(text)
    assert meta["name"] == "Test"
    assert meta["version"] == "1.0"  # Quoted string stays string


def test_parse_frontmatter_mixed_scalar_and_list():
    text = '---\nname: Skill\ntags:\n  - a\n  - b\nversion: "1.0"\n---\nBody'
    meta, _body = _parse_frontmatter(text)
    assert meta["name"] == "Skill"
    assert meta["tags"] == ["a", "b"]
    assert meta["version"] == "1.0"  # Quoted string stays string


def test_parse_frontmatter_numeric_version():
    """PyYAML parses unquoted 1.0 as float — SKILL.md should quote versions."""
    text = "---\nname: Test\nversion: 1.0\n---\nBody"
    meta, _ = _parse_frontmatter(text)
    # Unquoted 1.0 becomes float in YAML — that's correct behavior
    assert meta["version"] == 1.0 or meta["version"] == "1.0"


# ── SkillConfig model ─────────────────────────────────────────────────────


def test_skill_config_defaults():
    cfg = SkillConfig(skill_id="test", name="Test")
    assert cfg.description == ""
    assert cfg.version == "1.0.0"
    assert cfg.allowed_tools == []
    assert cfg.enabled is True
    assert cfg.content == ""
    assert cfg.tags == []


def test_skill_config_skill_md_path():
    cfg = SkillConfig(skill_id="test", name="Test", root_dir="/skills/test")
    assert cfg.skill_md_path == Path("/skills/test/SKILL.md")


def test_skill_config_no_trigger_keywords():
    """trigger_keywords field should no longer exist (dead code cleanup)."""
    cfg = SkillConfig(skill_id="test", name="Test")
    assert not hasattr(cfg, "trigger_keywords") or "trigger_keywords" not in cfg.model_fields


def test_skill_config_no_prompt_injection():
    """prompt_injection method should no longer exist (dead code cleanup)."""
    cfg = SkillConfig(skill_id="test", name="Test")
    assert not hasattr(cfg, "prompt_injection")


# ── SkillLoader ───────────────────────────────────────────────────────────


@pytest.fixture()
def skill_dir(tmp_path):
    """Create a temporary skills directory with two test skills."""
    skills = tmp_path / "skills"
    skills.mkdir()

    # Skill 1: enabled
    s1 = skills / "code-review"
    s1.mkdir()
    (s1 / "SKILL.md").write_text(
        "---\nname: Code Review\ndescription: Review code\nversion: 1.0.0\n"
        "allowed-tools:\n  - bash\n  - read_text\ntags:\n  - code\n---\n# Workflow\nStep 1"
    )

    # Skill 2: enabled
    s2 = skills / "debug"
    s2.mkdir()
    (s2 / "SKILL.md").write_text(
        "---\nname: Debug\ndescription: Debug issues\nversion: 2.0.0\n"
        "allowed-tools:\n  - bash\ntags:\n  - debug\n---\n# Debug Workflow\nStep 1"
    )

    # Not a skill (no SKILL.md)
    s3 = skills / "not-a-skill"
    s3.mkdir()
    (s3 / "README.md").write_text("This is not a skill")

    # File (not a directory)
    (skills / "stray_file.txt").write_text("ignore me")

    return skills


@pytest.fixture()
def loader(skill_dir):
    from agent_core.components.skills import ExtensionsConfig

    config = ExtensionsConfig()  # All skills enabled by default
    return FileSystemSkillLoader(skill_dirs=[skill_dir], extensions_config=config)


def test_discover_finds_skills(loader):
    skills = loader.discover()
    assert len(skills) == 2
    assert "code-review" in skills
    assert "debug" in skills


def test_discover_ignores_non_skills(loader):
    skills = loader.discover()
    assert "not-a-skill" not in skills
    assert "stray_file.txt" not in skills


def test_list_skills_sorted(loader):
    skills = loader.list_skills()
    names = [s.name for s in skills]
    assert names == sorted(names)


def test_get_skill_found(loader):
    skill = loader.get_skill("code-review")
    assert skill is not None
    assert skill.name == "Code Review"
    assert skill.description == "Review code"
    assert skill.allowed_tools == ["bash", "read_text"]
    assert skill.content.startswith("# Workflow")


def test_get_skill_not_found(loader):
    assert loader.get_skill("nonexistent") is None


def test_get_enabled_skills(loader):
    enabled = loader.get_enabled_skills()
    assert len(enabled) == 2


def test_toggle_skill(loader):
    loader.discover()
    assert loader.toggle_skill("code-review", False) is True
    skill = loader.get_skill("code-review")
    assert skill is not None
    assert skill.enabled is False

    enabled = loader.get_enabled_skills()
    assert len(enabled) == 1


def test_toggle_skill_not_found(loader):
    loader.discover()
    assert loader.toggle_skill("nonexistent", True) is False


def test_reload_rediscovers(loader, skill_dir):
    loader.discover()
    assert len(loader.list_skills()) == 2

    # Add a new skill
    s3 = skill_dir / "new-skill"
    s3.mkdir()
    (s3 / "SKILL.md").write_text("---\nname: New Skill\ndescription: Fresh\n---\n# New")

    loader.reload()
    assert len(loader.list_skills()) == 3
    assert loader.get_skill("new-skill") is not None


def test_discover_with_scripts_and_resources(tmp_path):
    skills = tmp_path / "skills"
    skills.mkdir()
    s = skills / "with-extras"
    s.mkdir()
    (s / "SKILL.md").write_text("---\nname: Extra\n---\n# Body")
    scripts_dir = s / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "run.sh").write_text("echo hi")
    resources_dir = s / "resources"
    resources_dir.mkdir()
    (resources_dir / "data.json").write_text("{}")

    from agent_core.components.skills import ExtensionsConfig

    ldr = FileSystemSkillLoader(skill_dirs=[skills], extensions_config=ExtensionsConfig())
    skill = ldr.get_skill("with-extras")
    assert skill is not None
    assert len(skill.scripts) == 1
    assert len(skill.resources) == 1


def test_discover_empty_directory(tmp_path):
    empty = tmp_path / "empty_skills"
    empty.mkdir()

    from agent_core.components.skills import ExtensionsConfig

    ldr = FileSystemSkillLoader(skill_dirs=[empty], extensions_config=ExtensionsConfig())
    skills = ldr.discover()
    assert len(skills) == 0


def test_discover_nonexistent_directory(tmp_path):
    from agent_core.components.skills import ExtensionsConfig

    ldr = FileSystemSkillLoader(
        skill_dirs=[tmp_path / "does_not_exist"],
        extensions_config=ExtensionsConfig(),
    )
    skills = ldr.discover()
    assert len(skills) == 0
