"""Reloading skill state must not invert an operator's decision.

``get_enabled_skills`` auto-reloads when the extensions config changes on
disk. It called ``ExtensionsConfig.from_file()`` with no argument, which
restarts the cwd/env search instead of re-reading the file the config actually
came from. For a config loaded from an explicit path outside those locations
the search finds nothing and returns empty defaults — and because
``is_skill_enabled`` defaults to True for anything unlisted, "disable one
skill" was applied as "enable everything". The empty config also carries no
``_file_path``, so ``has_changed`` returned False from then on and the state
could never recover.
"""

from __future__ import annotations

import json
import os

import pytest

from agent_core.components.skills import ExtensionsConfig, FileSystemSkillLoader


@pytest.fixture()
def skill_dir(tmp_path):
    skills = tmp_path / "skills"
    skills.mkdir()
    for name in ("code-review", "debug"):
        d = skills / name
        d.mkdir()
        (d / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: d\nversion: 1.0.0\n---\n# Body\n",
        )
    return skills


@pytest.fixture()
def config_file(tmp_path):
    """A config in a directory the cwd/env search will never look in."""
    path = tmp_path / "elsewhere" / "extensions.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"skills": {}}))
    return path


def _write(path, skills: dict[str, bool]) -> None:
    path.write_text(json.dumps({
        "skills": {k: {"enabled": v} for k, v in skills.items()},
    }))
    # Ensure the mtime moves even on a coarse-grained filesystem clock.
    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime + 10))


def _loader(skill_dir, config_file) -> FileSystemSkillLoader:
    return FileSystemSkillLoader(
        skill_dirs=[skill_dir],
        extensions_config=ExtensionsConfig.from_file(config_file),
    )


def test_source_path_is_exposed():
    """The reloader needs to know where the config came from."""
    assert ExtensionsConfig().source_path is None


def test_source_path_records_an_explicit_load(config_file):
    config = ExtensionsConfig.from_file(config_file)
    assert config.source_path == config_file


def test_disable_survives_the_auto_reload(skill_dir, config_file):
    loader = _loader(skill_dir, config_file)
    assert {s.skill_id for s in loader.get_enabled_skills()} == {
        "code-review", "debug",
    }

    _write(config_file, {"debug": False})

    assert {s.skill_id for s in loader.get_enabled_skills()} == {"code-review"}


def test_reload_reads_the_same_file(skill_dir, config_file):
    loader = _loader(skill_dir, config_file)
    _write(config_file, {"debug": False})

    loader.reload()

    assert {s.skill_id for s in loader.get_enabled_skills()} == {"code-review"}


def test_relative_source_path_survives_cwd_change(
    skill_dir, config_file, monkeypatch,
):
    """Reload must keep tracking the originally opened relative path."""
    monkeypatch.chdir(config_file.parent)
    config = ExtensionsConfig.from_file(config_file.name)
    assert config.source_path == config_file
    loader = FileSystemSkillLoader(
        skill_dirs=[skill_dir],
        extensions_config=config,
    )

    other = config_file.parent / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    _write(config_file, {"debug": False})

    assert {s.skill_id for s in loader.get_enabled_skills()} == {"code-review"}

    _write(config_file, {"code-review": False})
    loader.reload()
    assert {s.skill_id for s in loader.get_enabled_skills()} == {"debug"}


def test_auto_reload_keeps_working_after_the_first_change(skill_dir, config_file):
    """The old failure mode wedged ``has_changed`` off permanently."""
    loader = _loader(skill_dir, config_file)
    _write(config_file, {"debug": False})
    assert {s.skill_id for s in loader.get_enabled_skills()} == {"code-review"}

    _write(config_file, {"code-review": False})

    assert {s.skill_id for s in loader.get_enabled_skills()} == {"debug"}


def test_unreadable_config_keeps_the_previous_state(skill_dir, config_file):
    """Failing open to "everything enabled" is the wrong direction."""
    loader = _loader(skill_dir, config_file)
    _write(config_file, {"debug": False})
    assert {s.skill_id for s in loader.get_enabled_skills()} == {"code-review"}

    config_file.unlink()
    loader.reload()

    assert {s.skill_id for s in loader.get_enabled_skills()} == {"code-review"}


# ── toggle_skill discovers lazily, like every other accessor ───────────────


def test_toggle_on_a_fresh_loader_is_not_a_silent_no_op(skill_dir, config_file):
    loader = _loader(skill_dir, config_file)

    assert loader.toggle_skill("debug", False) is True

    skill = loader.get_skill("debug")
    assert skill is not None
    assert skill.enabled is False


def test_toggle_still_reports_a_genuinely_missing_skill(skill_dir, config_file):
    loader = _loader(skill_dir, config_file)
    assert loader.toggle_skill("no-such-skill", False) is False
