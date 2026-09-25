from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import check_version_bump as gate
from scripts import version as version_script
from scripts.prepare_release import prepare
from scripts.release_notes import validate_release


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def write_lock(root: Path, version: str) -> None:
    (root / "uv.lock").write_text(f'[[package]]\nname = "apodex-agent-core"\nversion = "{version}"\n')


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "apodex-agent-core"\nversion = "0.2.0"\n')
    write_lock(tmp_path, "0.2.0")
    (tmp_path / "CHANGELOG.md").write_text('# Changelog\n\n## [0.2.0] - 2026-01-01\n\nExisting notes.\n')
    (tmp_path / "changes").mkdir()
    (tmp_path / "changes/.gitkeep").touch()
    (tmp_path / "agent_core").mkdir()
    (tmp_path / "agent_core/x.py").write_text('x = 1\n')
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.name", "Test")
    git(tmp_path, "config", "user.email", "test@example.com")
    commit(tmp_path)
    git(tmp_path, "branch", "base")
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "_git", lambda *args: git(tmp_path, *args))
    monkeypatch.setattr(gate, "read_version", lambda: version_script.read_version(tmp_path / "pyproject.toml"))
    return tmp_path


def commit(root: Path) -> None:
    git(root, "add", ".")
    git(root, "commit", "-qm", "test")


def test_code_change_requires_new_fragment(repo: Path) -> None:
    (repo / "agent_core/x.py").write_text('x = 2\n')
    commit(repo)
    assert gate.main(["--base", "base"]) == 1
    (repo / "changes/42.fix.md").write_text("Fix the result.")
    commit(repo)
    assert gate.main(["--base", "base"]) == 0


def test_existing_fragment_cannot_cover_new_code(repo: Path) -> None:
    (repo / "changes/old.fix.md").write_text("Earlier change.")
    commit(repo)
    git(repo, "branch", "-f", "base")
    (repo / "agent_core/x.py").write_text('x = 2\n')
    (repo / "changes/old.fix.md").write_text("Repurposed old change.")
    commit(repo)
    assert gate.main(["--base", "base"]) == 1


@pytest.mark.parametrize("name,text", [("42.md", "A fix"), ("42.fix.md", " "), ("42.fix.md", "## Heading")])
def test_malformed_fragment_rejected(repo: Path, name: str, text: str) -> None:
    (repo / "changes" / name).write_text(text)
    commit(repo)
    assert gate.main(["--base", "base"]) == 1


def test_docs_only_needs_no_fragment(repo: Path) -> None:
    (repo / "README.md").write_text("Documentation")
    commit(repo)
    assert gate.main(["--base", "base"]) == 0


@pytest.mark.parametrize(("kind", "expected"), [("fix", "0.2.1"), ("feature", "0.3.0"), ("breaking", "0.3.0")])
def test_prepare_release_and_gate(repo: Path, kind: str, expected: str) -> None:
    note = repo / f"changes/42.{kind}.md"
    note.write_text("Consumer-facing change.")
    commit(repo)
    git(repo, "branch", "-f", "base")
    original = (repo / "pyproject.toml").read_text()
    preview = prepare(repo, dry_run=True)
    assert expected in preview
    assert note.exists()
    assert (repo / "pyproject.toml").read_text() == original
    prepare(repo)
    assert not note.exists()
    assert version_script.read_version(repo / "pyproject.toml") == expected
    commit(repo)
    assert gate.main(["--base", "base"]) == 1  # lock not regenerated yet
    write_lock(repo, expected)
    commit(repo)
    assert gate.main(["--base", "base"]) == 0
    validate_release(repo, expected)


def test_tag_refuses_unreleased_tree(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(version_script, "ROOT", repo)
    monkeypatch.setattr(version_script, "read_version", lambda: "0.2.0")
    (repo / "changes/42.fix.md").write_text("Not released yet.")
    assert version_script.main(["--check-tag", "v0.2.0"]) == 1


def test_release_refuses_missing_notes_and_new_fragments(repo: Path) -> None:
    with pytest.raises(ValueError, match="stale"):
        validate_release(repo, "0.2.1")
    write_lock(repo, "0.2.1")
    with pytest.raises(ValueError, match="no non-empty"):
        validate_release(repo, "0.2.1")
    (repo / "changes/late.fix.md").write_text("Merged after release preparation.")
    with pytest.raises(ValueError, match="Unreleased"):
        validate_release(repo, "0.2.1")


def test_prepare_rejects_insufficient_version_without_mutation(repo: Path) -> None:
    note = repo / "changes/43.feature.md"
    note.write_text("New public API.")
    with pytest.raises(ValueError, match="below"):
        prepare(repo, version="0.2.1")
    assert note.exists()
    assert version_script.read_version(repo / "pyproject.toml") == "0.2.0"


def test_gate_rejects_downgrade(repo: Path) -> None:
    p = repo / "pyproject.toml"
    p.write_text(p.read_text().replace("0.2.0", "0.1.9"))
    write_lock(repo, "0.1.9")
    commit(repo)
    assert gate.main(["--base", "base"]) == 1


def test_two_feature_branches_merge_without_release_file_conflicts(repo: Path) -> None:
    git(repo, "checkout", "-qb", "feature-a")
    (repo / "changes/42.fix.md").write_text("Fix A.")
    (repo / "agent_core/a.py").write_text('a = 1\n')
    commit(repo)
    git(repo, "checkout", "-qb", "feature-b", "base")
    (repo / "changes/43.feature.md").write_text("Feature B.")
    (repo / "agent_core/b.py").write_text('b = 1\n')
    commit(repo)
    git(repo, "merge", "--no-edit", "feature-a")
    assert gate.main(["--base", "feature-a"]) == 0
    assert version_script.read_version(repo / "pyproject.toml") == "0.2.0"
    notes = prepare(repo)
    assert "Fix A." in notes and "Feature B." in notes
    assert "0.3.0" in notes


@pytest.mark.parametrize("operation", ["delete", "edit", "rename"])
def test_feature_pr_cannot_remove_or_repurpose_pending_notes(repo: Path, operation: str) -> None:
    note = repo / "changes/old.fix.md"
    note.write_text("Earlier consumer-facing change.")
    commit(repo)
    git(repo, "branch", "-f", "base")
    if operation == "delete":
        note.unlink()
    elif operation == "edit":
        note.write_text("Different note.")
    else:
        note.rename(repo / "changes/new.fix.md")
    commit(repo)
    assert gate.main(["--base", "base"]) == 1



def test_manual_release_cannot_hide_feature_in_patch(repo: Path) -> None:
    note = repo / "changes/43.feature.md"
    note.write_text("New API.")
    commit(repo)
    git(repo, "branch", "-f", "base")
    note.unlink()
    p = repo / "pyproject.toml"
    p.write_text(p.read_text().replace("0.2.0", "0.2.1"))
    write_lock(repo, "0.2.1")
    p = repo / "CHANGELOG.md"
    p.write_text(p.read_text().replace("## [0.2.0]", "## [0.2.1]"))
    commit(repo)
    assert gate.main(["--base", "base"]) == 1
