from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent_core.runtime.spill import (
    SpillStore,
    budgeted_preview,
    scope_component,
    truncate_preview,
)


def test_scope_is_hashed_and_session_isolated(tmp_path: Path) -> None:
    first = SpillStore(tmp_path, "task:session-a", visible_root="/spill")
    second = SpillStore(tmp_path, "task:session-b", visible_root="/spill")

    assert first.component == scope_component("task:session-a")
    assert first.directory != second.directory
    assert first.visible_directory == f"/spill/{first.component}"


def test_write_is_content_addressed_idempotent_and_readable(tmp_path: Path) -> None:
    store = SpillStore(tmp_path, "session", visible_root="/spill")
    body = "important body\n" * 500

    first = store.write("bash", body, require_visible=True)
    second = store.write("bash", body, require_visible=True)

    assert first is not None and second is not None
    assert first == second
    physical, visible = first
    assert physical.is_file()
    assert visible.startswith(store.visible_directory + "/")
    assert store.read(physical) == body
    assert store.read(visible) == body
    assert physical.stat().st_mode & 0o222 == 0


def test_read_rejects_paths_outside_the_scope(tmp_path: Path) -> None:
    store = SpillStore(tmp_path / "spill", "session", visible_root="/spill")
    outside = tmp_path / "outside.md"
    outside.write_text("secret", encoding="utf-8")

    assert store.read(outside) is None
    assert store.read(f"{store.visible_directory}/../outside.md") is None
    assert not store.contains_path(outside)


def test_read_created_supports_lifecycle_reads_without_scope(tmp_path: Path) -> None:
    store = SpillStore(tmp_path / "spill", "session", visible_root="/spill")
    written = store.write("bash", "captured")
    outside = tmp_path / "outside.md"
    outside.write_text("secret", encoding="utf-8")

    assert written is not None
    assert SpillStore.read_created(written[0]) == "captured"
    assert SpillStore.read_created(outside) is None


def test_compaction_spill_skips_tiny_bodies(tmp_path: Path) -> None:
    store = SpillStore(tmp_path, "session", visible_root="/spill")

    assert store.spill_compacted_body("tool", "small") is None
    assert not store.directory.exists()


def test_overflow_keeps_head_tail_and_a_recovery_pointer(tmp_path: Path) -> None:
    store = SpillStore(tmp_path, "session", visible_root="/spill")
    body = "HEAD\n" + "middle\n" * 2_000 + "TAIL\n"

    preview = store.overflow("bash", body, cap=2_000)

    assert len(preview) <= 2_000
    assert "HEAD" in preview
    assert "TAIL" in preview
    assert "chars elided" in preview
    assert store.visible_directory in preview
    saved = next(store.directory.glob("*.md"))
    assert store.read(saved) == body


def test_head_mode_uses_budget_on_ranked_prefix() -> None:
    text = "first\n" + "x" * 3_000 + "\nlast"

    result = truncate_preview(text, 1_000, mode="head")

    assert len(result) <= 1_000
    assert result.startswith("first")
    assert "last" not in result


def test_footer_is_charged_to_the_same_cap() -> None:
    result = budgeted_preview(
        "x" * 10_000,
        cap=2_000,
        ref="/spill/scope/body.md",
    )

    assert len(result) <= 2_000
    assert "/spill/scope/body.md" in result


def test_aggregate_budget_spills_before_shortening(tmp_path: Path) -> None:
    store = SpillStore(tmp_path, "session", visible_root="/spill")
    results = ["a" * 7_000, "b" * 7_000, "small"]

    adjusted = store.enforce_aggregate_budget(
        results,
        tool_names=["first", "second", "third"],
        cap=9_000,
    )

    assert len(adjusted) == len(results)
    assert sum(map(len, adjusted)) <= 9_000
    assert adjusted[2] == "small"
    assert list(store.directory.glob("*.md"))


def test_unmounted_store_persists_but_does_not_advertise_a_path(tmp_path: Path) -> None:
    store = SpillStore(tmp_path, "session")
    body = "x" * 5_000

    preview = store.overflow("tool", body, cap=1_500)

    assert "not readable from this backend" in preview
    assert store.spill_compacted_body("tool", body) is None
    assert list(store.directory.glob("*.md"))


def test_cleanup_removes_only_its_own_session(tmp_path: Path) -> None:
    first = SpillStore(tmp_path, "first", visible_root="/spill")
    second = SpillStore(tmp_path, "second", visible_root="/spill")
    assert first.write("tool", "a" * 2_000)
    assert second.write("tool", "b" * 2_000)

    assert first.cleanup() == 1
    assert not first.directory.exists()
    assert second.directory.exists()
    assert second.cleanup() == 1


def test_body_pointer_detection_is_root_bounded(tmp_path: Path) -> None:
    store = SpillStore(tmp_path, "session", visible_root="/spill")
    written = store.write("tool", "x" * 2_000)
    assert written is not None

    assert store.body_names_file(f"recover at {written[1]}")
    assert not store.body_names_file("/spillover/not-a-pointer.md")


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_read_does_not_follow_a_spill_file_symlink_outside_scope(tmp_path: Path) -> None:
    store = SpillStore(tmp_path / "root", "session", visible_root="/spill")
    store.directory.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_text("secret", encoding="utf-8")
    link = store.directory / "escape.md"
    link.symlink_to(outside)

    assert store.read(link) is None
