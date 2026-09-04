"""End-to-end deliverable reconciliation against a real filesystem.

``tests/test_deliverables.py`` pins each function's contract in isolation. This
file exercises the two incidents the module was extracted for as whole rounds —
baseline, publish (or fail to), reconcile, render the terminal report — plus the
filesystem edges that only a real tree can produce: symlink escapes, unreadable
directories, chunked hashing, and ``/outputs`` persisting across rounds.

Every case builds its own tree under pytest's ``tmp_path`` (repo convention: no
``conftest.py``), and passes it as ``outputs_host_root`` so the sandbox path
``/outputs`` resolves there instead of the machine's real root.
"""
from __future__ import annotations

import os
import sys

import pytest

from agent_core.components.deliverables import (
    claimed_output_paths,
    incomplete_manifest_paths,
    is_complete_run,
    prepend_publication_failure,
    reconcile_publication_state,
    reconcile_publication_state_async,
    snapshot_manifest,
    unverified_claim_paths,
)

COMPLETE = ("no_tool", "finalized")


def _round(tmp_path, manifest, *, baseline=None, final_text="", language=""):
    """Run one reconciliation the way a host would, against a real tree."""
    state = {
        "outputs_host_root": str(tmp_path),
        "deliverable_manifest": manifest,
        "manifest_baseline": baseline,
    }
    return reconcile_publication_state(state, final_text=final_text, language=language)


# --- the two original incidents, as whole rounds --------------------------


def test_publisher_that_only_printed_a_bash_block_is_caught(tmp_path):
    """Incident 1: the write was emitted as prose, the tool never ran."""
    manifest = ("/outputs/report.pdf",)
    baseline = snapshot_manifest(manifest, host_root=str(tmp_path))
    assert baseline == {"/outputs/report.pdf": ()}

    answer = "Done. I saved the report to /outputs/report.pdf.\n\n```bash\ncp x /outputs/report.pdf\n```"
    failures, message = _round(tmp_path, manifest, baseline=baseline, final_text=answer)

    assert failures == ("/outputs/report.pdf",)
    assert "Deliverable publication failed" in message
    report = prepend_publication_failure(answer, message)
    assert report.startswith("## Deliverable publication failed")
    assert not is_complete_run("no_tool", complete_stop_reasons=COMPLETE, unverified_claims=failures)


def test_bare_text_answer_that_did_write_the_file_is_complete(tmp_path):
    """Incident 2's mirror: delivered, but stopped on ``no_tool``."""
    manifest = ("/outputs/report.pdf",)
    baseline = snapshot_manifest(manifest, host_root=str(tmp_path))
    (tmp_path / "report.pdf").write_bytes(b"%PDF-1.7 real bytes")

    failures, message = _round(
        tmp_path, manifest, baseline=baseline, final_text="Saved to /outputs/report.pdf.",
    )

    assert failures == ()
    assert message == ""
    assert is_complete_run("no_tool", complete_stop_reasons=COMPLETE, unverified_claims=failures)


# --- /outputs persists across rounds -------------------------------------


def test_previous_rounds_file_is_not_this_rounds_delivery(tmp_path):
    """Round 2 declares the same path, writes nothing, and must not pass."""
    manifest = ("/outputs/report.pdf",)
    (tmp_path / "report.pdf").write_bytes(b"round one output")

    baseline = snapshot_manifest(manifest, host_root=str(tmp_path))
    assert baseline["/outputs/report.pdf"]  # round 1's file is on disk

    failures, message = _round(
        tmp_path, manifest, baseline=baseline, final_text="Saved to /outputs/report.pdf.",
    )
    assert failures == ("/outputs/report.pdf",)
    assert "not updated this round" in message

    # Overwriting it with new content in round 2 is a delivery.
    (tmp_path / "report.pdf").write_bytes(b"round two output, different bytes")
    failures, message = _round(
        tmp_path, manifest, baseline=baseline, final_text="Saved to /outputs/report.pdf.",
    )
    assert (failures, message) == ((), "")


def test_directory_manifest_delivers_when_one_new_file_appears(tmp_path):
    manifest = ("/outputs/site",)
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("round one")
    baseline = snapshot_manifest(manifest, host_root=str(tmp_path))

    # Round 2 adds a page; the untouched index.html alone would not have counted.
    (site / "about.html").write_text("round two")
    failures, _ = _round(
        tmp_path, manifest, baseline=baseline, final_text="Published /outputs/site/about.html.",
    )
    assert failures == ()


def test_nested_tree_is_hashed_recursively(tmp_path):
    manifest = ("/outputs/site",)
    deep = tmp_path / "site" / "a" / "b"
    deep.mkdir(parents=True)
    (deep / "c.txt").write_text("deep")
    snap = snapshot_manifest(manifest, host_root=str(tmp_path))
    assert [entry[0] for entry in snap["/outputs/site"]] == [os.path.join("a", "b", "c.txt")]
    assert incomplete_manifest_paths(manifest, host_root=str(tmp_path)) == ()


# --- filesystem edges ----------------------------------------------------


def test_symlinked_file_never_counts_as_published(tmp_path):
    secret = tmp_path.parent / "secret.txt"
    secret.write_text("not a deliverable")
    (tmp_path / "report.pdf").symlink_to(secret)

    assert snapshot_manifest(("/outputs/report.pdf",), host_root=str(tmp_path)) == {
        "/outputs/report.pdf": (),
    }
    assert incomplete_manifest_paths(("/outputs/report.pdf",), host_root=str(tmp_path)) == (
        "/outputs/report.pdf",
    )


def test_symlinked_subdirectory_cannot_smuggle_content_into_a_manifest(tmp_path):
    """A dir symlink inside the declared subtree must not be walked."""
    outside = tmp_path.parent / "elsewhere"
    outside.mkdir()
    (outside / "borrowed.txt").write_text("lives outside the manifest")

    site = tmp_path / "site"
    site.mkdir()
    (site / "linked").symlink_to(outside, target_is_directory=True)

    snap = snapshot_manifest(("/outputs/site",), host_root=str(tmp_path))
    assert snap["/outputs/site"] == ()
    assert incomplete_manifest_paths(("/outputs/site",), host_root=str(tmp_path)) == ("/outputs/site",)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode bits")
@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_an_unreadable_subdirectory_does_not_raise(tmp_path):
    site = tmp_path / "site"
    locked = site / "locked"
    locked.mkdir(parents=True)
    (locked / "x.txt").write_text("unreachable")
    (site / "readable.txt").write_text("counted")
    locked.chmod(0o000)
    try:
        snap = snapshot_manifest(("/outputs/site",), host_root=str(tmp_path))
        assert [entry[0] for entry in snap["/outputs/site"]] == ["readable.txt"]
    finally:
        locked.chmod(0o700)


def test_hashing_spans_the_chunk_boundary(tmp_path):
    """Files are read in 1 MiB chunks; a change past the first chunk must show."""
    import hashlib

    target = tmp_path / "big.bin"
    payload = bytearray(os.urandom(1 << 20) + b"A" * 4096)
    target.write_bytes(payload)
    baseline = snapshot_manifest(("/outputs/big.bin",), host_root=str(tmp_path))
    assert baseline["/outputs/big.bin"][0][3] == hashlib.sha256(payload).hexdigest()

    payload[-1:] = b"B"  # only the final byte, well past chunk one
    target.write_bytes(payload)
    assert incomplete_manifest_paths(
        ("/outputs/big.bin",), baseline, host_root=str(tmp_path),
    ) == ()


def test_scratch_is_not_a_deliverable_even_when_written(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "notes.md").write_text("intermediate")
    failures, message = _round(
        tmp_path, (), final_text="Working notes are in /outputs/scratch/notes.md.",
    )
    assert (failures, message) == ((), "")


# --- claim scanning against real prose -----------------------------------


def test_non_ascii_deliverable_claim_is_detected(tmp_path):
    """A Chinese-named claim that was never written must not pass silently."""
    failures, message = _round(tmp_path, (), final_text="已写出 /outputs/报告.pdf。", language="zh")
    assert failures == ("/outputs/报告.pdf",)
    assert "交付声明无法验证" in message


def test_non_ascii_deliverable_inside_its_manifest_is_not_flagged(tmp_path):
    manifest = ("/outputs/报告.pdf",)
    baseline = snapshot_manifest(manifest, host_root=str(tmp_path))
    (tmp_path / "报告.pdf").write_text("交付内容")
    failures, message = _round(
        tmp_path, manifest, baseline=baseline, final_text="已写出 /outputs/报告.pdf。",
    )
    assert (failures, message) == ((), "")


def test_a_space_bearing_path_is_taken_whole_when_the_prose_delimits_it(tmp_path):
    manifest = ("/outputs/final report.pdf",)
    baseline = snapshot_manifest(manifest, host_root=str(tmp_path))
    (tmp_path / "final report.pdf").write_text("delivered")
    failures, message = _round(
        tmp_path, manifest, baseline=baseline, final_text="Saved to `/outputs/final report.pdf`.",
    )
    assert (failures, message) == ((), "")
    assert claimed_output_paths("Saved to `/outputs/final report.pdf`.") == (
        "/outputs/final report.pdf",
    )


def test_markdown_link_target_is_a_claim(tmp_path):
    assert claimed_output_paths("see [the report](/outputs/report.pdf)") == ("/outputs/report.pdf",)


def test_sibling_paths_under_a_common_prefix_both_survive():
    assert claimed_output_paths("`/outputs/a` and `/outputs/a/b.md`") == (
        "/outputs/a",
        "/outputs/a/b.md",
    )


def test_claim_under_a_declared_directory_needs_the_directory_to_exist(tmp_path):
    manifest = ("/outputs/site",)
    # Nothing on disk yet: the claim is not covered by a directory that is absent.
    assert unverified_claim_paths(
        "Published /outputs/site/index.html.", manifest=manifest, host_root=str(tmp_path),
    ) == ("/outputs/site/index.html",)

    (tmp_path / "site").mkdir()
    assert unverified_claim_paths(
        "Published /outputs/site/index.html.", manifest=manifest, host_root=str(tmp_path),
    ) == ()


# --- host wiring ---------------------------------------------------------


def test_ignore_matcher_applies_to_nested_entries(tmp_path):
    site = tmp_path / "site"
    (site / "logs").mkdir(parents=True)
    (site / "logs" / "debug.log").write_text("noise")
    (site / "index.html").write_text("real")

    def ignore(rel: str) -> bool:
        return rel.startswith("site/logs")

    snap = snapshot_manifest(("/outputs/site",), host_root=str(tmp_path), ignore_matcher=ignore)
    assert [entry[0] for entry in snap["/outputs/site"]] == ["index.html"]


def test_an_ignored_manifest_entry_is_reported_incomplete(tmp_path):
    (tmp_path / "report.pdf").write_text("present but ignored by host policy")
    assert incomplete_manifest_paths(
        ("/outputs/report.pdf",), host_root=str(tmp_path), ignore_matcher=lambda rel: True,
    ) == ("/outputs/report.pdf",)


def test_without_host_root_the_sandbox_path_is_used_verbatim(tmp_path):
    """No mount resolver, no guessing: an absent /outputs simply delivers nothing."""
    assert incomplete_manifest_paths(("/outputs/nope.pdf",)) == ("/outputs/nope.pdf",)


async def test_async_reconciliation_matches_sync_on_a_real_tree(tmp_path):
    manifest = ("/outputs/site",)
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("published")
    state = {
        "outputs_host_root": str(tmp_path),
        "deliverable_manifest": manifest,
        "manifest_baseline": {"/outputs/site": ()},
    }
    text = "Published /outputs/site/index.html."
    assert await reconcile_publication_state_async(state, final_text=text) == (
        reconcile_publication_state(state, final_text=text)
    )


def test_prose_quotes_do_not_swallow_a_sentence_as_a_path():
    """Straight quotes delimit phrases, not paths; the real claim must survive."""
    assert claimed_output_paths("'/outputs/a.pdf and more prose here'") == ("/outputs/a.pdf",)
    assert claimed_output_paths('"/outputs/a.pdf, which took a while"') == ("/outputs/a.pdf",)
    assert claimed_output_paths("it's saved at /outputs/x.pdf, don't worry") == ("/outputs/x.pdf",)


def test_an_undelimited_space_path_is_still_covered_by_its_manifest_entry(tmp_path):
    """The scanner truncates at the space; the manifest outranks that guess."""
    manifest = ("/outputs/final report.pdf",)
    baseline = snapshot_manifest(manifest, host_root=str(tmp_path))
    (tmp_path / "final report.pdf").write_text("delivered")
    failures, message = _round(
        tmp_path, manifest, baseline=baseline, final_text='Saved to "/outputs/final report.pdf".',
    )
    assert (failures, message) == ((), "")


def test_full_width_sentence_punctuation_is_not_part_of_the_path():
    assert claimed_output_paths("已写出 /outputs/报告.pdf。") == ("/outputs/报告.pdf",)
    assert claimed_output_paths("见 /outputs/a.pdf，以及 /outputs/b.pdf！") == (
        "/outputs/a.pdf",
        "/outputs/b.pdf",
    )
