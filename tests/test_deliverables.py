"""Tests for deliverable reconciliation.

No shared fixtures by repo convention (there is no ``conftest.py``); each case
builds its own tree under pytest's ``tmp_path``. ``asyncio_mode = "auto"`` means
async tests need no marker.

The two scenarios the module exists for are named explicitly:

* ``no_tool`` + file written  → complete   (a bare-text answer is this
  workflow's terminal signal; reporting it incomplete loses the report)
* ``no_tool`` + nothing written → incomplete (the run claimed a deliverable in
  prose and never produced it, yet was reported successful)
"""
from __future__ import annotations

import sys

import pytest

from agent_core.components.deliverables import (
    claimed_output_paths,
    incomplete_manifest_paths,
    incomplete_manifest_paths_async,
    is_complete_run,
    normalise_output_paths,
    prepend_publication_failure,
    reconcile_publication_state,
    reconcile_publication_state_async,
    snapshot_manifest,
    unverified_claim_paths,
    unverified_claim_text,
)

_NO_TOOL = "no_tool"


# ── claim extraction ──────────────────────────────────────────────────────────


def test_bare_outputs_root_is_not_a_delivery_claim():
    """The sentence that names ``/outputs`` bare is usually the OPPOSITE of a
    claim ("no deliverable was produced; /outputs is empty"). Counting it once
    inverted the verdict on runs that correctly delivered nothing."""
    assert claimed_output_paths("no deliverable was produced; /outputs is empty") == ()


def test_scratch_is_not_a_delivery_claim():
    assert claimed_output_paths("staged at /outputs/scratch/wip.md") == ()
    assert claimed_output_paths("see /outputs/scratch") == ()


def test_escaping_path_is_not_a_claim():
    assert claimed_output_paths("read /outputs/../etc/passwd") == ()


def test_real_claim_is_extracted_with_trailing_punctuation_stripped():
    assert claimed_output_paths("Saved to /outputs/synthesis.md.") == ("/outputs/synthesis.md",)
    assert claimed_output_paths("wrote `/outputs/a/b.html`") == ("/outputs/a/b.html",)


def test_claims_are_deduplicated_in_first_seen_order():
    text = "wrote /outputs/b.md then /outputs/a.md then /outputs/b.md again"
    assert claimed_output_paths(text) == ("/outputs/b.md", "/outputs/a.md")


# ── delivery identity ─────────────────────────────────────────────────────────


def test_zero_byte_file_delivers_nothing(tmp_path):
    """``touch /outputs/final.pdf`` creates a path but delivers no content."""
    (tmp_path / "final.pdf").write_bytes(b"")
    assert snapshot_manifest(["/outputs/final.pdf"], host_root=str(tmp_path)) == {
        "/outputs/final.pdf": (),
    }
    assert incomplete_manifest_paths(
        ["/outputs/final.pdf"], host_root=str(tmp_path),
    ) == ("/outputs/final.pdf",)


def test_missing_file_is_incomplete(tmp_path):
    assert incomplete_manifest_paths(
        ["/outputs/never.md"], host_root=str(tmp_path),
    ) == ("/outputs/never.md",)


def test_newly_created_file_counts_as_delivered(tmp_path):
    (tmp_path / "report.md").write_text("real content", encoding="utf-8")
    assert incomplete_manifest_paths(["/outputs/report.md"], host_root=str(tmp_path)) == ()


def test_mtime_only_rewrite_is_not_delivery(tmp_path):
    """Byte-identical carryover must not count: a watcher omits unchanged
    mounted history from the share manifest, so calling it delivered would
    contradict what the user can download."""
    path = tmp_path / "carry.md"
    path.write_text("same bytes", encoding="utf-8")
    baseline = snapshot_manifest(["/outputs/carry.md"], host_root=str(tmp_path))

    path.touch()  # mtime moves, sha256 does not

    assert incomplete_manifest_paths(
        ["/outputs/carry.md"], baseline, host_root=str(tmp_path),
    ) == ("/outputs/carry.md",)


def test_content_change_over_baseline_is_delivery(tmp_path):
    path = tmp_path / "carry.md"
    path.write_text("old", encoding="utf-8")
    baseline = snapshot_manifest(["/outputs/carry.md"], host_root=str(tmp_path))

    path.write_text("new content this round", encoding="utf-8")

    assert incomplete_manifest_paths(
        ["/outputs/carry.md"], baseline, host_root=str(tmp_path),
    ) == ()


def test_deletion_alone_is_not_delivery(tmp_path):
    path = tmp_path / "gone.md"
    path.write_text("was here", encoding="utf-8")
    baseline = snapshot_manifest(["/outputs/gone.md"], host_root=str(tmp_path))

    path.unlink()

    assert incomplete_manifest_paths(
        ["/outputs/gone.md"], baseline, host_root=str(tmp_path),
    ) == ("/outputs/gone.md",)


def test_host_root_maps_sandbox_path_to_real_tree(tmp_path):
    """Sandbox ``/outputs/x`` resolves under the injected host root, not ``/``."""
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "deep.txt").write_text("content", encoding="utf-8")
    assert incomplete_manifest_paths(["/outputs/sub"], host_root=str(tmp_path)) == ()


def test_ignore_matcher_excludes_a_path(tmp_path):
    (tmp_path / "skipme.log").write_text("noise", encoding="utf-8")
    assert incomplete_manifest_paths(
        ["/outputs/skipme.log"],
        host_root=str(tmp_path),
        ignore_matcher=lambda rel: rel.endswith(".log"),
    ) == ("/outputs/skipme.log",)


def test_a_raising_ignore_matcher_never_breaks_reconciliation(tmp_path):
    """Reconciliation runs on the terminal path; a host predicate that throws
    must degrade to "not ignored" rather than lose the verdict."""
    (tmp_path / "ok.md").write_text("content", encoding="utf-8")

    def boom(_rel: str) -> bool:
        raise RuntimeError("host predicate exploded")

    assert incomplete_manifest_paths(
        ["/outputs/ok.md"], host_root=str(tmp_path), ignore_matcher=boom,
    ) == ()


# ── unverified claims (the no-manifest branch) ─────────────────────────────────


def test_claim_outside_declared_manifest_is_unverified(tmp_path):
    assert unverified_claim_paths(
        "saved to /outputs/rogue.md",
        manifest=["/outputs/declared.md"],
        host_root=str(tmp_path),
    ) == ("/outputs/rogue.md",)


def test_claim_matching_declared_manifest_is_verified(tmp_path):
    assert unverified_claim_paths(
        "saved to /outputs/declared.md",
        manifest=["/outputs/declared.md"],
        host_root=str(tmp_path),
    ) == ()


def test_claim_under_a_declared_directory_is_verified(tmp_path):
    (tmp_path / "site").mkdir()
    assert unverified_claim_paths(
        "wrote /outputs/site/index.html",
        manifest=["/outputs/site"],
        host_root=str(tmp_path),
    ) == ()


# ── top-level reconciliation ──────────────────────────────────────────────────


def test_empty_manifest_with_a_claim_is_flagged(tmp_path):
    """The r4 shape: the answer names ``/outputs/synthesis.md`` while nothing was
    ever written, and the run was reported successful."""
    state = {"outputs_host_root": str(tmp_path)}
    failures, message = reconcile_publication_state(
        state, final_text="# /outputs/synthesis.md (conceptual contents)",
    )
    assert failures == ("/outputs/synthesis.md",)
    assert "/outputs/synthesis.md" in message


def test_empty_manifest_without_a_claim_stays_silent(tmp_path):
    """A run that legitimately produces no deliverable must not be flagged —
    this is the false-positive direction that matters most."""
    state = {"outputs_host_root": str(tmp_path)}
    assert reconcile_publication_state(state, final_text="Here is the answer: 42.") == ((), "")


def test_declared_manifest_is_authoritative(tmp_path):
    (tmp_path / "done.md").write_text("shipped", encoding="utf-8")
    state = {
        "outputs_host_root": str(tmp_path),
        "deliverable_manifest": ["/outputs/done.md", "/outputs/absent.md"],
    }
    failures, message = reconcile_publication_state(state, final_text="all set")
    assert failures == ("/outputs/absent.md",)
    assert "/outputs/absent.md" in message


async def test_async_wrappers_match_sync(tmp_path):
    (tmp_path / "x.md").write_text("content", encoding="utf-8")
    state = {"outputs_host_root": str(tmp_path)}
    assert await reconcile_publication_state_async(
        state, final_text="wrote /outputs/x.md",
    ) == reconcile_publication_state(state, final_text="wrote /outputs/x.md")
    assert await incomplete_manifest_paths_async(
        ["/outputs/x.md"], host_root=str(tmp_path),
    ) == incomplete_manifest_paths(["/outputs/x.md"], host_root=str(tmp_path))


def test_zh_banner_when_language_is_chinese():
    assert "交付声明无法验证" in unverified_claim_text(["/outputs/a.md"], language="zh")
    assert "Unverified delivery claim" in unverified_claim_text(["/outputs/a.md"])


def test_failure_banner_is_prepended_once():
    banner = unverified_claim_text(["/outputs/a.md"])
    once = prepend_publication_failure("body text", banner)
    assert once.startswith("## Unverified delivery claim")
    assert prepend_publication_failure(once, banner) == once


# ── completion verdict: the two incidents ─────────────────────────────────────


def test_no_tool_with_deliverable_is_complete():
    """FrontierAgent PR #24: an OpenAI-compatible model ends with a plain-text
    answer after writing its file. Judging on stopped_by alone reported
    ``run incomplete`` and discarded the report."""
    assert is_complete_run(
        _NO_TOOL, complete_stop_reasons={_NO_TOOL, "max_turns"}, unverified_claims=(),
    ) is True


def test_no_tool_without_deliverable_is_incomplete():
    """ApodexHarness r4: same stop reason, but the claimed file was never
    written and the run still reported ``success``."""
    assert is_complete_run(
        _NO_TOOL,
        complete_stop_reasons={_NO_TOOL, "max_turns"},
        unverified_claims=("/outputs/synthesis.md",),
    ) is False


def test_stop_reason_outside_the_caller_set_is_incomplete():
    """Which exits count as finished is workflow policy, not a library constant."""
    assert is_complete_run("llm_error", complete_stop_reasons={_NO_TOOL}) is False
    assert is_complete_run(_NO_TOOL, complete_stop_reasons=set()) is False


def test_unverified_claims_veto_even_a_completing_stop_reason():
    assert is_complete_run(
        "max_turns",
        complete_stop_reasons={"max_turns"},
        unverified_claims=("/outputs/a.md",),
    ) is False


# ── portability ───────────────────────────────────────────────────────────────


def test_module_pulls_in_no_product_packages():
    """The whole point of extracting this is that the library owns the logic;
    importing it must not drag a consumer's packages in."""
    leaked = [
        name for name in sys.modules
        if name.split(".")[0] in {"plugins", "workflows", "miroharness", "frontier_agent"}
    ]
    assert leaked == []


def test_manifest_paths_are_validated():
    with pytest.raises(ValueError):
        normalise_output_paths(["relative/path.md"])
    with pytest.raises(ValueError):
        normalise_output_paths(["/outputs"])
    with pytest.raises(ValueError):
        normalise_output_paths(["/outputs/*.md"])
