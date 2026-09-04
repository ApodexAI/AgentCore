"""Deliverable reconciliation: did the run actually produce what it claims?

Extracted from ApodexHarness ``workflows/agent_team/publication.py`` (landed as
``84278ff0``, "reconcile publisher claims against real /outputs content") so
every consumer shares one definition of "delivered" instead of each guessing
from ``stopped_by``.

The problem it solves, in the original incident's words: *a publisher that
emitted its write as a fenced bash block never ran the tool, yet the run still
reported the deliverable as shipped.* The same failure has a mirror image — a
model that ends with a plain-text answer AND wrote its file gets reported as
incomplete purely because ``stopped_by == "no_tool"``. Both are cured by judging
completion on **content identity**, not on the stop reason:

* delivered  = the path holds non-empty bytes whose ``(relpath, size, sha256)``
  is new relative to this round's baseline;
* claimed    = the terminal answer's own prose names a ``/outputs/...`` file;
* incomplete = claimed but not delivered.

**Two path spaces meet here.** A model always names deliverables by their
*sandbox* path (``/outputs/report.pdf``), while reconciliation runs in the host
process, which sees the directory ``/outputs`` is bound to. They coincide when
the harness runs inside the task container and differ under bwrap-style mounts.
Every filesystem probe therefore goes through :func:`_host_path`.

**Two host-specific concerns are injected rather than imported**, which is what
lets this module live in the portable library:

* ``host_root`` — where ``/outputs`` really is. Callers that know it pass it;
  omitting it means "the sandbox path *is* the host path". This module never
  reaches into a product's mount resolver to find out.
* ``ignore_matcher`` — an optional ``(relpath) -> bool`` predicate implementing
  a host's ``.outputsignore`` policy. Omitted means nothing is ignored.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

_PUBLICATION_FAILURE_HEADING = "## Deliverable publication failed"
_PUBLICATION_FAILURE_HEADING_ZH = "## 交付物发布失败"
_UNVERIFIED_CLAIM_HEADING = "## Unverified delivery claim"
_UNVERIFIED_CLAIM_HEADING_ZH = "## 交付声明无法验证"

SANDBOX_OUTPUTS_ROOT = "/outputs"
# ``/outputs/scratch`` is the agreed cross-round intermediate area, explicitly
# NOT a deliverable.
SCRATCH_PREFIX = "/outputs/scratch"

_ALL_HEADINGS = (
    _PUBLICATION_FAILURE_HEADING,
    _PUBLICATION_FAILURE_HEADING_ZH,
    _UNVERIFIED_CLAIM_HEADING,
    _UNVERIFIED_CLAIM_HEADING_ZH,
)

# Prose reference to a deliverable path: "saved to /outputs/final.pdf",
# "`/outputs/report/index.html`". Trailing punctuation is stripped below so a
# sentence-ending period does not become part of the path.
#
# The character class is "anything but whitespace and the delimiters prose wraps
# a path in", NOT an ASCII allowlist. An allowlist truncated every non-ASCII
# name mid-path — ``/outputs/报告.pdf`` matched only the bare root and was
# discarded as "no claim", and ``/outputs/résumé.pdf`` became ``/outputs/r``.
# Both directions were wrong for the incident this module exists to catch: the
# first waved through a claim that was never delivered, and the second flagged a
# correctly delivered file as unverified because the truncated path matched no
# manifest entry.
#
# Full-width punctuation is excluded from the class rather than only stripped at
# the end: Chinese prose puts no space after it, so "见 /outputs/a.pdf，以及…"
# scanned as the path "/outputs/a.pdf，以及" when the class accepted it.
_CJK_PUNCTUATION = "，。、；：！？（）［］｛｝【】《》〈〉「」『』…～"
_OUTPUTS_CLAIM_RE = re.compile(
    r"/outputs(?:/[^\s`'\"<>|*?" + re.escape(_CJK_PUNCTUATION) + r"]+)?"
)

# A path containing spaces cannot be delimited by whitespace, so it is only
# recognised when the prose delimits it unambiguously: a backtick code span or a
# markdown link target. Straight quotes are deliberately NOT delimiters here —
# prose uses them for phrases, and "'/outputs/a.pdf, which took a while'" would
# be captured as one absurd path that then suppresses the real one. Undelimited
# spaces stay ambiguous (a filename's second word is indistinguishable from the
# sentence's next word) and are left to the whitespace-delimited scan above.
_DELIMITED_CLAIM_RE = re.compile(
    r"`\s*(/outputs/[^`\n]+?)\s*`" r"|\]\(\s*(/outputs/[^)\n]+?)\s*\)"
)

# Sentence-ending punctuation to peel off a matched path. Full-width forms are
# here because the character class above accepts non-ASCII names: "已写出
# /outputs/报告.pdf。" would otherwise yield a path ending in "。".
_CLAIM_TRAILING_PUNCTUATION = ".,;:!?)]}'\"`" + _CJK_PUNCTUATION

SnapshotEntry = tuple[str, int, int, str]

# ``(relpath) -> should_ignore``. Relpaths are posix, relative to outputs root.
IgnoreMatcher = Callable[[str], bool]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_output_path(path: str) -> str:
    raw = str(path or "").strip()
    if not raw.startswith("/outputs/"):
        raise ValueError(
            f"declared output path must be an absolute file under /outputs: {raw!r}"
        )
    normalised = os.path.normpath(raw)
    if normalised == SANDBOX_OUTPUTS_ROOT or not normalised.startswith("/outputs/"):
        raise ValueError(f"invalid declared output path: {raw!r}")
    if any(ch in normalised for ch in "*?[]{}"):
        raise ValueError(f"output path must not contain glob characters: {raw!r}")
    return normalised


def normalise_output_paths(paths: Iterable[str]) -> tuple[str, ...]:
    """Validate and de-duplicate a publishing task's output manifest."""
    result: list[str] = []
    for raw in paths:
        path = _normalise_output_path(str(raw))
        if path not in result:
            result.append(path)
    return tuple(result)


def _resolve_host_outputs_root(host_root: str | None) -> str:
    """Return the host directory that the sandbox's ``/outputs`` resolves to.

    Unlike the product original this has NO mount-resolver fallback: a library
    that guessed the host root would silently reconcile against the wrong tree.
    Omitting ``host_root`` means the sandbox path is the host path.
    """
    return str(host_root) if host_root else SANDBOX_OUTPUTS_ROOT


def _host_path(path: str, host_root: str | None = None) -> Path:
    """Translate a sandbox-absolute manifest entry to its host location.

    Non-``/outputs`` paths pass through unchanged: callers that already hold a
    host path must not be rewritten.
    """
    root = _resolve_host_outputs_root(host_root)
    if path == SANDBOX_OUTPUTS_ROOT or path.startswith(SANDBOX_OUTPUTS_ROOT + "/"):
        normalised = os.path.normpath(path)
        if normalised != SANDBOX_OUTPUTS_ROOT and not normalised.startswith(
            SANDBOX_OUTPUTS_ROOT + "/"
        ):
            raise ValueError(f"output path escapes /outputs: {path!r}")
        path = normalised
    if not root or root == SANDBOX_OUTPUTS_ROOT:
        return Path(path)
    if path == SANDBOX_OUTPUTS_ROOT:
        return Path(root)
    prefix = SANDBOX_OUTPUTS_ROOT + "/"
    if path.startswith(prefix):
        return Path(root) / path[len(prefix):]
    return Path(path)


def _path_snapshot(
    path: str,
    host_root: str | None = None,
    *,
    ignore_matcher: IgnoreMatcher | None = None,
) -> tuple[SnapshotEntry, ...]:
    """Return a watcher-compatible identity for one manifest entry.

    Zero-byte files are omitted deliberately: ``touch /outputs/final.pdf``
    creates a path that exists but delivers nothing, and an empty snapshot is
    exactly how the caller expresses "nothing was published here".
    """
    try:
        root = _host_path(path, host_root)
        outputs_root = Path(_resolve_host_outputs_root(host_root))

        def _ignored(candidate: Path) -> bool:
            if ignore_matcher is None:
                return False
            try:
                rel = candidate.relative_to(outputs_root).as_posix()
            except ValueError:
                return False
            try:
                return bool(ignore_matcher(rel))
            except Exception:
                return False

        if _ignored(root):
            return ()
        if root.is_symlink() or not root.exists():
            return ()
        if root.is_file():
            stat = root.stat()
            if stat.st_size <= 0:
                return ()
            return ((".", int(stat.st_size), int(stat.st_mtime_ns), _sha256(root)),)
        if not root.is_dir():
            return ()
    except (OSError, ValueError):
        return ()

    entries: list[SnapshotEntry] = []
    try:
        for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
            # A symlinked directory can redirect the manifest outside its
            # declared subtree. Security scanning handles quarantine elsewhere;
            # reconciliation simply refuses to count it as published.
            dirnames[:] = [
                name
                for name in dirnames
                if not Path(dirpath, name).is_symlink() and not _ignored(Path(dirpath, name))
            ]
            for name in filenames:
                candidate = Path(dirpath, name)
                if candidate.is_symlink() or _ignored(candidate):
                    continue
                try:
                    stat = candidate.stat()
                except OSError:
                    continue
                if stat.st_size <= 0:
                    continue
                relpath = str(candidate.relative_to(root))
                try:
                    digest = _sha256(candidate)
                except OSError:
                    continue
                entries.append((relpath, int(stat.st_size), int(stat.st_mtime_ns), digest))
    except OSError:
        return ()
    return tuple(sorted(entries))


def snapshot_manifest(
    paths: Iterable[str],
    *,
    host_root: str | None = None,
    ignore_matcher: IgnoreMatcher | None = None,
) -> dict[str, tuple[SnapshotEntry, ...]]:
    """Capture manifest state before a publisher first receives its lease."""
    return {
        path: _path_snapshot(path, host_root, ignore_matcher=ignore_matcher)
        for path in normalise_output_paths(paths)
    }


async def snapshot_manifest_async(
    paths: Iterable[str],
    *,
    host_root: str | None = None,
    ignore_matcher: IgnoreMatcher | None = None,
) -> dict[str, tuple[SnapshotEntry, ...]]:
    """Off-loop :func:`snapshot_manifest`.

    Hashing a multi-hundred-MB deliverable tree synchronously inside the event
    loop stalls every concurrent sub-agent, the heartbeat and the event stream —
    and callers typically hold a publication lock while doing it.
    """
    return await asyncio.to_thread(
        snapshot_manifest, list(paths), host_root=host_root, ignore_matcher=ignore_matcher,
    )


def incomplete_manifest_paths(
    paths: Iterable[str],
    baseline: dict[str, Any] | None = None,
    *,
    host_root: str | None = None,
    ignore_matcher: IgnoreMatcher | None = None,
) -> tuple[str, ...]:
    """Return manifest entries that hold no deliverable content.

    An entry counts as published when it holds a non-empty snapshot AND either

    * its run baseline was empty — the publisher genuinely created it. This is
      the case the original bug was about: manifest declared, file never
      written, prose claiming success.
    * or the snapshot gained content the baseline did not have, comparing
      ``(relpath, size, sha256)`` exactly like a filesystem watcher would.

    ``/outputs`` persists across rounds, so the baseline is what separates "the
    publisher delivered this round" from "a previous round's file is still lying
    there". ``mtime_ns`` remains in the snapshot for diagnostics but is
    deliberately excluded from the success identity: a watcher omits
    byte-identical mounted history from the terminal share manifest even when it
    was touched, and reconciliation must not call such a path delivered while
    the protocol publishes no downloadable file.

    Deletions alone never count as publishing: a snapshot that only *lost*
    entries relative to the baseline yields no new identity.
    """
    baseline = baseline or {}
    missing: list[str] = []
    for path in normalise_output_paths(paths):
        current = _path_snapshot(path, host_root, ignore_matcher=ignore_matcher)
        if not current:
            missing.append(path)
            continue
        previous = {
            (str(item[0]), int(item[1]), str(item[3]))
            for item in (baseline.get(path) or ())
            if len(item) >= 4
        }
        if not previous:
            continue
        current_content = {(str(item[0]), int(item[1]), str(item[3])) for item in current}
        if not current_content - previous:
            missing.append(path)
    return tuple(missing)


async def incomplete_manifest_paths_async(
    paths: Iterable[str],
    baseline: dict[str, Any] | None = None,
    *,
    host_root: str | None = None,
    ignore_matcher: IgnoreMatcher | None = None,
) -> tuple[str, ...]:
    """Off-loop :func:`incomplete_manifest_paths` (see snapshot_manifest_async)."""
    return await asyncio.to_thread(
        incomplete_manifest_paths,
        list(paths),
        baseline,
        host_root=host_root,
        ignore_matcher=ignore_matcher,
    )


def claimed_output_paths(text: str) -> tuple[str, ...]:
    """Extract deliverable paths a terminal answer claims to have written.

    ``/outputs/scratch`` is excluded: it is declared intermediate storage, so
    naming it is not a delivery claim.

    The bare directory is excluded too. ``/outputs`` names no file, so it cannot
    be a claim that one was written — and the sentences that mention it bare are
    usually the OPPOSITE of a claim ("no deliverable was produced; /outputs is
    empty"). Counting it inverted the verdict on exactly the runs that correctly
    delivered nothing: banner shown, status flipped from not-requested to
    incomplete.
    """
    body = text or ""
    found: list[str] = []
    # Delimited paths first: a quoted "/outputs/final report.pdf" must be taken
    # whole, before the whitespace-delimited scan reduces it to "/outputs/final".
    delimited = [
        group for match in _DELIMITED_CLAIM_RE.finditer(body) for group in match.groups() if group
    ]
    candidates = [*delimited, *_OUTPUTS_CLAIM_RE.findall(body)]
    for raw in candidates:
        path = os.path.normpath(raw.rstrip(_CLAIM_TRAILING_PUNCTUATION))
        # normpath also collapses ``/outputs/../x``, which lands outside the
        # root and is therefore not a deliverable claim either.
        if not path.startswith(SANDBOX_OUTPUTS_ROOT + "/"):
            continue
        if path == SCRATCH_PREFIX or path.startswith(SCRATCH_PREFIX + "/"):
            continue
        if path not in found:
            found.append(path)
    # The whitespace-delimited scan also yields the truncated head of every
    # space-bearing path the delimited scan already captured whole
    # ("/outputs/final" beside "/outputs/final report.pdf"). Drop a candidate
    # that another claim continues at a non-separator boundary: that is only
    # ever a truncation artifact, while "/outputs/a" beside "/outputs/a/b.md"
    # continues at "/" and stays two genuine claims.
    return tuple(
        path
        for path in found
        if not any(
            other != path and other.startswith(path) and not other[len(path):].startswith("/")
            for other in found
        )
    )


def unverified_claim_paths(
    text: str,
    *,
    manifest: Iterable[str] = (),
    host_root: str | None = None,
) -> tuple[str, ...]:
    """Return claims not covered by this round's declared manifest.

    This is the no-manifest counterpart to :func:`incomplete_manifest_paths`. A
    round can end with an empty manifest and still claim delivery — no publisher
    was ever dispatched, or the publish call was refused — yet the prose says
    "saved to /outputs/x.pdf".

    Disk existence alone is intentionally insufficient. ``/outputs`` is mounted
    across rounds and a watcher excludes unchanged history from the terminal
    share manifest. With no manifest, nothing had authority to create a root
    deliverable this round, so every delivery claim is unverified even when a
    stale same-named file exists.
    """
    declared = normalise_output_paths(manifest)

    def _covered(path: str) -> bool:
        # Unreachable via ``claimed_output_paths`` (the bare root is filtered
        # there), kept so a future widening of the claim scanner degrades to
        # "covered whenever anything is declared" instead of flagging the root.
        if path == SANDBOX_OUTPUTS_ROOT:
            return bool(declared)
        for entry in declared:
            if path == entry:
                return True
            # A claim whose text is the truncated head of a declared entry — an
            # undelimited "/outputs/final report.pdf" scans as "/outputs/final"
            # — is covered by that entry. The manifest is authoritative here, so
            # believing the scanner's truncation over it would flag a correctly
            # delivered file as an unverified claim.
            if entry.startswith(path) and not entry[len(path):].startswith("/"):
                return True
            if path.startswith(entry + "/"):
                try:
                    if _host_path(entry, host_root).is_dir():
                        return True
                except (OSError, ValueError):
                    continue
        return False

    return tuple(path for path in claimed_output_paths(text) if not _covered(path))


def publication_failure_text(
    manifest: Iterable[str],
    incomplete: Iterable[str],
    *,
    language: str = "",
) -> str:
    """Banner for a declared manifest that was not fully published."""
    declared = normalise_output_paths(manifest)
    missing = normalise_output_paths(incomplete)
    declared_text = ", ".join(f"`{path}`" for path in declared) or "(none)"
    missing_text = ", ".join(f"`{path}`" for path in missing) or "(none)"
    if str(language or "").lower().startswith("zh"):
        return (
            f"{_PUBLICATION_FAILURE_HEADING_ZH}\n\n"
            "本轮虽声明了交付 manifest，但未实际发布全部条目。请勿将 workspace "
            "或 `/outputs/scratch/` 路径视为可下载交付物。\n\n"
            f"- 声明的 manifest：{declared_text}\n"
            f"- 缺失、为空或本轮未更新：{missing_text}"
        )
    return (
        f"{_PUBLICATION_FAILURE_HEADING}\n\n"
        "The run declared a deliverable manifest but did not publish every "
        "entry during this round. Do not treat workspace or "
        "`/outputs/scratch/` paths as downloadable deliverables.\n\n"
        f"- Declared manifest: {declared_text}\n"
        f"- Missing, empty or not updated this round: {missing_text}"
    )


def unverified_claim_text(
    unverified: Iterable[str],
    *,
    language: str = "",
) -> str:
    """Banner for a delivery claim that never went through the publish flow."""
    paths = tuple(dict.fromkeys(str(path) for path in unverified))
    if not paths:
        return ""
    paths_text = ", ".join(f"`{path}`" for path in paths)
    if str(language or "").lower().startswith("zh"):
        return (
            f"{_UNVERIFIED_CLAIM_HEADING_ZH}\n\n"
            "答案声称已写出下列交付物，但这些路径未被本轮发布 manifest 覆盖。"
            "磁盘上存在同名历史文件也不代表它会进入本轮下载列表。\n\n"
            f"- 声称但未验证：{paths_text}"
        )
    return (
        f"{_UNVERIFIED_CLAIM_HEADING}\n\n"
        "The answer claims the following deliverables were written, but no "
        "publishing manifest covers them for this round. A same-named historical "
        "file on disk does not put it in this round's download list.\n\n"
        f"- Claimed but unverified: {paths_text}"
    )


def reconcile_publication_state(
    state: dict[str, Any],
    *,
    language: str = "",
    final_text: str = "",
    ignore_matcher: IgnoreMatcher | None = None,
) -> tuple[tuple[str, ...], str]:
    """Inspect a runtime publication state and return ``(incomplete, message)``.

    With a declared manifest the manifest is authoritative. Without one, the
    terminal answer's own prose is checked for delivery claims, so "reported
    delivered but the manifest is empty" is caught rather than waved through.

    Reads ``outputs_host_root``, ``deliverable_manifest`` and
    ``manifest_baseline`` from ``state``; every key is optional.
    """
    host_root = str(state.get("outputs_host_root") or "") or None
    manifest = tuple(state.get("deliverable_manifest") or ())
    if not manifest:
        unverified = unverified_claim_paths(final_text, host_root=host_root)
        if not unverified:
            return (), ""
        return unverified, unverified_claim_text(unverified, language=language)
    incomplete = incomplete_manifest_paths(
        manifest,
        state.get("manifest_baseline"),
        host_root=host_root,
        ignore_matcher=ignore_matcher,
    )
    unverified = unverified_claim_paths(final_text, manifest=manifest, host_root=host_root)
    failures = tuple(dict.fromkeys((*incomplete, *unverified)))
    messages: list[str] = []
    if incomplete:
        messages.append(publication_failure_text(manifest, incomplete, language=language))
    if unverified:
        messages.append(unverified_claim_text(unverified, language=language))
    return failures, "\n\n".join(messages)


async def reconcile_publication_state_async(
    state: dict[str, Any],
    *,
    language: str = "",
    final_text: str = "",
    ignore_matcher: IgnoreMatcher | None = None,
) -> tuple[tuple[str, ...], str]:
    """Off-loop :func:`reconcile_publication_state`."""
    return await asyncio.to_thread(
        reconcile_publication_state,
        state,
        language=language,
        final_text=final_text,
        ignore_matcher=ignore_matcher,
    )


def prepend_publication_failure(report: str, failure: str) -> str:
    """Make a terminal report incapable of claiming success over a failed publish."""
    body = (report or "").strip()
    if not failure or body.startswith(_ALL_HEADINGS):
        return body
    return f"{failure}\n\n---\n\n{body}" if body else failure


def is_complete_run(
    stopped_by: str,
    *,
    complete_stop_reasons: Iterable[str],
    unverified_claims: Iterable[str] = (),
) -> bool:
    """Whether a finished run should be reported as complete.

    Completion is the conjunction of two independent facts:

    * the loop ended on a stop reason the caller counts as completing, and
    * the answer made good on every deliverable it claimed.

    Splitting them is the point. Judging on ``stopped_by`` alone mis-reports in
    BOTH directions, as two real incidents showed:

    * a model that answers in plain text after writing its file stops on
      ``no_tool`` and was reported incomplete though it delivered;
    * a model that answers in plain text WITHOUT writing the file stops on the
      same reason and was reported successful though it delivered nothing.

    ``complete_stop_reasons`` stays a caller decision because "which exits count
    as finished" is workflow policy, not a library constant — a workflow whose
    terminal signal is a bare-text turn includes ``no_tool``; one that requires
    an explicit finalize tool does not.
    """
    if tuple(unverified_claims):
        return False
    return str(stopped_by or "") in frozenset(str(r) for r in complete_stop_reasons)


__all__ = [
    "SANDBOX_OUTPUTS_ROOT",
    "SCRATCH_PREFIX",
    "IgnoreMatcher",
    "SnapshotEntry",
    "claimed_output_paths",
    "incomplete_manifest_paths",
    "incomplete_manifest_paths_async",
    "is_complete_run",
    "normalise_output_paths",
    "prepend_publication_failure",
    "publication_failure_text",
    "reconcile_publication_state",
    "reconcile_publication_state_async",
    "snapshot_manifest",
    "snapshot_manifest_async",
    "unverified_claim_paths",
    "unverified_claim_text",
]
