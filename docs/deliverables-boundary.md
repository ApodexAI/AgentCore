# Deliverable reconciliation boundary

`agent_core.components.deliverables` answers one question for a finished round:
**did the run actually produce what it claims?** It was extracted from
ApodexHarness `workflows/agent_team/publication.py` so every consumer shares one
definition of "delivered" instead of each inferring it from `stopped_by`.

The incident it exists for: a publisher emitted its write as a fenced bash block,
never ran the tool, and the run still reported the deliverable as shipped. The
mirror image is just as wrong — a model that answers in plain text *after*
writing its file stops on `no_tool` and was reported incomplete though it
delivered. Both are cured by judging on content identity, not the stop reason.

## What core owns

- **The three verdicts.** `delivered` = the path holds non-empty bytes whose
  `(relpath, size, sha256)` is new relative to this round's baseline; `claimed` =
  the terminal answer's prose names a `/outputs/...` file; `incomplete` = claimed
  but not delivered.
- **Baseline semantics.** `/outputs` persists across rounds, so
  `snapshot_manifest` before the publisher's lease is what separates "delivered
  this round" from "a previous round's file is still lying there". `mtime_ns` is
  recorded for diagnostics but excluded from the success identity: a watcher
  omits byte-identical history from the terminal share manifest even when it was
  touched, so a `touch` must not read as a delivery.
- **Refusals.** Zero-byte files, symlinked files, and symlinked subdirectories
  inside a declared subtree never count as published. `/outputs/scratch` is
  declared intermediate storage, so naming it is not a delivery claim, and the
  bare `/outputs` root is not a claim either — the sentences that mention it bare
  are usually the opposite of one ("no deliverable was produced").
- **Claim scanning.** Paths are matched by anything-but-whitespace, not an ASCII
  allowlist, so `/outputs/报告.pdf` is a claim rather than silence; full-width
  punctuation terminates a match because Chinese prose puts no space after it.
  Space-bearing paths are recognised only where the prose delimits them
  unambiguously (a backtick span or a markdown link target) — straight quotes
  delimit phrases, not paths. Where the scan does truncate at a space, a declared
  manifest entry outranks the guess.
- **Banners** (`publication_failure_text`, `unverified_claim_text`, English and
  Chinese) and `prepend_publication_failure`, which makes a terminal report
  incapable of claiming success over a failed publish.
- **Off-loop variants.** Every filesystem-touching entry point has an `_async`
  twin that runs in a thread. Hashing a multi-hundred-MB tree inside the event
  loop stalls every concurrent sub-agent, the heartbeat, and the event stream —
  and callers typically hold a publication lock while doing it.

## What the product owns

- **`host_root`** — where the sandbox's `/outputs` really is. A model always
  names deliverables by sandbox path; reconciliation runs in the host process.
  The two coincide when the harness runs inside the task container and differ
  under bwrap-style mounts. Unlike the product original, core has **no** mount
  resolver fallback: a library that guessed the host root would silently
  reconcile against the wrong tree. Omitting it means the sandbox path *is* the
  host path.
- **`ignore_matcher`** — an optional `(relpath) -> bool` implementing the host's
  `.outputsignore` policy. Omitted means nothing is ignored. A matcher that
  raises is treated as "not ignored" rather than failing the round.
- **`complete_stop_reasons`** for `is_complete_run`. Which exits count as
  finished is workflow policy, not a library constant: a workflow whose terminal
  signal is a bare-text turn includes `no_tool`; one that requires an explicit
  finalize tool does not. Core only guarantees the conjunction — a completing
  stop reason **and** no unverified claim.
- **Quarantine and security scanning** of anything reconciliation refused to
  count. Core declines to call it published; it does not act on it.

## Injection seam

```python
from agent_core.components.deliverables import (
    reconcile_publication_state_async,
    snapshot_manifest_async,
)

baseline = await snapshot_manifest_async(manifest, host_root=host_root)
# ... publisher runs ...
failures, banner = await reconcile_publication_state_async(
    {
        "outputs_host_root": host_root,
        "deliverable_manifest": manifest,
        "manifest_baseline": baseline,
    },
    final_text=answer,
    language=language,
    ignore_matcher=outputsignore.match,
)
```

Every key of the state dict is optional. With no `deliverable_manifest` the
terminal answer's own prose is checked instead, so "reported delivered but the
manifest is empty" is caught rather than waved through — disk existence alone is
intentionally insufficient there, because nothing had authority to create a root
deliverable that round.
