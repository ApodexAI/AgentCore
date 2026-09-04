# Write-audit cycle and verifier boundary

AgentCore owns the product-neutral iterate-until-pass machinery:

- the write → audit → feedback → re-write loop and its termination rules;
- max-rounds, wall-clock budget with between-round estimation, and the
  missing-artifact output check;
- writer-exception and missing-output capture as synthetic audits, so a crash
  becomes a finding the next round can read rather than an aborted cycle;
- per-round persistence of the audit trail (`audit_round_{N}.json`);
- the round-observer protocol and its failure-isolated dispatch;
- the `AuditReport` / `WriterOutput` / `AuditFinding` / `CycleOutput` data
  contracts and their JSON round-trip;
- the `Verifier` / `Generator` protocols, the `Verdict` model, and the six
  composers (Pipeline, Ensemble, Fallback, Cascade, Parallel,
  ConsensusVerifier), each of which implements `Verifier` so they nest;
- the `AuditReport` ↔ `Verdict` bridge, so a `Verifier` can be used wherever a
  `CycleAuditor` is expected.

## What the product owns

Concrete writers and auditors. That is the whole point of the split: the cycle
knows how many rounds are left and whether an artifact exists, and knows
nothing about what makes an artifact good.

- **Scoring vocabulary.** `AuditFinding.category` is a free-form `str` on
  purpose, not an enum — the useful vocabulary differs between a paper, a
  patch, a dataset and a plan, and freezing one here would force every product
  to translate into someone else's taxonomy.
- **Verdict words.** `AuditReport.verdict` is likewise free-form. The cycle
  compares it against the caller-supplied `terminal_verdicts` set, which
  defaults to `{"success", "abandon"}` only so that the common case needs no
  configuration.
- **Prompts, tools, and role definitions** for the writer and auditor sessions.
- **The oracle.** `GroundTruth` carries underscore-prefixed oracle fields
  (`_reference`, `_formal_spec`, `_test_cases`) that `VerifierContext` strips
  automatically when `is_runtime=True`. Core enforces the isolation; the
  product decides what the reference answer is and when it may be seen.
- **Feedback rendering**, beyond the default Markdown renderer. Core ships
  `DefaultFeedbackRenderer` and guarantees it never returns an empty string,
  because an empty feedback body silently turns a revise round into a re-roll.

## Injection seams

`SessionBackedWriter` and `SessionBackedAuditor` take `bus: Any` and call
`create_session` / `submit_task_to_session` / `collect` structurally. They work
against `agent_core.components.agent_bus`, but the annotation is deliberately
loose so a product can substitute its own session runtime without implementing
a Protocol it does not otherwise need.

`output_check` is a `Callable[[Path], list[str]]` returning the missing paths —
a plain callable rather than a policy object, because "what must exist when the
writer is done" is a per-task fact, not a per-product one.

Observers receive a live view of `history_so_far`. Every observer hook is
dispatched inside `try/except`: an observer that raises is logged and skipped,
never allowed to abort a cycle that would otherwise have produced an artifact.

## Not in this package

Claim-level verification engines, evidence calibration, debate arbitration,
benchmark judges, and cross-trajectory consensus over a specific answer shape
all stay in the product. They are verification *policies* over
product-specific subjects; only the contract they implement and the primitives
that compose them belong here.
