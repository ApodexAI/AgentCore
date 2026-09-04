"""Default FeedbackRenderer — turns an AuditReport into the next-round prompt.

The renderer is deliberately verbose: it emits both a Markdown table
(human-readable) and a JSON detail block (machine-readable) in the same
prompt segment, plus the auditor's free-form ``summary`` paragraph
verbatim above. This lets downstream LLMs read either form depending on
their tool / parsing capability, and gives humans inspecting the trace a
readable artifact.

Design notes:

- Empty findings + non-terminal verdict produces an explicit
  "re-attempt" message rather than an empty string (FR6). Returning ""
  would make the writer think the audit was successful and the writer's
  prior round was fine, when in fact the auditor failed to articulate
  what went wrong.
- ``raw_text`` is **not** included in the output. It is for debugging
  only — including it would spam the writer's context with the
  auditor's chain-of-thought / probing.
- The output is plain text (Markdown). The cycle injects it as a
  ``HumanMessage`` content; we do not nest it in any envelope here.
"""
from __future__ import annotations

import json
from collections.abc import Iterable

from agent_core.components.cycle.types import AuditFinding, AuditReport

_HEADER = "## Audit feedback for the previous round\n"

_REATTEMPT_MESSAGE = (
    "**Audit verdict**: `{verdict}`. The auditor did not list specific "
    "findings, which means the artifact is not yet acceptable but the "
    "auditor could not pinpoint a single issue. Re-attempt the task in "
    "full — review the requirements, do not assume the prior draft is "
    "salvageable."
)

_TABLE_HEADER = (
    "| # | Severity | Category | Where | Short message | Suggested action |\n"
    "|---|---|---|---|---|---|\n"
)


class DefaultFeedbackRenderer:
    """Hybrid Markdown + JSON renderer (FR6).

    Output structure (top-to-bottom):

    1. Section header.
    2. Verdict line.
    3. Auditor's free-form ``summary`` paragraph (if present).
    4. **If findings empty + verdict not in success set** — an explicit
       "re-attempt" instruction.
    5. **Else** — Markdown summary table (one row per finding) followed
       by a fenced JSON block with the full structured findings.

    The implementation does not know which verdicts are "success" — the
    renderer is told via the ``success_verdicts`` constructor argument
    so it can choose between (4) and (5) without hardcoding domain
    vocabulary.
    """

    def __init__(
        self,
        *,
        success_verdicts: Iterable[str] = ("success",),
    ) -> None:
        self._success = frozenset(success_verdicts)

    def render(self, audit: AuditReport) -> str:
        parts: list[str] = [_HEADER]
        parts.append(f"**Verdict**: `{audit.verdict}`")
        if audit.confidence:
            parts.append(f" (auditor confidence: {audit.confidence:.2f})")
        parts.append("\n\n")

        if audit.summary.strip():
            parts.append("**Auditor summary**:\n\n")
            parts.append(audit.summary.strip())
            parts.append("\n\n")

        is_terminal_success = audit.verdict in self._success
        if not audit.findings and not is_terminal_success:
            parts.append(
                _REATTEMPT_MESSAGE.format(verdict=audit.verdict),
            )
            parts.append("\n")
            return "".join(parts)

        if audit.findings:
            parts.append("**Findings**:\n\n")
            parts.append(_TABLE_HEADER)
            for i, f in enumerate(audit.findings, start=1):
                parts.append(_render_table_row(i, f))
            parts.append("\n")

            parts.append("**Findings (structured)**:\n\n")
            parts.append("```json\n")
            parts.append(
                json.dumps(
                    [f.to_dict() for f in audit.findings],
                    indent=2,
                    ensure_ascii=False,
                ),
            )
            parts.append("\n```\n")

        return "".join(parts)


def _render_table_row(idx: int, f: AuditFinding) -> str:
    where = ""
    if f.file:
        where = f.file
        if f.line is not None:
            where = f"{f.file}:{f.line}"
    return (
        f"| {idx} | {_md_cell(f.severity)} | {_md_cell(f.category)} "
        f"| {_md_cell(where)} | {_md_cell(f.short_message)} "
        f"| {_md_cell(f.suggested_action)} |\n"
    )


def _md_cell(text: str) -> str:
    """Escape pipe characters and newlines so a single cell stays one row."""
    if not text:
        return ""
    return text.replace("|", "\\|").replace("\n", " ")
