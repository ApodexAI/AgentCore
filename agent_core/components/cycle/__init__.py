"""WriteAuditCycle — iterate-until-pass artifact production.

Generic write → audit → feedback → re-write loop over arbitrary
caller-supplied :class:`Writer` and :class:`Auditor` implementations.
The cycle owns the loop mechanics (max-rounds, wall-clock budget,
output existence check, exception capture, per-round persistence,
observer hooks); callers own the role-specific prompts, tools, and
rendering.

Quick start::

    from agent_core.components.cycle import (
        WriteAuditCycle, SessionBackedWriter, SessionBackedAuditor,
        DefaultFeedbackRenderer,
    )

    writer = SessionBackedWriter(
        bus=bus, task_id=task_id,
        role_id="paper_writer", name="paper_writer",
        system_prompt=PAPER_WRITER_PROMPT,
        initial_prompt="Write a 3-section research paper on X.",
        tools=[file_editor, read_text, web_search],
        max_turns=80, llm_timeout=300,
        work_dir=work_dir,
        output_check=lambda wd: missing_in(
            wd, ["sections/intro.md", "sections/method.md"],
        ),
    )
    auditor = SessionBackedAuditor(
        bus=bus, task_id=task_id,
        role_id="paper_auditor",
        system_prompt=PAPER_AUDITOR_PROMPT,  # must define AuditReport JSON schema
        # tools defaults to [] — Codex-style hard tool restriction.
        tools=[read_text, grep],             # whitelist read-only inspection
        max_turns=40, llm_timeout=300,
        conclude_ratio=0.8,
    )
    cycle = WriteAuditCycle(
        writer=writer, auditor=auditor,
        work_dir=work_dir,
        output_check=writer.output_check,
        max_rounds=10,
    )
    output: CycleOutput = await cycle.run()

See ``internal-docs/WRITE_AUDIT_CYCLE_REQUIREMENTS.md`` for the full spec
(FR1–FR11, NFR1–NFR7) and
``internal-docs/architecture/guide-write-audit-cycle.md`` for the user guide.
"""
from agent_core.components.cycle.builders import (
    MajorityVoteAuditor,
    ScoreThresholdAuditor,
    SessionBackedAuditor,
    SessionBackedWriter,
    parse_score_response,
    select_best_attempt,
    select_by_answer_consensus,
)
from agent_core.components.cycle.default_renderer import DefaultFeedbackRenderer
from agent_core.components.cycle.observers import (
    BaseRoundObserver,
    CycleContext,
    RoundIntervention,
    RoundObserver,
)
from agent_core.components.cycle.observers_builtin import (
    BestSoFarObserver,
    MetricsObserver,
    PlateauAbortObserver,
)
from agent_core.components.cycle.protocols import FeedbackRenderer
from agent_core.components.cycle.types import (
    AuditFinding,
    AuditReport,
    CycleOutput,
    WriterOutput,
)
from agent_core.components.cycle.write_audit_cycle import WriteAuditCycle

__all__ = [
    "AuditFinding",
    "AuditReport",
    "BaseRoundObserver",
    "BestSoFarObserver",
    "CycleContext",
    "CycleOutput",
    "DefaultFeedbackRenderer",
    "FeedbackRenderer",
    "MajorityVoteAuditor",
    "MetricsObserver",
    "PlateauAbortObserver",
    "RoundIntervention",
    "RoundObserver",
    "ScoreThresholdAuditor",
    "SessionBackedAuditor",
    "SessionBackedWriter",
    "WriteAuditCycle",
    "WriterOutput",
    "parse_score_response",
    "select_best_attempt",
    "select_by_answer_consensus",
]
