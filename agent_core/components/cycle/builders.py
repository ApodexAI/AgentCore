"""Session-backed default implementations of Writer / Auditor.

These wrap :class:`AgentBus` so callers don't have to repeat the
boilerplate of creating a session, submitting a task, collecting the
result, and parsing structured output. Sophisticated callers
(implementing :class:`Writer` / :class:`Auditor` directly) bypass these.

Two ready-made auditor flavours ship here:

- :class:`SessionBackedAuditor` — generic AuditReport-emitting auditor
  that expects the LLM to produce a JSON object matching the full
  :class:`~agent_core.components.cycle.types.AuditReport` schema (verdict +
  findings + summary + …).
- :class:`ScoreThresholdAuditor` — score-based grader. The LLM emits a
  short ``{"score"|"points": int, "feedback"|"explanation": str}`` JSON
  blob and the auditor maps the score against a caller-supplied
  ``pass_threshold`` to produce the verdict. Useful for rubric-based
  workflows (Olympiad-style proof grading, code review with rubric,
  generic score-out-of-N evaluators) where the auditor *is* a grader,
  not a critic that authors structured findings itself.

Invariants this module enforces:

- ``SessionBackedWriter`` creates **exactly one persistent session** on
  the first call to :meth:`write` and reuses it for every subsequent
  round (FR9). Conversation history accumulates across rounds.
- ``SessionBackedAuditor`` / ``ScoreThresholdAuditor`` create a
  **fresh session per round** with
  ``name=name_template.format(round=round_num)`` so prior-round
  reasoning does not contaminate (FR9). Auditor sessions default to
  ``tools_override=[]`` (empty tool set) — Codex-style hard restriction
  (PRD §2.5). Caller whitelists explicitly via ``tools=[...]``.
- :class:`ConcludePhaseObserver` is auto-injected into every auditor
  session so the LLM is nudged toward emitting structured output before
  turn exhaustion (FR7).
- ``llm_timeout`` is propagated through to
  ``AgentBus.create_session(llm_timeout=...)`` (FR8).
"""
from __future__ import annotations

import asyncio
import collections
import json
import logging
import statistics
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast

from agent_core.components.cycle.protocols import CycleAuditor
from agent_core.components.cycle.types import (
    AuditFinding,
    AuditReport,
    WriterOutput,
)
from agent_core.components.observers.conclude_phase_observer import ConcludePhaseObserver
from agent_core.tool import Tool

logger = logging.getLogger(__name__)


# Default revision instruction appended after the rendered feedback on
# rounds > 0. Generic; override via ``revision_instruction``.
_DEFAULT_REVISION_INSTRUCTION = (
    "Revise the artifact addressing every finding above. Maintain "
    "everything that was already correct. Do not start over from "
    "scratch unless the feedback explicitly requires it."
)

# Default audit prompt envelope; the auditor's ``system_prompt`` is
# expected to contain the AuditReport JSON schema. This template only
# wraps the writer's output and a final reminder. Override via
# ``audit_prompt_template``.
_DEFAULT_AUDIT_PROMPT_TEMPLATE = (
    "The writer produced this artifact for round {round}.\n\n"
    "---\n"
    "{writer_content}\n"
    "---\n\n"
    "Files reported produced this round: {files}\n\n"
    "Audit per your instructions and emit your AuditReport JSON now."
)


# ── SessionBackedWriter ─────────────────────────────────────────────────────


class SessionBackedWriter:
    """:class:`Writer` backed by a persistent ``AgentBus`` session.

    The writer's ``initial_prompt`` is submitted on round 0. On
    subsequent rounds, the rendered ``feedback_md`` is submitted as the
    next task on the same session, with the ``revision_instruction``
    appended. Because the session is persistent, the LLM sees the entire
    conversation (initial task + every prior round's feedback + every
    prior round's writer output) when deciding what to revise.

    Parameters
    ----------
    bus
        The shared :class:`AgentBus` instance.
    task_id
        Parent task id. Sub-agent sessions live under this scope.
    role_id
        Stable role identifier registered in :class:`AgentRegistry` (or
        a free-form string when ``llm_override`` + ``tools`` are passed
        explicitly).
    name
        Session name. Combined with ``task_id`` to form
        ``session_id`` (idempotent reuse — see ``AgentBus.create_session``).
    system_prompt
        Writer's persistent system prompt. Defines what the writer is /
        does and how it should respond to feedback.
    initial_prompt
        The task description submitted on round 0. The "what to write".
    tools
        Tools the writer may call. Default ``None`` → resolved by
        ResourceManager from ``role_id``.
    max_turns
        Per-task turn budget. Default 80.
    llm_timeout
        Per-LLM-call timeout (seconds). Default 300. Propagated to
        ``AgentBus.create_session``.
    work_dir
        Filesystem root for artifacts. Stored on the writer so the
        cycle can borrow ``output_check``.
    output_check
        ``(work_dir) -> list[str]`` returning missing required artifacts.
        Stored on the writer so the cycle can read it as
        ``writer.output_check``. The writer itself does not call this.
    llm_override
        Optional alternate LLM. Default ``None`` → resolved by
        ResourceManager from ``role_id``.
    observers_factory
        Optional ``() -> list[observer]`` returning observers to attach
        to every round's task. Each round calls this fresh, so observers
        with internal state get a clean instance per round.
    revision_instruction
        Text appended after ``feedback_md`` on rounds > 0. Default is
        a generic "revise addressing every finding" message.
    timeout
        ``collect`` timeout per round. Default 600s.
    """

    def __init__(
        self,
        *,
        bus: Any,
        task_id: str,
        role_id: str,
        name: str,
        system_prompt: str,
        initial_prompt: str,
        work_dir: Path | str,
        output_check: Callable[[Path], list[str]],
        tools: list[Tool] | None = None,
        max_turns: int = 80,
        llm_timeout: int = 300,
        llm_override: Any = None,
        observers_factory: Callable[[], list[Any]] | None = None,
        revision_instruction: str = _DEFAULT_REVISION_INSTRUCTION,
        timeout: float = 600.0,
    ) -> None:
        self._bus = bus
        self._task_id = task_id
        self.role_id = role_id
        self._name = name
        self._system_prompt = system_prompt
        self._initial_prompt = initial_prompt
        self._tools = tools
        self._max_turns = max_turns
        self._llm_timeout = llm_timeout
        self._llm_override = llm_override
        self._observers_factory = observers_factory
        self._revision_instruction = revision_instruction
        self._timeout = timeout

        self.work_dir = Path(work_dir)
        self.output_check = output_check

        self._session_id: str | None = None

    async def _ensure_session(self) -> str:
        if self._session_id is None:
            # Bind through a declared local first. ``self._bus`` is duck-typed,
            # so the await yields ``Any``; assigning it straight to the
            # ``str | None`` attribute left the declared ``-> str`` return
            # unprovable, which is what the type checker objected to.
            session_id: str = await self._bus.create_session(
                task_id=self._task_id,
                name=self._name,
                role_id=self.role_id,
                system_prompt=self._system_prompt,
                tools_override=self._tools,
                llm_override=self._llm_override,
                max_turns=self._max_turns,
                llm_timeout=self._llm_timeout,
            )
            self._session_id = session_id
        return self._session_id

    async def generate(
        self,
        prev_audit: Any,
        round_num: int,
        feedback_md: str,
    ) -> WriterOutput:
        del prev_audit  # the Protocol passes it; the renderer already used it
        sid = await self._ensure_session()

        if round_num == 0:
            prompt = self._initial_prompt
        else:
            prompt = (
                f"{feedback_md}\n\n{self._revision_instruction}"
                if feedback_md
                else self._revision_instruction
            )

        observers = (
            self._observers_factory() if self._observers_factory else []
        )
        job_id = await self._bus.submit_task_to_session(
            sid, prompt, observers=observers,
        )
        cr = await self._bus.collect([job_id], timeout=self._timeout)

        sub_result = _extract_one_result(cr, label=f"writer round {round_num}")
        return WriterOutput(
            content=sub_result.final_content,
            files=list(sub_result.metadata.get("files", []) or []),
            message_count=int(sub_result.metadata.get("message_count", 0) or 0),
            metadata=dict(sub_result.metadata),
            loop_result=sub_result,
        )


# ── SessionBackedAuditor ────────────────────────────────────────────────────


class SessionBackedAuditor:
    """:class:`Auditor` backed by a fresh ``AgentBus`` session per round.

    Defaults to ``tools_override=[]`` — auditors run with no tools at
    all. The auditor inspects the writer's output as text inside its
    prompt; it does not rummage. Caller whitelists tools explicitly via
    ``tools=[read_text, grep, ...]`` when read-only inspection is
    required.

    Auto-injects :class:`ConcludePhaseObserver` so the auditor reliably
    emits its structured output before turn exhaustion (FR7).

    Parameters
    ----------
    bus
        The shared :class:`AgentBus`.
    task_id
        Parent task id.
    role_id
        Stable role identifier.
    name_template
        Format string with one ``{round}`` placeholder for fresh per-round
        session names. Default ``"auditor_R{round}"``.
    system_prompt
        Auditor's system prompt. **Should** include the AuditReport
        JSON schema so the auditor knows what to emit.
    tools
        Whitelisted tool subset. Default ``[]`` (Codex-style hard
        restriction). Pass ``None`` to fall back to ResourceManager —
        only do this when the auditor genuinely needs the role's full
        tool set.
    max_turns
        Per-task turn budget. Default 40 (auditors should be fast).
    llm_timeout
        Per-LLM-call timeout. Default 300.
    conclude_ratio
        ConcludePhaseObserver threshold. Default 0.8.
    llm_override
        Optional alternate LLM (e.g. a stronger reasoning model just for
        review — Codex's ``review_model`` pattern).
    observers_factory
        Optional extra observers per round.
    audit_prompt_template
        Prompt envelope wrapping the writer's output. Override when the
        default phrasing doesn't suit.
    timeout
        ``collect`` timeout per round.
    """

    def __init__(
        self,
        *,
        bus: Any,
        task_id: str,
        role_id: str,
        system_prompt: str,
        name_template: str = "auditor_R{round}",
        tools: list[Tool] | None = None,
        max_turns: int = 40,
        llm_timeout: int = 300,
        conclude_ratio: float = 0.8,
        llm_override: Any = None,
        observers_factory: Callable[[], list[Any]] | None = None,
        audit_prompt_template: str = _DEFAULT_AUDIT_PROMPT_TEMPLATE,
        timeout: float = 600.0,
    ) -> None:
        self._bus = bus
        self._task_id = task_id
        self.role_id = role_id
        self._system_prompt = system_prompt
        self._name_template = name_template
        # Default to empty tool set — Codex-style hard restriction.
        # An explicit ``None`` is the (rare) opt-in to ResourceManager
        # fallback. Sentinel _DEFAULT to disambiguate.
        self._tools: list[Tool] = list(tools) if tools is not None else []
        self._max_turns = max_turns
        self._llm_timeout = llm_timeout
        self._conclude_ratio = conclude_ratio
        self._llm_override = llm_override
        self._observers_factory = observers_factory
        self._audit_prompt_template = audit_prompt_template
        self._timeout = timeout

    def _build_observers(self) -> list[Any]:
        observers: list[Any] = [
            ConcludePhaseObserver(conclude_ratio=self._conclude_ratio),
        ]
        if self._observers_factory:
            extra = self._observers_factory()
            if extra:
                observers.extend(extra)
        return observers

    def _build_prompt(
        self, writer_output: WriterOutput, round_num: int,
    ) -> str:
        files_text = (
            ", ".join(writer_output.files)
            if writer_output.files
            else "(none reported)"
        )
        return self._audit_prompt_template.format(
            round=round_num,
            writer_content=writer_output.content,
            files=files_text,
        )

    async def verify(
        self,
        writer_output: WriterOutput,
        round_num: int,
    ) -> AuditReport:
        name = self._name_template.format(round=round_num)
        sid = await self._bus.create_session(
            task_id=self._task_id,
            name=name,
            role_id=self.role_id,
            system_prompt=self._system_prompt,
            tools_override=self._tools,  # default [] — empty tool set
            llm_override=self._llm_override,
            max_turns=self._max_turns,
            llm_timeout=self._llm_timeout,
        )
        observers = self._build_observers()
        prompt = self._build_prompt(writer_output, round_num)
        job_id = await self._bus.submit_task_to_session(
            sid, prompt, observers=observers,
        )
        cr = await self._bus.collect([job_id], timeout=self._timeout)

        try:
            sub_result = _extract_one_result(
                cr, label=f"auditor round {round_num}",
            )
        except _CycleRuntimeError as exc:
            return _runtime_error_audit(str(exc))

        return _parse_audit_report(
            sub_result.final_content,
            metadata={
                "auditor_session": sid,
                "auditor_role": self.role_id,
                "round": round_num,
            },
        )


# ── Helpers ─────────────────────────────────────────────────────────────────


class _CycleRuntimeError(RuntimeError):
    """Internal — raised when a session task neither completes nor fails."""


def _extract_one_result(cr: Any, *, label: str) -> Any:
    if cr.completed:
        return cr.completed[0]
    if cr.failed:
        return cr.failed[0]
    raise _CycleRuntimeError(
        f"{label}: AgentBus.collect returned no result (timeout / pending)",
    )


def _runtime_error_audit(message: str) -> AuditReport:
    return AuditReport(
        verdict="iterate",
        findings=[
            AuditFinding(
                category="auditor_runtime_error",
                severity="error",
                short_message=message[:200],
                detailed_message=message,
                suggested_action=(
                    "Inspect AgentBus / session state. The cycle treats "
                    "this as iterate so it can retry on the next round."
                ),
            ),
        ],
        summary=f"Auditor session did not return a result: {message}",
        confidence=1.0,
        metadata={"synthetic": True, "reason": "auditor_runtime_error"},
    )


def _parse_audit_report(
    raw: str, *, metadata: dict[str, Any] | None = None,
) -> AuditReport:
    """Extract AuditReport JSON from the auditor's free-form response.

    Looks for a fenced ``json`` block first, then for the largest
    ``{...}`` substring. On failure, returns a structured
    ``auditor_parse_failure`` audit so the cycle can iterate rather
    than hang. The auditor's full raw text is preserved in
    ``raw_text`` for debugging.
    """
    data, parse_error = _try_parse_json(raw)
    if data is None:
        return AuditReport(
            verdict="iterate",
            findings=[
                AuditFinding(
                    category="auditor_parse_failure",
                    severity="error",
                    short_message=(
                        "Auditor did not emit parseable AuditReport JSON"
                    ),
                    detailed_message=parse_error or "no json found in response",
                    suggested_action=(
                        "Inspect auditor's system prompt — it must "
                        "instruct the LLM to emit structured AuditReport "
                        "JSON."
                    ),
                ),
            ],
            summary=(
                "The auditor's response did not contain parseable JSON. "
                "Treating verdict as iterate so the cycle can retry."
            ),
            raw_text=raw,
            confidence=1.0,
            metadata={
                "synthetic": True,
                "reason": "auditor_parse_failure",
                **(metadata or {}),
            },
        )

    try:
        report = AuditReport.from_dict(data)
        report.raw_text = raw
        if metadata:
            report.metadata = {**report.metadata, **metadata}
        return report
    except (KeyError, TypeError, ValueError) as exc:
        return AuditReport(
            verdict="iterate",
            findings=[
                AuditFinding(
                    category="auditor_schema_mismatch",
                    severity="error",
                    short_message=(
                        f"Auditor JSON did not match AuditReport schema: "
                        f"{exc!s}"[:200]
                    ),
                    detailed_message=str(exc),
                ),
            ],
            summary=(
                "The auditor's JSON parsed but did not match the "
                "AuditReport schema. Treating as iterate."
            ),
            raw_text=raw,
            confidence=1.0,
            metadata={
                "synthetic": True,
                "reason": "auditor_schema_mismatch",
                **(metadata or {}),
            },
        )


def _try_parse_json(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    """Best-effort JSON extraction. Returns (data, error_message)."""
    if not raw:
        return None, "empty response"

    # Prefer a fenced ```json ... ``` block.
    fenced = _extract_fenced_json(raw)
    if fenced is not None:
        try:
            data = json.loads(fenced)
        except json.JSONDecodeError as exc:
            return None, f"fenced block did not parse: {exc!s}"
        if isinstance(data, dict):
            return cast("dict[str, Any]", data), None
        return None, "fenced JSON was not an object"

    # Fall back to the widest {...} substring.
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return None, "no JSON object found"
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        return None, f"bare JSON did not parse: {exc!s}"
    if isinstance(data, dict):
        return cast("dict[str, Any]", data), None
    return None, "JSON was not an object"


def _extract_fenced_json(raw: str) -> str | None:
    marker = "```json"
    idx = raw.find(marker)
    if idx < 0:
        return None
    rest = raw[idx + len(marker):]
    end = rest.find("```")
    if end < 0:
        return None
    return rest[:end].strip()


# ── ScoreThresholdAuditor ───────────────────────────────────────────────────


# Default user-prompt envelope for the grader. Kept generic — domain
# context (problem statement, ground-truth, rubric) is the caller's job
# to compose into either ``system_prompt`` or ``audit_prompt_template``.
_DEFAULT_GRADE_PROMPT_TEMPLATE = (
    "Grade the following candidate solution for round {round}.\n\n"
    "---\n"
    "{writer_content}\n"
    "---\n\n"
    "Files reported produced this round: {files}\n\n"
    "Emit your grade as a single JSON object inside a ```json fenced "
    "block. Schema:\n"
    "  {{\"score\": <integer>, \"feedback\": <string>}}\n"
    "No prose outside the JSON block."
)


class ScoreThresholdAuditor:
    """:class:`Auditor` that maps a grader LLM's score to a verdict.

    Designed for the common "rubric grader" pattern where the auditor
    LLM emits a tiny JSON blob — ``{"score": 7, "feedback": "..."}`` or
    ``{"points": 6, "explanation": "..."}`` — and the cycle decides
    whether to keep iterating based on whether the score crosses
    ``pass_threshold``.

    Compared with :class:`SessionBackedAuditor`, this auditor:

    - Asks the LLM for a much simpler JSON shape (no ``findings``
      authoring required from the model).
    - Maps ``score >= pass_threshold`` → ``pass_verdict`` (default
      ``"success"``); else → ``fail_verdict`` (default ``"iterate"``).
    - Always packs the parsed ``score`` into ``AuditReport.metadata``
      (under key ``"score"``) and into a single
      ``category="grade"`` :class:`AuditFinding` so the writer's next
      round sees the score in its rendered feedback.
    - Sets ``confidence = (score - score_min) / (score_max - score_min)``
      (clamped to [0, 1]) when ``score_range`` is non-empty.

    Common patterns:

    - **Iterate-until-pass** (default): score ≥ threshold ends the cycle
      with ``success=True``. Cheaper — early-exits on a good attempt.
    - **Always-run-k** (parity with fixed-budget benchmark agents like
      MiroVerifier's IMO-GVR): pass ``pass_verdict="iterate"`` so a high
      score does *not* terminate the cycle. The cycle runs the full
      ``max_rounds`` and the workflow picks the best attempt by score
      from ``CycleOutput.history`` (use :func:`select_best_attempt`).

    Parameters
    ----------
    bus
        Shared :class:`AgentBus`.
    task_id
        Parent task id.
    role_id
        Stable role identifier.
    system_prompt
        Grader's system prompt — should describe the rubric and
        instruct the LLM to emit the score JSON. The default
        :data:`audit_prompt_template` already restates the JSON schema
        on every round; you may rely on either or both.
    name_template
        Format string with one ``{round}`` placeholder. Default
        ``"grader_R{round}"``.
    pass_threshold
        Minimum score that counts as a pass. Default ``6`` (matches the
        IMO-GVR rubric's "essentially correct" cutoff). Inclusive.
    pass_verdict
        Verdict emitted when ``score >= pass_threshold``. Default
        ``"success"``. Set to ``"iterate"`` for "always-run-k" mode.
    fail_verdict
        Verdict emitted when ``score < pass_threshold`` or the score is
        unparseable. Default ``"iterate"``.
    score_keys
        Tuple of JSON keys to try, in order, when extracting the score.
        Default ``("score", "points")`` — matches both common
        conventions (raw "score" plus IMO-GVR's "points").
    feedback_keys
        Tuple of JSON keys to try when extracting the feedback string.
        Default ``("feedback", "explanation")``.
    score_range
        ``(min, max)`` for confidence calculation. Default ``(0, 7)``
        (IMO rubric). Pass ``None`` to disable confidence derivation.
    audit_prompt_template
        Prompt envelope wrapping the writer's output. ``{round}``,
        ``{writer_content}``, ``{files}`` are substituted. Override when
        you need to inject domain context (problem, ground-truth,
        rubric) directly per-round.
    tools, max_turns, llm_timeout, conclude_ratio, llm_override,
    observers_factory, timeout
        Same semantics as :class:`SessionBackedAuditor`.
    """

    def __init__(
        self,
        *,
        bus: Any,
        task_id: str,
        role_id: str,
        system_prompt: str,
        name_template: str = "grader_R{round}",
        pass_threshold: int | float = 6,
        pass_verdict: str = "success",
        fail_verdict: str = "iterate",
        score_keys: tuple[str, ...] = ("score", "points"),
        feedback_keys: tuple[str, ...] = ("feedback", "explanation"),
        score_range: tuple[float, float] | None = (0, 7),
        audit_prompt_template: str = _DEFAULT_GRADE_PROMPT_TEMPLATE,
        tools: list[Tool] | None = None,
        max_turns: int = 8,
        llm_timeout: int = 300,
        conclude_ratio: float = 0.8,
        llm_override: Any = None,
        observers_factory: Callable[[], list[Any]] | None = None,
        timeout: float = 600.0,
    ) -> None:
        if not score_keys:
            raise ValueError("score_keys must be non-empty")
        if not feedback_keys:
            raise ValueError("feedback_keys must be non-empty")
        if score_range is not None and score_range[1] <= score_range[0]:
            raise ValueError(
                f"score_range max must exceed min, got {score_range!r}",
            )

        self._bus = bus
        self._task_id = task_id
        self.role_id = role_id
        self._system_prompt = system_prompt
        self._name_template = name_template
        self._pass_threshold = pass_threshold
        self._pass_verdict = pass_verdict
        self._fail_verdict = fail_verdict
        self._score_keys = tuple(score_keys)
        self._feedback_keys = tuple(feedback_keys)
        self._score_range = score_range
        self._audit_prompt_template = audit_prompt_template
        # Same Codex-style default as SessionBackedAuditor: empty tool
        # set unless the caller explicitly opts into ResourceManager
        # fallback by passing ``tools=None`` after construction.
        self._tools: list[Tool] = list(tools) if tools is not None else []
        self._max_turns = max_turns
        self._llm_timeout = llm_timeout
        self._conclude_ratio = conclude_ratio
        self._llm_override = llm_override
        self._observers_factory = observers_factory
        self._timeout = timeout

    def _build_observers(self) -> list[Any]:
        observers: list[Any] = [
            ConcludePhaseObserver(conclude_ratio=self._conclude_ratio),
        ]
        if self._observers_factory:
            extra = self._observers_factory()
            if extra:
                observers.extend(extra)
        return observers

    def _build_prompt(
        self, writer_output: WriterOutput, round_num: int,
    ) -> str:
        files_text = (
            ", ".join(writer_output.files)
            if writer_output.files
            else "(none reported)"
        )
        return self._audit_prompt_template.format(
            round=round_num,
            writer_content=writer_output.content,
            files=files_text,
        )

    async def verify(
        self,
        writer_output: WriterOutput,
        round_num: int,
    ) -> AuditReport:
        name = self._name_template.format(round=round_num)
        sid = await self._bus.create_session(
            task_id=self._task_id,
            name=name,
            role_id=self.role_id,
            system_prompt=self._system_prompt,
            tools_override=self._tools,
            llm_override=self._llm_override,
            max_turns=self._max_turns,
            llm_timeout=self._llm_timeout,
        )
        observers = self._build_observers()
        prompt = self._build_prompt(writer_output, round_num)
        job_id = await self._bus.submit_task_to_session(
            sid, prompt, observers=observers,
        )
        cr = await self._bus.collect([job_id], timeout=self._timeout)

        try:
            sub_result = _extract_one_result(
                cr, label=f"grader round {round_num}",
            )
        except _CycleRuntimeError as exc:
            return _runtime_error_audit(str(exc))

        return self._score_to_audit_report(
            sub_result.final_content,
            metadata={
                "auditor_session": sid,
                "auditor_role": self.role_id,
                "round": round_num,
            },
        )

    def _score_to_audit_report(
        self,
        raw: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> AuditReport:
        return parse_score_response(
            raw,
            pass_threshold=self._pass_threshold,
            pass_verdict=self._pass_verdict,
            fail_verdict=self._fail_verdict,
            score_keys=self._score_keys,
            feedback_keys=self._feedback_keys,
            score_range=self._score_range,
            metadata=metadata,
        )


def parse_score_response(
    raw: str,
    *,
    pass_threshold: int | float = 6,
    pass_verdict: str = "success",
    fail_verdict: str = "iterate",
    score_keys: tuple[str, ...] = ("score", "points"),
    feedback_keys: tuple[str, ...] = ("feedback", "explanation"),
    score_range: tuple[float, float] | None = (0, 7),
    metadata: dict[str, Any] | None = None,
) -> AuditReport:
    """Build an :class:`AuditReport` from a grader LLM's raw response.

    Public helper extracting :class:`ScoreThresholdAuditor`'s parsing
    + verdict-mapping logic. Useful when a workflow drives the grader
    via raw LLM calls (no :class:`AgentBus` session) but still wants
    the same structured output and ``WriteAuditCycle`` integration —
    the IMO-proof workflow at ``workflows/imo_proof/`` is the canonical
    consumer.

    Behaviour:

    - JSON parses + score key present + value coerces to a number →
      verdict derived from ``pass_threshold``; one ``category="grade"``
      finding carries the score; ``metadata['score']`` populated.
    - JSON parses but no score key (or non-numeric) →
      ``grader_score_missing`` finding, verdict = ``fail_verdict``.
    - JSON did not parse →
      ``grader_parse_failure`` finding, verdict = ``fail_verdict``.

    The grader's full raw text is always preserved in
    :attr:`AuditReport.raw_text` for debugging / replay. Caller-supplied
    ``metadata`` is merged into :attr:`AuditReport.metadata` (with the
    parser's own keys — ``score``, ``feedback``, ``passed``,
    ``pass_threshold``, ``grader_data`` — taking precedence).
    """
    meta_base: dict[str, Any] = dict(metadata or {})
    data, parse_error = _try_parse_json(raw)
    if data is None:
        return AuditReport(
            verdict=fail_verdict,
            findings=[
                AuditFinding(
                    category="grader_parse_failure",
                    severity="error",
                    short_message=(
                        "Grader did not emit parseable JSON"
                    ),
                    detailed_message=parse_error or "no JSON found",
                    suggested_action=(
                        "Inspect grader system prompt — it must "
                        "instruct the LLM to emit a JSON object "
                        "with the score and feedback keys."
                    ),
                ),
            ],
            summary=(
                "The grader's response did not contain parseable "
                f"JSON. Treating verdict as {fail_verdict!r}."
            ),
            raw_text=raw,
            confidence=0.0,
            metadata={
                **meta_base,
                "synthetic": True,
                "reason": "grader_parse_failure",
            },
        )

    score = _extract_first_number(data, score_keys)
    feedback = _extract_first_str(data, feedback_keys) or ""

    if score is None:
        return AuditReport(
            verdict=fail_verdict,
            findings=[
                AuditFinding(
                    category="grader_score_missing",
                    severity="error",
                    short_message=(
                        "Grader JSON did not contain a numeric "
                        f"score under {score_keys!r}"
                    ),
                    detailed_message=(
                        f"Parsed JSON: {data!r}. Expected one of "
                        f"{score_keys!r} to be a number."
                    ),
                ),
            ],
            summary=feedback or (
                "Grader returned JSON without a numeric score."
            ),
            raw_text=raw,
            confidence=0.0,
            metadata={
                **meta_base,
                "synthetic": True,
                "reason": "grader_score_missing",
                "grader_data": data,
            },
        )

    passed = score >= pass_threshold
    verdict = pass_verdict if passed else fail_verdict
    confidence = _confidence_from_score(score, score_range)

    finding = AuditFinding(
        category="grade",
        severity="info" if passed else "warning",
        short_message=(
            f"Grader score: {score} "
            f"(threshold {pass_threshold}, "
            f"{'pass' if passed else 'fail'})"
        ),
        detailed_message=feedback,
        suggested_action=(
            "" if passed else
            "Revise the artifact to address the grader's feedback "
            "and re-submit on the next round."
        ),
        metadata={"score": score, "passed": passed},
    )

    summary_lines = [
        f"Grader score: {score} (threshold "
        f"{pass_threshold}, "
        f"{'pass' if passed else 'fail'})",
    ]
    if feedback:
        summary_lines.append("")
        summary_lines.append(feedback)
    summary = "\n".join(summary_lines)

    return AuditReport(
        verdict=verdict,
        findings=[finding],
        summary=summary,
        confidence=confidence,
        raw_text=raw,
        metadata={
            **meta_base,
            "score": score,
            "feedback": feedback,
            "passed": passed,
            "pass_threshold": pass_threshold,
            "grader_data": data,
        },
    )


# ── Score / selection helpers ───────────────────────────────────────────────


def _extract_first_number(
    data: dict[str, Any], keys: tuple[str, ...],
) -> float | None:
    """Return ``float(data[k])`` for the first ``k`` in ``keys`` whose
    value is numeric (``int`` / ``float``) or a numeric string. Returns
    ``None`` if no key yields a number."""
    for k in keys:
        if k not in data:
            continue
        v = data[k]
        if isinstance(v, bool):
            # bool is a subclass of int in Python; reject explicitly.
            continue
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v.strip())
            except ValueError:
                continue
    return None


def _extract_first_str(
    data: dict[str, Any], keys: tuple[str, ...],
) -> str | None:
    """Return the first ``k`` in ``keys`` whose value is a non-empty
    string. Returns ``None`` if no key yields a string."""
    for k in keys:
        v = data.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def _confidence_from_score(
    score: float,
    score_range: tuple[float, float] | None,
) -> float:
    """Linearly map ``score`` into ``[0, 1]`` over ``score_range``;
    clamp to the unit interval. Returns ``0.0`` when ``score_range`` is
    ``None`` (caller opted out)."""
    if score_range is None:
        return 0.0
    lo, hi = score_range
    if hi <= lo:
        return 0.0
    raw = (score - lo) / (hi - lo)
    if raw < 0.0:
        return 0.0
    if raw > 1.0:
        return 1.0
    return raw


def select_best_attempt(
    history: list[tuple[WriterOutput, AuditReport]],
    *,
    score_key: str = "score",
    default_score: float = float("-inf"),
) -> tuple[int, WriterOutput, AuditReport]:
    """Pick the highest-scoring attempt from a cycle's history.

    Companion to :class:`ScoreThresholdAuditor` for "always-run-k"
    workflows: with ``pass_verdict="iterate"`` the cycle runs the full
    ``max_rounds`` and the caller post-selects the best attempt from
    ``CycleOutput.history``.

    Selection rule mirrors MiroVerifier IMO-GVR's ``_best_by_score``:
    highest score wins; ties go to the *latest* attempt (later
    revisions are at least as good as earlier ones).

    Parameters
    ----------
    history
        ``(WriterOutput, AuditReport)`` pairs as produced by
        :class:`~agent_core.components.cycle.WriteAuditCycle`. Empty history
        raises :class:`ValueError`.
    score_key
        Key under :attr:`AuditReport.metadata` carrying the score.
        Default ``"score"`` (what :class:`ScoreThresholdAuditor`
        writes).
    default_score
        Score used when an audit's metadata lacks ``score_key``. Default
        ``-inf`` so unscored audits never beat scored ones.

    Returns
    -------
    tuple
        ``(zero_indexed_attempt_number, writer_output, audit_report)``
        for the selected attempt.
    """
    if not history:
        raise ValueError("history is empty; cannot select")

    def _key(item: tuple[int, tuple[WriterOutput, AuditReport]]) -> tuple[float, int]:
        idx, (_w, audit) = item
        raw = audit.metadata.get(score_key, default_score)
        try:
            score = float(raw)
        except (TypeError, ValueError):
            score = default_score
        return (score, idx)

    best_idx, (best_w, best_a) = max(enumerate(history), key=_key)
    return best_idx, best_w, best_a


# ── select_by_answer_consensus ──────────────────────────────────────────────


async def select_by_answer_consensus(
    history: list[tuple[WriterOutput, AuditReport]],
    *,
    equivalence: Callable[[str, str], Awaitable[bool]],
    answer_key: str = "extracted_answer",
    score_key: str = "score",
    default_score: float = float("-inf"),
) -> tuple[int, WriterOutput, AuditReport]:
    """Pick the best attempt via answer-equivalence bucketing.

    Mirrors MiroVerifier@fac1b9e:agents/imo_gvr.py:_select answer mode
    (the "bucket form" the user requested). Used by imo_proof's
    answer mode where the writer emits a final symbolic answer per
    round and we want consensus across the K sequential GVR attempts.

    Algorithm:

    1. Filter ``history`` to *valid* attempts — those where
       ``audit.metadata[answer_key]`` is a non-empty string.
    2. If none valid, fall back to :func:`select_best_attempt` over
       the full history (parity with MiroVerifier's ``_select``).
    3. Bucket valid attempts: each placed in the first existing
       bucket whose head answer it's equivalent to (per
       ``equivalence``); else opens a new bucket. ``equivalence``
       raising is treated as "not equivalent" (defensive — the
       upstream rule-based grader can throw on weird LaTeX).
    4. Sort buckets by ``(sum_of_scores, bucket_size)`` descending.
    5. Within the winning bucket, run :func:`select_best_attempt`
       (highest score; ties → latest).

    Note this is *async* (unlike :func:`select_best_attempt`) because
    realistic equivalence checks may dispatch to LLM rule-based
    graders (`is_equivalent` was async in the original).

    Parameters
    ----------
    history
        ``(WriterOutput, AuditReport)`` pairs from the cycle. Empty
        history raises :class:`ValueError`.
    equivalence
        Async callable ``(a, b) -> bool`` deciding if two answer
        strings represent the same answer.
    answer_key
        Metadata key holding the extracted answer string. Default
        ``"extracted_answer"`` (what
        :class:`workflows.imo_proof.auditor.IMOProofGrader` writes
        when ``mode="answer"``).
    score_key
        Metadata key for the per-attempt score. Default ``"score"``.
    default_score
        Score used when an audit lacks ``score_key`` (or it's
        non-numeric). Default ``-inf`` — unscored attempts never win
        a bucket selection.

    Returns
    -------
    tuple
        ``(zero_indexed_attempt_number, writer_output, audit_report)``.
    """
    if not history:
        raise ValueError("history is empty; cannot select")

    def _score_of(audit: AuditReport) -> float:
        raw = audit.metadata.get(score_key, default_score)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default_score

    valid: list[tuple[int, str]] = []
    for i, (_w, audit) in enumerate(history):
        ans = audit.metadata.get(answer_key)
        if isinstance(ans, str) and ans.strip():
            valid.append((i, ans))

    if not valid:
        return select_best_attempt(
            history, score_key=score_key, default_score=default_score
        )

    # Bucket by equivalence — each bucket holds (idx, answer) pairs;
    # the head's answer represents the bucket.
    buckets: list[list[tuple[int, str]]] = []
    for idx, ans in valid:
        placed = False
        for bucket in buckets:
            head_ans = bucket[0][1]
            if head_ans.strip() == ans.strip():
                bucket.append((idx, ans))
                placed = True
                break
            try:
                eq = await equivalence(ans, head_ans)
            except Exception:
                eq = False
            if eq:
                bucket.append((idx, ans))
                placed = True
                break
        if not placed:
            buckets.append([(idx, ans)])

    # Sort buckets by (sum_score, bucket_size) descending
    def _bucket_key(bucket: list[tuple[int, str]]) -> tuple[float, int]:
        score_sum = sum(_score_of(history[idx][1]) for idx, _ in bucket)
        return (score_sum, len(bucket))

    buckets.sort(key=_bucket_key, reverse=True)
    winner = buckets[0]

    # Within winning bucket: best-by-score (ties → latest, per
    # select_best_attempt parity).
    winner_history = [history[idx] for idx, _ in winner]
    rel_idx, best_w, best_a = select_best_attempt(
        winner_history, score_key=score_key, default_score=default_score
    )
    abs_idx = winner[rel_idx][0]
    return abs_idx, best_w, best_a


# ── MajorityVoteAuditor ─────────────────────────────────────────────────────


# Aggregator over the per-judge numeric scores. Default: median (robust
# to one outlier judge in n=3). Use ``statistics.mean`` for averaging,
# ``max`` / ``min`` for optimistic / pessimistic, or any custom callable.
ScoreAggregator = Callable[[list[float]], float]


class MajorityVoteAuditor:
    """Aggregate ``n`` independent calls of a base :class:`Auditor`.

    Wraps any caller-supplied auditor (typically a
    :class:`ScoreThresholdAuditor` or a stateless raw-LLM grader) and
    runs it ``n_votes`` times **in parallel** per round, then collapses
    the results into a single :class:`AuditReport`:

    - ``verdict`` — majority over the per-judge verdicts. Ties are
      broken in favour of the first verdict observed (i.e.
      ``Counter.most_common(1)`` semantics).
    - ``metadata['score']`` — ``score_aggregator(per_judge_scores)``;
      default :func:`statistics.median`.
    - ``confidence`` — fraction of judges that returned a parseable
      AuditReport (i.e. ``n_succeeded / n_votes``).
    - ``findings`` — one synthetic ``category="grade"`` finding
      carrying the aggregated score plus the per-judge score list.
    - ``summary`` — concatenation of every judge's free-form feedback.
    - ``raw_text`` — every judge's raw response, separated by a clear
      delimiter for debugging / replay.
    - ``metadata['individual_scores']`` / ``['individual_verdicts']``
      preserved for downstream selection / analysis.

    Reduces single-judge grader variance — particularly useful for
    rubric-based scoring where one model run can plausibly land on
    score=6 vs. score=7. With ``n_votes=3`` and median, you need two
    out of three judges to agree on the lower score for the verdict
    to flip — much more stable.

    Note: ``MajorityVoteAuditor`` and
    :class:`agent_core.components.verifier.ConsensusVerifier`
    are **not equivalent** despite both being "majority over N parallel
    runs" — they vote on different fields:

    - This class votes on ``AuditReport.verdict`` label (pass/fail).
    - ``ConsensusVerifier`` votes on ``Verdict.metadata[answer_key]``
      (extracted candidate answer).

    Both legitimately coexist; pick the one that matches the topology.
    (Earlier docstrings suggested this collapses into
    ``cycle_auditor_from_verifier(ConsensusVerifier(...))`` — that was
    wrong; see ``internal-docs/designs/2026-04-30-verifier-first-class-lite.md``
    §0.1.)

    The base auditor is responsible for its own per-call freshness
    (e.g. :class:`SessionBackedAuditor` / :class:`ScoreThresholdAuditor`
    create a fresh session per call already; for stateless raw-LLM
    auditors, repeated calls naturally produce independent samples
    when the LLM has non-zero temperature).

    Parameters
    ----------
    base
        Any :class:`Auditor` implementation. Called ``n_votes`` times
        per :meth:`audit` invocation.
    n_votes
        Number of independent base-auditor calls per round. Default
        ``3`` (cheapest non-trivial majority). Must be ≥ 1.
    score_aggregator
        Function reducing the list of per-judge numeric scores into a
        single number. Default :func:`statistics.median`. Callers may
        pass ``statistics.mean`` for averaging or a custom callable.
    role_id
        Override the auditor's role identifier. Default copies
        ``base.role_id``.

    Notes
    -----
    Per-judge exceptions are caught: if a base call raises (or
    returns a non-AuditReport), it is excluded from the aggregate
    and contributes to a lowered ``confidence``. If **every** base
    call fails, the wrapper emits a synthetic
    ``category="majority_vote_failed"`` finding and verdict
    ``"iterate"``.
    """

    role_id: str = "majority_vote_auditor"

    def __init__(
        self,
        *,
        base: CycleAuditor,
        n_votes: int = 3,
        score_aggregator: ScoreAggregator = statistics.median,
        role_id: str | None = None,
    ) -> None:
        if n_votes < 1:
            raise ValueError(
                f"n_votes must be >= 1, got {n_votes!r}",
            )
        self._base = base
        self._n = n_votes
        self._aggregator = score_aggregator
        self.role_id = role_id or base.role_id

    async def verify(
        self,
        writer_output: WriterOutput,
        round_num: int,
    ) -> AuditReport:
        results: list[Any] = await asyncio.gather(
            *[
                self._base.verify(writer_output, round_num)
                for _ in range(self._n)
            ],
            return_exceptions=True,
        )

        valid: list[AuditReport] = [
            r for r in results if isinstance(r, AuditReport)
        ]
        errors: list[BaseException] = [
            r for r in results if isinstance(r, BaseException)
        ]

        if not valid:
            error_msgs = "; ".join(
                f"{type(e).__name__}: {str(e)[:120]}" for e in errors[:5]
            )
            return AuditReport(
                verdict="iterate",
                findings=[
                    AuditFinding(
                        category="majority_vote_failed",
                        severity="error",
                        short_message=(
                            f"All {self._n} base audits failed"
                        ),
                        detailed_message=error_msgs,
                        suggested_action=(
                            "Inspect the base auditor — every parallel "
                            "call raised. The cycle treats this as "
                            "iterate so it can retry on the next round."
                        ),
                    ),
                ],
                summary=(
                    f"Majority vote failed: all {self._n} base "
                    f"auditors raised exceptions."
                ),
                confidence=0.0,
                metadata={
                    "synthetic": True,
                    "reason": "majority_vote_failed",
                    "n_votes": self._n,
                    "n_succeeded": 0,
                },
            )

        verdict_counts = collections.Counter(r.verdict for r in valid)
        majority_verdict, _ = verdict_counts.most_common(1)[0]

        per_judge_scores: list[float] = []
        for r in valid:
            score_meta = r.metadata.get("score")
            if isinstance(score_meta, (int, float)) and not isinstance(
                score_meta, bool,
            ):
                per_judge_scores.append(float(score_meta))

        agg_score: float | None
        if per_judge_scores:
            try:
                agg_score = float(self._aggregator(per_judge_scores))
            except Exception:
                logger.exception(
                    "MajorityVoteAuditor: score_aggregator raised; "
                    "falling back to median",
                )
                agg_score = float(statistics.median(per_judge_scores))
        else:
            agg_score = None

        feedbacks = [
            (r.metadata.get("feedback") or r.summary or "").strip()
            for r in valid
        ]
        summary_lines = [
            f"Majority vote of {len(valid)}/{self._n} judges:",
            f"  verdict={majority_verdict!r} score={agg_score} "
            f"(per-judge scores={per_judge_scores})",
            "",
        ]
        for i, fb in enumerate(feedbacks, start=1):
            summary_lines.append(f"=== Judge {i} ===")
            summary_lines.append(fb if fb else "(no feedback)")
            summary_lines.append("")
        summary = "\n".join(summary_lines).rstrip()

        finding = AuditFinding(
            category="grade",
            severity="info" if (
                agg_score is not None and agg_score >= 6
            ) else "warning",
            short_message=(
                f"Majority vote score: {agg_score} "
                f"({len(valid)}/{self._n} judges, "
                f"verdict={majority_verdict})"
            ),
            detailed_message=summary,
            metadata={
                "score": agg_score,
                "individual_scores": per_judge_scores,
                "individual_verdicts": [r.verdict for r in valid],
            },
        )

        raw_text = "\n\n=== JUDGE BREAK ===\n\n".join(
            r.raw_text for r in valid if r.raw_text
        )

        return AuditReport(
            verdict=majority_verdict,
            findings=[finding],
            summary=summary,
            confidence=len(valid) / self._n,
            raw_text=raw_text,
            metadata={
                "score": agg_score,
                "feedback": summary,
                "individual_scores": per_judge_scores,
                "individual_verdicts": [r.verdict for r in valid],
                "n_votes": self._n,
                "n_succeeded": len(valid),
                "n_failed": len(errors),
            },
        )
