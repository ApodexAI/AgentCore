"""Verifier protocol + supporting dataclasses.

Pure typing — no kernel / state / workflow imports. The ``is_runtime``
field on ``VerifierContext`` provides framework-level information
isolation: oracle fields on ``GroundTruth`` (reference answer, formal
spec, test cases) are stripped automatically when ``is_runtime=True``,
so the verifier physically cannot see them during business runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class GroundTruth:
    """Reference material a verifier may consult.

    Public fields are visible in both runtime and eval; underscore
    fields are oracle-only and only surface when ``is_runtime=False``.
    """

    rubric: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])
    _reference: str | None = None
    _formal_spec: str | None = None
    _test_cases: list[Any] | None = None


@dataclass
class VerifierContext:
    """Context threaded through ``Verifier.verify`` calls."""

    is_runtime: bool
    _ground_truth: GroundTruth | None = None
    call_llm: Any | None = None
    call_search: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])

    @property
    def ground_truth(self) -> GroundTruth | None:
        """Return ground truth with oracle fields stripped at runtime."""
        if self._ground_truth is None:
            return None
        if self.is_runtime:
            return GroundTruth(
                rubric=self._ground_truth.rubric,
                metadata=dict(self._ground_truth.metadata),
            )
        return self._ground_truth


@dataclass
class Finding:
    severity: str
    message: str
    location: str | None = None


def _no_sub_verdicts() -> list[Verdict]:
    """Empty default for :attr:`Verdict.sub_verdicts`.

    ``field(default_factory=list[Verdict])`` -- the spelling used for every
    other collection field here -- cannot work inside ``Verdict``'s own body,
    because ``default_factory`` is evaluated at class-creation time when the
    name does not exist yet. ``from __future__ import annotations`` keeps this
    function's return annotation lazy, so the type is still declared.
    """
    return []


@dataclass
class Verdict:
    """Unified verifier return value.

    ``sub_verdicts`` carries nested results so composers can express
    arbitrarily deep trees without bespoke types.
    """

    score: float | None = None
    passed: bool = False
    findings: list[Finding] = field(default_factory=list[Finding])
    reasoning: str = ""
    sub_verdicts: list[Verdict] = field(default_factory=_no_sub_verdicts)
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])


@runtime_checkable
class Verifier(Protocol):
    """Unified verifier contract: ``(subject, ctx) → Verdict``."""

    role_id: str

    async def verify(
        self,
        subject: Any,
        ctx: VerifierContext,
    ) -> Verdict: ...


@runtime_checkable
class Generator(Protocol):
    """Produces an artifact for one round.

    Mirrors ``cycle.Writer`` semantics: stateful across rounds (typically
    a persistent agent session). The ``prev_verdict`` argument is loosely
    typed because cycle GVR engines pass an ``AuditReport`` while pure
    Verifier flows pass a ``Verdict``; both are handled uniformly until
    the cycle engine migrates to ``Verdict`` natively.
    """

    role_id: str

    async def generate(
        self,
        prev_verdict: Any,
        round_num: int,
        feedback_md: str,
    ) -> Any: ...
