"""Unified verifier protocol + composers.

``(subject, ctx) → Verdict`` covers LLM-rubric, search-grounded,
multi-phase, debate, code-exec, and cross-trajectory consensus verify
forms. Six composers (Pipeline / Ensemble / Fallback / Cascade /
Parallel / ConsensusVerifier) each implement the Verifier protocol so
they nest freely.

This package ships only the contract and the composition primitives.
Concrete verifiers belong to the consuming product, beside the workflow
whose subject they judge: the scoring rubric, the domain vocabulary, and
the oracle are all product decisions. See ``docs/cycle-verifier-boundary.md``.
"""

from agent_core.components.verifier._compat import (
    audit_report_from_verdict,
    cycle_auditor_from_verifier,
    verdict_from_audit_report,
)
from agent_core.components.verifier.composers import (
    Cascade,
    ConsensusVerifier,
    Ensemble,
    Fallback,
    Parallel,
    Pipeline,
)
from agent_core.components.verifier.protocols import (
    Finding,
    Generator,
    GroundTruth,
    Verdict,
    Verifier,
    VerifierContext,
)

__all__ = [
    "Cascade",
    "ConsensusVerifier",
    "Ensemble",
    "Fallback",
    "Finding",
    "Generator",
    "GroundTruth",
    "Parallel",
    "Pipeline",
    "Verdict",
    "Verifier",
    "VerifierContext",
    "audit_report_from_verdict",
    "cycle_auditor_from_verifier",
    "verdict_from_audit_report",
]
