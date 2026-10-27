"""Cross-source correlation over initial evidence."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List

from cutracer.service.contracts import (
    AnalysisSession,
    EvidenceBundle,
    FindingRecord,
    TriageDecision,
    TriageOutcome,
)


def has_correctness_signal(evidence: EvidenceBundle) -> bool:
    """Whether either initial detector produced a positive correctness signal."""
    return any(result.has_positive_signal for result in evidence.sanitizer) or any(
        result.has_positive_signal for result in evidence.stress
    )


def sanitizer_is_trustworthy_clean(evidence: EvidenceBundle) -> bool:
    """Require every declared sanitizer result to prove a complete clean run."""
    return bool(evidence.sanitizer) and all(
        result.has_trustworthy_clean for result in evidence.sanitizer
    )


def stress_is_trustworthy_non_reproduction(evidence: EvidenceBundle) -> bool:
    """Require every stress result to cover its complete declared trial scope."""
    return bool(evidence.stress) and all(
        result.has_trustworthy_non_reproduction for result in evidence.stress
    )


def both_initial_sources_are_trustworthy_clean(evidence: EvidenceBundle) -> bool:
    """An absent or partial peer source is incomplete rather than neutral."""
    return sanitizer_is_trustworthy_clean(
        evidence
    ) and stress_is_trustworthy_non_reproduction(evidence)


def initial_evidence_requires_inconclusive(evidence: EvidenceBundle) -> bool:
    """Reject a no-signal initial fan-in unless both peer sources are complete."""
    has_initial_evidence = bool(evidence.sanitizer or evidence.stress)
    return (
        has_initial_evidence
        and not has_correctness_signal(evidence)
        and not both_initial_sources_are_trustworthy_clean(evidence)
    )


@dataclasses.dataclass
class CorrelatedFinding:
    correlation_key: str
    evidence_sources: List[str]
    findings: List[FindingRecord]
    stress_reproduced: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "correlation_key": self.correlation_key,
            "evidence_sources": list(self.evidence_sources),
            "findings": [finding.to_dict() for finding in self.findings],
            "stress_reproduced": self.stress_reproduced,
        }


@dataclasses.dataclass
class TriageResult:
    correlated_findings: List[CorrelatedFinding]
    should_analyze: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "correlated_findings": [x.to_dict() for x in self.correlated_findings],
            "should_analyze": self.should_analyze,
        }


class TriagePolicy:
    """Decides, without any AI, what a completed initial campaign warrants."""

    def decide(self, session: AnalysisSession) -> TriageDecision:
        if initial_evidence_requires_inconclusive(session.evidence):
            return TriageDecision(
                outcome=TriageOutcome.INCONCLUSIVE,
                reason="initial evidence is incomplete",
            )
        return TriageDecision(
            outcome=TriageOutcome.ANALYZE,
            reason="initial evidence warrants analysis",
        )


def correlate_evidence(unit_id: str, evidence: EvidenceBundle) -> TriageResult:
    findings = [finding for result in evidence.sanitizer for finding in result.findings]
    sanitizer_signals = [
        result for result in evidence.sanitizer if result.has_positive_signal
    ]
    stress_reproduced = any(result.has_positive_signal for result in evidence.stress)
    if not sanitizer_signals and not stress_reproduced:
        return TriageResult(correlated_findings=[], should_analyze=False)

    source_tools = {finding.source_tool for finding in findings}
    source_tools.update(
        f"compute_sanitizer/{result.tool}" for result in sanitizer_signals
    )
    if stress_reproduced:
        source_tools.add("cutracer/random_delay")
    sources = sorted(source_tools)
    return TriageResult(
        correlated_findings=[
            CorrelatedFinding(
                correlation_key=unit_id,
                evidence_sources=sources,
                findings=findings,
                stress_reproduced=stress_reproduced,
            )
        ],
        should_analyze=True,
    )
