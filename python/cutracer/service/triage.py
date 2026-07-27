# pyre-strict
"""Cross-source correlation over initial evidence."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List

from cutracer.service.contracts import EvidenceBundle, FindingRecord, StressOutcome


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


def correlate_evidence(unit_id: str, evidence: EvidenceBundle) -> TriageResult:
    findings = [finding for result in evidence.sanitizer for finding in result.findings]
    stress_reproduced = any(
        result.outcome == StressOutcome.REPRODUCED for result in evidence.stress
    )
    if not findings and not stress_reproduced:
        return TriageResult(correlated_findings=[], should_analyze=False)

    source_tools = {finding.source_tool for finding in findings}
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
