"""Cross-source correlation over initial evidence."""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional

from cutracer.service.baseline import (
    baseline_from_scope,
    BaselineStore,
    diff_against_baseline,
)
from cutracer.service.contracts import (
    AnalysisSession,
    BaselineVerdict,
    EvidenceBundle,
    ExecutionStatus,
    FindingRecord,
    parse_ymd_date,
    ResultCompleteness,
    SanitizerSweepResult,
    ScopeId,
    ScopeTriageSnapshot,
    SuppressionAuditEntry,
    TriageDecision,
    TriageOutcome,
    TriageSnapshot,
)
from cutracer.service.suppression import SuppressionOutcome, SuppressionPolicy


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


def test_family(test_id: str) -> str:
    """Strip the pytest parametrization from a test id.

    Parametrized cases are one defect reported many times: in the sweep corpus
    320 findings land on 312 distinct test ids but only 40 families, so keying
    on the id would dedupe almost nothing.
    """
    if not test_id.endswith("]"):
        return test_id
    depth = 0
    for index in range(len(test_id) - 1, -1, -1):
        if test_id[index] == "]":
            depth += 1
        elif test_id[index] == "[":
            depth -= 1
            if depth == 0:
                family = test_id[:index]
                return family or test_id
    return test_id


def correlation_key(finding: FindingRecord) -> str:
    """Group findings by where the defect is, not by which run saw it.

    The kernel symbol is the only component the sanitizer always attributes
    (100% of corpus findings carry one), so it anchors the key. The source
    location tightens it where present and is simply absent otherwise -- 15% of
    real corpus findings have an unresolvable offset, and demanding a location
    would either drop them or merge unrelated defects in the same kernel.
    """
    symbol = finding.kernel_name or "<unattributed>"
    location = (
        None
        if finding.source is None
        else f"{finding.source.file}:{finding.source.line}"
    )
    if location is None:
        return f"{symbol}|{finding.error_type}"
    return f"{symbol}|{location}|{finding.error_type}"


@dataclasses.dataclass
class CorrelatedFinding:
    correlation_key: str
    evidence_sources: List[str]
    findings: List[FindingRecord]
    stress_reproduced: bool
    kernel_symbol: Optional[str] = None
    source_location: Optional[str] = None
    error_type: str = ""
    test_family: str = ""
    tools: List[str] = dataclasses.field(default_factory=list)
    total_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "correlation_key": self.correlation_key,
            "evidence_sources": list(self.evidence_sources),
            "findings": [finding.to_dict() for finding in self.findings],
            "stress_reproduced": self.stress_reproduced,
            "kernel_symbol": self.kernel_symbol,
            "source_location": self.source_location,
            "error_type": self.error_type,
            "test_family": self.test_family,
            "tools": list(self.tools),
            "total_count": self.total_count,
        }


@dataclasses.dataclass
class TriageResult:
    correlated_findings: List[CorrelatedFinding]
    should_analyze: bool
    suppressed: SuppressionOutcome = dataclasses.field(
        default_factory=SuppressionOutcome
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "correlated_findings": [x.to_dict() for x in self.correlated_findings],
            "should_analyze": self.should_analyze,
            "suppressed": self.suppressed.audit(),
        }


class TriagePolicy:
    """Decides, without any AI, what a completed initial campaign warrants.

    Only a newly introduced finding, an unbaselined one, or an independent
    stress reproduction is worth a reasoning turn. Measured on the sweep
    corpus this is the difference between one AI turn per test (458) and one
    per distinct defect (34), before any baseline has even been established.
    """

    def __init__(
        self,
        *,
        baselines: Optional[BaselineStore] = None,
        suppression: Optional[SuppressionPolicy] = None,
        today: Optional[str] = None,
        record_baselines: bool = True,
    ) -> None:
        if today is not None and parse_ymd_date(today) is None:
            raise ValueError(f"today must be YYYY-MM-DD, got {today!r}")
        if record_baselines and baselines is not None and today is None:
            # Silently skipping the write would leave the store permanently
            # empty and every run reporting as unbaselined.
            raise ValueError("recording baselines requires today=YYYY-MM-DD")
        self._baselines = baselines
        self._suppression = suppression
        self._today = today
        self._record_baselines = record_baselines

    def _scope_completeness(
        self, session: AnalysisSession
    ) -> Dict[str, ResultCompleteness]:
        """Whether each tool proved it covered its declared scope."""
        complete: Dict[str, ResultCompleteness] = {}
        for result in session.evidence.sanitizer:
            observed = result.completeness
            # A complete parse proves only that the log is whole; a target that
            # did not succeed may have stopped before covering its scope.
            if (
                observed == ResultCompleteness.COMPLETE
                and result.execution_status != ExecutionStatus.SUCCEEDED
            ):
                observed = ResultCompleteness.UNKNOWN
            conflicting_tools = {
                finding.source_tool.split("/")[-1]
                for finding in result.findings
                if finding.source_tool.split("/")[-1] != result.tool
            }
            if conflicting_tools:
                observed = ResultCompleteness.UNKNOWN
                for tool in conflicting_tools:
                    complete[tool] = ResultCompleteness.UNKNOWN
            previous = complete.get(result.tool)
            if previous is None:
                complete[result.tool] = observed
            elif previous != ResultCompleteness.COMPLETE:
                complete[result.tool] = previous
            elif observed != ResultCompleteness.COMPLETE:
                complete[result.tool] = observed
        return complete

    def _triage_snapshot(
        self,
        session: AnalysisSession,
        triage: TriageResult,
        *,
        record_baselines: bool = True,
    ) -> TriageSnapshot:
        family = test_family(session.unit.test_id)
        completeness = self._scope_completeness(session)
        by_scope: Dict[str, List[str]] = {}
        for entry in triage.correlated_findings:
            for tool in entry.tools:
                by_scope.setdefault(tool.split("/")[-1], []).append(
                    entry.correlation_key
                )

        scopes: List[ScopeTriageSnapshot] = []
        for tool in sorted(completeness.keys() | by_scope.keys()):
            scope_completeness = completeness.get(tool, ResultCompleteness.UNKNOWN)
            scope = ScopeId(test_family=family, tool=tool)
            keys = by_scope.get(tool, [])
            baseline_recorded = False
            if self._baselines is None:
                verdict = BaselineVerdict.NOT_EVALUATED
                observed = tuple(sorted(set(keys)))
                new = resolved = unchanged = ()
            else:
                diff = diff_against_baseline(
                    scope,
                    keys,
                    self._baselines.load(scope),
                )
                verdict = diff.verdict
                observed = diff.observed_keys
                new = diff.new_keys
                resolved = diff.resolved_keys
                unchanged = diff.unchanged_keys
            if (
                record_baselines
                and self._baselines is not None
                and self._record_baselines
                and self._today is not None
            ):
                recorded = baseline_from_scope(
                    scope,
                    keys,
                    is_complete=scope_completeness == ResultCompleteness.COMPLETE,
                    source_revision=session.unit.source_revision,
                    recorded_on=self._today,
                )
                if recorded is not None:
                    self._baselines.record(recorded)
                    baseline_recorded = True
            scopes.append(
                ScopeTriageSnapshot(
                    scope=scope,
                    completeness=scope_completeness,
                    baseline_verdict=verdict,
                    observed_keys=observed,
                    new_keys=new,
                    resolved_keys=resolved,
                    unchanged_keys=unchanged,
                    baseline_recorded=baseline_recorded,
                )
            )

        return TriageSnapshot(
            scopes=tuple(scopes),
            stress_completeness=_aggregate_completeness(
                [result.completeness for result in session.evidence.stress]
            ),
            suppression_audit=tuple(
                SuppressionAuditEntry(
                    finding_key=correlation_key(entry.finding),
                    source_tool=entry.finding.source_tool,
                    error_type=entry.finding.error_type,
                    kernel_name=entry.finding.kernel_name,
                    source=entry.finding.source,
                    decision=entry.decision,
                )
                for entry in triage.suppressed.suppressed
            ),
        )

    def decide(self, session: AnalysisSession) -> TriageDecision:
        triage = correlate_evidence(
            session.unit.unit_id,
            session.evidence,
            test_id=session.unit.test_id,
            suppression=self._suppression,
            today=self._today,
        )
        if initial_evidence_requires_inconclusive(session.evidence):
            return TriageDecision(
                outcome=TriageOutcome.INCONCLUSIVE,
                reason="initial evidence is incomplete",
                snapshot=self._triage_snapshot(session, triage, record_baselines=False),
            )
        snapshot = self._triage_snapshot(session, triage)

        if not triage.should_analyze:
            suppressed = len(triage.suppressed.suppressed)
            reason = (
                f"no actionable findings; {suppressed} suppressed"
                if suppressed
                else "both initial sources are complete and clean"
            )
            return TriageDecision(
                outcome=TriageOutcome.FINISH,
                reason=reason,
                snapshot=snapshot,
            )

        if any(entry.stress_reproduced for entry in triage.correlated_findings):
            # Stress reproduces a real correctness failure and has no baseline
            # of its own; it is an independent signal, not a repeat.
            return TriageDecision(
                outcome=TriageOutcome.ANALYZE,
                reason="random-delay stress reproduced a correctness failure",
                snapshot=snapshot,
            )

        if self._baselines is None:
            return TriageDecision(
                outcome=TriageOutcome.ANALYZE,
                reason=f"{len(triage.correlated_findings)} findings, no baseline store",
                snapshot=snapshot,
            )

        introduced = [key for scope in snapshot.scopes for key in scope.new_keys]
        unbaselined = [
            key
            for scope in snapshot.scopes
            if scope.baseline_verdict == BaselineVerdict.NO_BASELINE
            for key in scope.observed_keys
        ]
        unproven = [
            scope.scope.key
            for scope in snapshot.scopes
            if scope.completeness != ResultCompleteness.COMPLETE
        ]

        if introduced:
            return TriageDecision(
                outcome=TriageOutcome.ANALYZE,
                reason=f"{len(set(introduced))} newly introduced findings",
                snapshot=snapshot,
            )
        if unbaselined:
            return TriageDecision(
                outcome=TriageOutcome.ANALYZE,
                reason=f"{len(set(unbaselined))} findings with no baseline to compare",
                snapshot=snapshot,
            )
        if unproven:
            # Not re-observing a baseline finding under partial coverage is not
            # evidence that it is gone. Calling that "unchanged" would let
            # incomplete coverage read as a verified result.
            return TriageDecision(
                outcome=TriageOutcome.INCONCLUSIVE,
                reason=f"scope coverage unproven: {', '.join(sorted(set(unproven)))}",
                snapshot=snapshot,
            )
        return TriageDecision(
            outcome=TriageOutcome.FINISH,
            reason="all findings are unchanged since the last complete baseline",
            snapshot=snapshot,
        )


def _aggregate_completeness(
    values: List[ResultCompleteness],
) -> ResultCompleteness:
    if not values:
        return ResultCompleteness.UNKNOWN
    if all(value == ResultCompleteness.COMPLETE for value in values):
        return ResultCompleteness.COMPLETE
    if all(
        value in (ResultCompleteness.COMPLETE, ResultCompleteness.COMPLETE_EARLY)
        for value in values
    ):
        return ResultCompleteness.COMPLETE_EARLY
    if all(value == ResultCompleteness.SKIPPED for value in values):
        return ResultCompleteness.SKIPPED
    return ResultCompleteness.PARTIAL


def _active_sanitizer_signals(
    evidence: EvidenceBundle,
) -> List[SanitizerSweepResult]:
    """Return positive results with evidence not attributable to detail blocks.

    A positive summary without findings is wholly unattributed. A truncated
    result remains partially unattributed even when its retained prefix contains
    findings, because additional finding blocks may have been omitted.
    """
    return [
        result
        for result in evidence.sanitizer
        if result.has_positive_signal
        and (not result.findings or result.output_truncated)
    ]


def _group_findings(
    findings: List[FindingRecord], family: str
) -> tuple[Dict[str, CorrelatedFinding], List[str]]:
    grouped: Dict[str, CorrelatedFinding] = {}
    order: List[str] = []
    for finding in findings:
        key = correlation_key(finding)
        entry = grouped.get(key)
        if entry is None:
            location = (
                None
                if finding.source is None
                else f"{finding.source.file}:{finding.source.line}"
            )
            entry = CorrelatedFinding(
                correlation_key=key,
                evidence_sources=[],
                findings=[],
                stress_reproduced=False,
                kernel_symbol=finding.kernel_name,
                source_location=location,
                error_type=finding.error_type,
                test_family=family,
            )
            grouped[key] = entry
            order.append(key)
        entry.findings.append(finding)
        entry.total_count += finding.count
        if finding.source_tool not in entry.tools:
            entry.tools.append(finding.source_tool)
    return grouped, order


def _add_unattributed_sanitizer_signals(
    grouped: Dict[str, CorrelatedFinding],
    order: List[str],
    sanitizer_signals: List[SanitizerSweepResult],
    family: str,
) -> None:
    for result in sanitizer_signals:
        tool = f"compute_sanitizer/{result.tool}"
        key = f"<unattributed-summary>|{tool}"
        if key in grouped:
            continue
        grouped[key] = CorrelatedFinding(
            correlation_key=key,
            evidence_sources=[tool],
            findings=[],
            stress_reproduced=False,
            error_type="unattributed",
            test_family=family,
            tools=[tool],
        )
        order.append(key)


def _add_stress_signal(
    grouped: Dict[str, CorrelatedFinding],
    order: List[str],
    *,
    unit_id: str,
    family: str,
) -> None:
    attributed = [
        entry
        for entry in grouped.values()
        if not entry.correlation_key.startswith("<unattributed-summary>|")
    ]
    for entry in attributed:
        entry.stress_reproduced = True
    if attributed:
        return
    key = f"<stress>|{unit_id}"
    grouped[key] = CorrelatedFinding(
        correlation_key=key,
        evidence_sources=[],
        findings=[],
        stress_reproduced=True,
        error_type="correctness-failure",
        test_family=family,
    )
    order.append(key)


def _populate_evidence_sources(grouped: Dict[str, CorrelatedFinding]) -> None:
    for entry in grouped.values():
        sources = set(entry.tools)
        if entry.stress_reproduced:
            sources.add("cutracer/random_delay")
        entry.evidence_sources = sorted(sources)


def correlate_evidence(
    unit_id: str,
    evidence: EvidenceBundle,
    *,
    test_id: str = "",
    suppression: Optional[SuppressionPolicy] = None,
    today: Optional[str] = None,
) -> TriageResult:
    """Group the initial evidence into one entry per distinct defect.

    Suppression runs first: in the sweep corpus 83% of findings are attributed
    to framework kernels, so correlating before filtering would spend the whole
    pass grouping noise.
    """
    raw = [finding for result in evidence.sanitizer for finding in result.findings]
    policy = suppression if suppression is not None else SuppressionPolicy()
    outcome = policy.apply(raw, today=today)
    findings = list(outcome.kept)

    # A tool whose every block was suppressed contributed no defect and must
    # not resurface below as an unattributed one. A tool that printed no block
    # at all is different: its positive summary is still a real signal.
    sanitizer_signals = _active_sanitizer_signals(evidence)
    stress_reproduced = any(result.has_positive_signal for result in evidence.stress)
    if not findings and not sanitizer_signals and not stress_reproduced:
        # Either nothing was reported, or everything reported was suppressed.
        return TriageResult(
            correlated_findings=[],
            should_analyze=False,
            suppressed=outcome,
        )

    family = test_family(test_id)
    grouped, order = _group_findings(findings, family)
    _add_unattributed_sanitizer_signals(
        grouped,
        order,
        sanitizer_signals,
        family,
    )

    if stress_reproduced:
        # Random-delay stress reproduces a correctness failure for the work
        # unit as a whole; it does not attribute a site, so it is recorded as a
        # peer source on every attributed correlated finding rather than
        # inventing one. Unattributed summaries remain a distinct signal.
        _add_stress_signal(grouped, order, unit_id=unit_id, family=family)

    _populate_evidence_sources(grouped)

    return TriageResult(
        correlated_findings=[grouped[key] for key in order],
        should_analyze=True,
        suppressed=outcome,
    )
