# pyre-strict
"""Structured session reasoner shared by Local and Sandcastle jobs."""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any, Callable, Dict, List, Optional

from cutracer.service.contracts import (
    AnalysisDecision,
    AnalysisSession,
    ArtifactRef,
    Confidence,
    CrossValidation,
    DecisionKind,
    EvidenceSufficiency,
    ExecutionStatus,
    ExperimentKind,
    ExperimentRequest,
    ExplainReport,
    SanitizerOutcome,
    StressOutcome,
)
from cutracer.service.triage import correlate_evidence

Runner = Callable[..., "subprocess.CompletedProcess[str]"]
ArtifactLoader = Callable[[ArtifactRef, int], Optional[str]]

_DECISION_SCHEMA = json.dumps(
    {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["final", "followup_required", "inconclusive"],
            },
            "rationale": {"type": "string"},
            "requests": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": [
                                "compute_sanitizer",
                                "random_delay_stress",
                                "reg_trace",
                                "mem_value_trace",
                                "reduce_delay_config",
                            ],
                        },
                        "rationale": {"type": "string"},
                    },
                    "required": ["kind", "rationale"],
                },
            },
            "report": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "root_cause": {"type": "string"},
                            "race_class": {"type": "string"},
                            "confidence": {
                                "type": "string",
                                "enum": ["H", "M", "L"],
                            },
                            "is_reproduced": {"type": "boolean"},
                            "cross_validation": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "agrees": {"type": "boolean"},
                                    "notes": {"type": "string"},
                                },
                                "required": ["agrees", "notes"],
                            },
                            "evidence_sufficiency": {
                                "type": "string",
                                "enum": ["sufficient", "insufficient", "degraded"],
                            },
                            "evidence_refs": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "recommended_action": {"type": "string"},
                        },
                        "required": [
                            "root_cause",
                            "race_class",
                            "confidence",
                            "is_reproduced",
                            "cross_validation",
                            "evidence_sufficiency",
                            "evidence_refs",
                            "recommended_action",
                        ],
                    },
                ]
            },
        },
        "required": ["kind", "rationale", "requests", "report"],
    },
    separators=(",", ":"),
)


def _subprocess_runner(
    argv: List[str], *, timeout: int, prompt: str
) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        input=prompt,
    )


def _parse_envelope(stdout: str) -> Optional[Dict[str, Any]]:
    try:
        envelope = json.loads(stdout)
    except (TypeError, ValueError):
        return None
    if not isinstance(envelope, dict):
        return None
    structured = envelope.get("structured_output")
    if isinstance(structured, dict):
        return structured
    result = envelope.get("result")
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except (TypeError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _report_from_dict(raw: Dict[str, Any], raw_ai_output: str) -> ExplainReport:
    cross = raw.get("cross_validation")
    cross_dict = cross if isinstance(cross, dict) else {}
    return ExplainReport(
        root_cause=str(raw.get("root_cause") or ""),
        race_class=str(raw.get("race_class") or "unknown"),
        confidence=Confidence(str(raw.get("confidence") or "L")),
        is_reproduced=bool(raw.get("is_reproduced")),
        cross_validation=CrossValidation(
            agrees=bool(cross_dict.get("agrees")),
            notes=str(cross_dict.get("notes") or ""),
        ),
        evidence_sufficiency=EvidenceSufficiency(
            str(raw.get("evidence_sufficiency") or "insufficient")
        ),
        evidence_refs=[str(x) for x in raw.get("evidence_refs", [])],
        recommended_action=str(raw.get("recommended_action") or ""),
        raw_ai_output=raw_ai_output,
    )


def _decision_from_dict(raw: Dict[str, Any], raw_ai_output: str) -> AnalysisDecision:
    kind = DecisionKind(raw["kind"])
    report_raw = raw.get("report")
    report = (
        _report_from_dict(report_raw, raw_ai_output)
        if isinstance(report_raw, dict)
        else None
    )
    return AnalysisDecision(
        kind=kind,
        rationale=str(raw.get("rationale") or ""),
        report=report,
        requests=[
            ExperimentRequest.from_dict(request)
            for request in raw.get("requests", [])
            if isinstance(request, dict)
        ],
    )


def _local_artifact_loader(ref: ArtifactRef, max_bytes: int) -> Optional[str]:
    if not ref.uri.startswith("file://"):
        return None
    path = ref.uri[len("file://") :]
    with open(path, "rb") as fh:
        payload = fh.read(max_bytes + 1)
    truncated = len(payload) > max_bytes
    text = payload[:max_bytes].decode("utf-8", errors="replace")
    return text + ("\n[truncated]" if truncated else "")


def _artifact_refs(session: AnalysisSession) -> List[ArtifactRef]:
    refs: List[ArtifactRef] = []
    for result in session.evidence.sanitizer:
        if result.log is not None:
            refs.append(result.log)
    for result in session.evidence.stress:
        if result.triggering_config is not None:
            refs.append(result.triggering_config.artifact)
        if result.log is not None:
            refs.append(result.log)
    for result in session.evidence.traces:
        if result.trace is not None:
            refs.append(result.trace)
        if result.log is not None:
            refs.append(result.log)
    for result in session.evidence.reductions:
        if result.minimized_config is not None:
            refs.append(result.minimized_config)
        if result.report is not None:
            refs.append(result.report)
    return refs


def _artifact_context(
    session: AnalysisSession,
    loader: ArtifactLoader,
    max_bytes: int,
) -> str:
    sections: List[str] = []
    seen = set()
    for ref in _artifact_refs(session):
        if ref.uri in seen:
            continue
        seen.add(ref.uri)
        try:
            content = loader(ref, max_bytes)
        # Loaders are deployment injection points and may wrap remote stores;
        # evidence retrieval failures must degrade the turn, not abort it.
        except Exception as exc:
            content = f"[artifact unavailable: {exc}]"
        if content is None:
            content = "[artifact content was not materialized for this reasoning turn]"
        sections.append(f"--- {ref.uri} ({ref.media_type}) ---\n{content}")
    return "\n\n".join(sections) if sections else "[no artifact content]"


def _prompt(session: AnalysisSession, artifact_context: str) -> str:
    triage = correlate_evidence(session.unit.unit_id, session.evidence)
    return (
        "You are analyzing a GPU correctness issue. The initial evidence sources "
        "compute-sanitizer and CUTracer random-delay stress are peers. Decide whether "
        "the accumulated evidence is sufficient for a final report or whether a "
        "bounded follow-up experiment is required. Request only the experiment kinds "
        "allowed by the schema. Do not emit shell commands. Prefer reg_trace for "
        "control/register provenance and mem_value_trace only when memory values are "
        "needed. Do not request reduce_delay_config unless the session already contains "
        "a triggering random-delay config; reduction never discovers or synthesizes "
        "one. When that config exists, the service will automatically add "
        "reduce_delay_config to a follow-up round. Ground every claim in the evidence "
        "and return inconclusive rather than inventing a root cause.\n\nSESSION:\n"
        + json.dumps(session.to_dict(), indent=2)
        + "\n\nCORRELATED TRIAGE:\n"
        + json.dumps(triage.to_dict(), indent=2)
        + "\n\nBOUNDED ARTIFACT CONTENT:\n"
        + artifact_context
    )


def _fallback_report(session: AnalysisSession, reason: str) -> ExplainReport:
    reproduced = any(
        result.outcome == StressOutcome.REPRODUCED for result in session.evidence.stress
    )
    refs: List[str] = []
    for result in session.evidence.sanitizer:
        if result.log is not None:
            refs.append(result.log.uri)
    for result in session.evidence.stress:
        if result.triggering_config is not None:
            refs.append(result.triggering_config.artifact.uri)
    for result in session.evidence.traces:
        if result.trace is not None:
            refs.append(result.trace.uri)
    for result in session.evidence.reductions:
        if result.report is not None:
            refs.append(result.report.uri)
    return ExplainReport(
        root_cause="",
        race_class="unknown",
        confidence=Confidence.L,
        is_reproduced=reproduced,
        cross_validation=CrossValidation(agrees=False, notes=reason),
        evidence_sufficiency=EvidenceSufficiency.DEGRADED,
        evidence_refs=refs,
        recommended_action="Inspect the preserved evidence or enable Claude reasoning.",
        raw_ai_output="",
    )


def fallback_decision(session: AnalysisSession, reason: str) -> AnalysisDecision:
    has_signal = any(
        result.outcome == SanitizerOutcome.FINDING
        for result in session.evidence.sanitizer
    ) or any(
        result.outcome == StressOutcome.REPRODUCED for result in session.evidence.stress
    )
    sanitizer_is_clean = bool(session.evidence.sanitizer) and all(
        result.execution_status == ExecutionStatus.SUCCEEDED
        and result.outcome == SanitizerOutcome.CLEAN
        for result in session.evidence.sanitizer
    )
    stress_is_clean = bool(session.evidence.stress) and all(
        result.execution_status == ExecutionStatus.SUCCEEDED
        and result.outcome == StressOutcome.NOT_REPRODUCED
        for result in session.evidence.stress
    )
    if not has_signal and sanitizer_is_clean and stress_is_clean:
        report = _fallback_report(session, reason)
        report.race_class = "none"
        report.evidence_sufficiency = EvidenceSufficiency.SUFFICIENT
        return AnalysisDecision(
            kind=DecisionKind.FINAL,
            rationale="both initial evidence sources completed without a finding",
            report=report,
        )
    if not has_signal:
        return AnalysisDecision(
            kind=DecisionKind.INCONCLUSIVE,
            rationale="one or more initial evidence sources were incomplete",
            report=_fallback_report(session, reason),
        )
    if not session.evidence.traces:
        return AnalysisDecision(
            kind=DecisionKind.FOLLOWUP_REQUIRED,
            rationale="a correctness signal exists but no localization trace is available",
            requests=[
                ExperimentRequest(
                    kind=ExperimentKind.REG_TRACE,
                    rationale="capture register/control provenance for localization",
                )
            ],
        )
    return AnalysisDecision(
        kind=DecisionKind.INCONCLUSIVE,
        rationale=reason,
        report=_fallback_report(session, reason),
    )


class ClaudeReasoner:
    def __init__(
        self,
        *,
        runner: Optional[Runner] = None,
        claude_bin: str = "claude",
        timeout: int = 1800,
        available: Optional[bool] = None,
        artifact_loader: Optional[ArtifactLoader] = None,
        max_artifact_bytes: int = 64 * 1024,
    ) -> None:
        self._runner = runner
        self._claude_bin = claude_bin
        self._timeout = timeout
        self._available = available
        self._artifact_loader = artifact_loader or _local_artifact_loader
        self._max_artifact_bytes = max_artifact_bytes

    def analyze(self, session: AnalysisSession) -> AnalysisDecision:
        available = (
            shutil.which(self._claude_bin) is not None
            if self._available is None
            else self._available
        )
        if not available:
            return fallback_decision(session, "Claude is unavailable")
        run = self._runner or _subprocess_runner
        prompt = _prompt(
            session,
            _artifact_context(
                session,
                self._artifact_loader,
                self._max_artifact_bytes,
            ),
        )
        try:
            proc = run(
                [
                    self._claude_bin,
                    "-p",
                    "--json-schema",
                    _DECISION_SCHEMA,
                    "--output-format",
                    "json",
                ],
                timeout=self._timeout,
                prompt=prompt,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return fallback_decision(session, f"Claude invocation failed: {exc}")
        if proc.returncode != 0 or not proc.stdout:
            return fallback_decision(session, "Claude returned no valid decision")
        structured = _parse_envelope(proc.stdout)
        if structured is None:
            return fallback_decision(session, "Claude decision was not valid JSON")
        try:
            return _decision_from_dict(structured, proc.stdout)
        except (KeyError, TypeError, ValueError) as exc:
            return fallback_decision(
                session, f"Claude decision failed validation: {exc}"
            )
