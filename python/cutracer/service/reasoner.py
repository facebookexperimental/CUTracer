"""Structured session reasoner shared by Local and Sandcastle jobs."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from typing import Any, Callable, Dict, List, Optional

from cutracer.service.artifacts import local_path
from cutracer.service.contracts import (
    AnalysisDecision,
    AnalysisSession,
    ArtifactRef,
    Confidence,
    CrossValidation,
    DecisionKind,
    EvidenceSufficiency,
    ExperimentKind,
    ExperimentRequest,
    ExplainReport,
    RaceClass,
)
from cutracer.service.prompt_projection import (
    project_artifact_context,
    project_session_context,
    PromptProjectionError,
)
from cutracer.service.triage import (
    correlate_evidence,
    has_correctness_signal,
    initial_evidence_requires_inconclusive,
    sanitizer_is_trustworthy_clean,
    stress_is_trustworthy_non_reproduction,
)
from jsonschema import validate
from jsonschema.exceptions import ValidationError

Runner = Callable[..., "subprocess.CompletedProcess[str]"]
ArtifactLoader = Callable[[ArtifactRef, int], Optional[str]]
logger: logging.Logger = logging.getLogger(__name__)

DECISION_SCHEMA = json.dumps(
    {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "kind": {
                "type": "string",
                "enum": ["final", "followup_required", "inconclusive"],
                "description": (
                    "Use final only with sufficient evidence for the reported "
                    "diagnosis. Use inconclusive for a remaining hypothesis "
                    "with insufficient or degraded evidence."
                ),
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
                        "instrument_modes": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["reg_trace", "mem_value_trace", "tma_trace"],
                            },
                            "maxItems": 3,
                            "uniqueItems": True,
                            "description": (
                                "Optional same-run instrumentation for trace requests "
                                "only; include the primary request kind."
                            ),
                        },
                        "trace_selector": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "kernel": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "name": {"type": "string"},
                                        "cubin_hash": {"type": "string"},
                                        "mangled_name": {"type": "string"},
                                    },
                                    "required": ["name", "cubin_hash"],
                                },
                                "launch_id": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "description": (
                                        "Observed global grid_launch_id for the pinned "
                                        "kernel, not its per-kernel iteration number. "
                                        "Omit when unknown; do not assume zero."
                                    ),
                                },
                                "warp_id": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": 31,
                                },
                                "warpgroup_id": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": 7,
                                },
                                "pc_offsets": {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                        "pattern": "^0[xX][0-9a-fA-F]+$",
                                    },
                                    "maxItems": 32,
                                    "uniqueItems": True,
                                },
                                "instruction_categories": {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                        "enum": ["mma", "tma", "sync"],
                                    },
                                    "maxItems": 3,
                                    "uniqueItems": True,
                                },
                                "config_source": {
                                    "type": "string",
                                    "enum": ["triggering", "minimized"],
                                },
                            },
                            "allOf": [
                                {"not": {"required": ["warp_id", "warpgroup_id"]}},
                                {"not": {"required": ["pc_offsets", "config_source"]}},
                            ],
                        },
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
                            "root_cause": {
                                "type": "string",
                                "description": (
                                    "Evidence-grounded explanation of the GPU "
                                    "correctness issue, which need not be a race."
                                ),
                            },
                            "race_class": {
                                "type": "string",
                                "enum": [race_class.value for race_class in RaceClass],
                                "description": (
                                    "Race mechanism diagnosed in this report. "
                                    "none means no race mechanism is diagnosed, "
                                    "not no GPU error or proof of race-freedom. "
                                    "The sanitizer error_type describes a "
                                    "separate observed symptom."
                                ),
                            },
                            "confidence": {
                                "type": "string",
                                "enum": ["H", "M", "L"],
                            },
                            "is_reproduced": {
                                "type": "boolean",
                                "description": (
                                    "An actual correctness signal was observed, "
                                    "including a sanitizer finding. Distinguish that "
                                    "signal from stress output-mismatch counts."
                                ),
                            },
                            "cross_validation": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "agrees": {
                                        "type": "boolean",
                                        "description": (
                                            "The observations are compatible; this "
                                            "does not establish independent proof "
                                            "of a cause or the absence of a race."
                                        ),
                                    },
                                    "notes": {"type": "string"},
                                },
                                "required": ["agrees", "notes"],
                            },
                            "evidence_sufficiency": {
                                "type": "string",
                                "enum": ["sufficient", "insufficient", "degraded"],
                                "description": (
                                    "Support for the reported diagnosis, not merely "
                                    "successful completion of a trace capture. "
                                    "A final report requires sufficient evidence."
                                ),
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


class ReasoningOutputError(ValueError):
    """A model response does not satisfy the reasoning output contract."""


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
        lines = result.strip().split("\n")
        if (
            len(lines) >= 3
            and lines[0].rstrip("\r") in ("```json", "```")
            and lines[-1] == "```"
        ):
            result = "\n".join(lines[1:-1])
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
        race_class=RaceClass.from_wire(str(raw.get("race_class") or "unknown")),
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


def parse_reasoning_output(stdout: str) -> AnalysisDecision:
    """Parse and strictly validate one Claude JSON output envelope."""
    structured = _parse_envelope(stdout)
    if structured is None:
        raise ReasoningOutputError("reasoning output is not a Claude JSON envelope")
    try:
        validate(instance=structured, schema=json.loads(DECISION_SCHEMA))
    except ValidationError as error:
        raise ReasoningOutputError(
            f"reasoning decision failed schema validation: {error.message}"
        ) from error
    try:
        return _decision_from_dict(structured, stdout)
    except (KeyError, TypeError, ValueError) as error:
        raise ReasoningOutputError(
            f"reasoning decision failed contract validation: {error}"
        ) from error


def _local_artifact_loader(ref: ArtifactRef, max_bytes: int) -> Optional[str]:
    if not ref.uri.startswith("file://"):
        return None
    path = local_path(ref)
    with open(path, "rb") as fh:
        payload = fh.read(max_bytes + 1)
    truncated = len(payload) > max_bytes
    text = payload[:max_bytes].decode("utf-8", errors="replace")
    return text + ("\n[truncated]" if truncated else "")


def _present_artifacts(
    *artifacts: Optional[ArtifactRef],
) -> List[ArtifactRef]:
    return [artifact for artifact in artifacts if artifact is not None]


def _artifact_refs(session: AnalysisSession) -> List[ArtifactRef]:
    # Deterministic structured output is the primary interpretation of every
    # trace. Newest evidence comes first so a tight global metadata budget
    # cannot be exhausted by old logs before the latest diagnosis is included.
    trace_analysis: List[ArtifactRef] = []
    trace_raw: List[ArtifactRef] = []
    trace_logs: List[ArtifactRef] = []
    for result in reversed(session.evidence.traces):
        analysis = result.analysis
        trace_analysis.extend(
            _present_artifacts(None if analysis is None else analysis.artifact)
        )
        trace_raw.extend(result.traces)
        trace_logs.extend(_present_artifacts(result.log))

    reduction_refs: List[ArtifactRef] = []
    for result in reversed(session.evidence.reductions):
        reduction_refs.extend(result.artifact_refs())

    stress_configs: List[ArtifactRef] = []
    stress_logs: List[ArtifactRef] = []
    for result in reversed(session.evidence.stress):
        triggering = result.triggering_config
        stress_configs.extend(
            _present_artifacts(
                None if triggering is None else triggering.artifact,
            )
        )
        stress_logs.extend(_present_artifacts(result.log))

    sanitizer_logs: List[ArtifactRef] = []
    for result in reversed(session.evidence.sanitizer):
        sanitizer_logs.extend(_present_artifacts(result.log))

    return [
        *trace_analysis,
        *reduction_refs,
        *stress_configs,
        *trace_raw,
        *trace_logs,
        *stress_logs,
        *sanitizer_logs,
    ]


def _artifact_context(
    session: AnalysisSession,
    loader: ArtifactLoader,
    max_bytes: int,
) -> str:
    return project_artifact_context(_artifact_refs(session), loader, max_bytes)


def _prompt(projected_state: str, artifact_context: str) -> str:
    return (
        "You are analyzing a GPU correctness issue. The initial evidence sources "
        "compute-sanitizer and CUTracer random-delay stress are peers. Decide whether "
        "the accumulated evidence is sufficient for a final report or whether a "
        "bounded follow-up experiment is required. Request only the experiment kinds "
        "allowed by the schema. Do not emit shell commands. Prefer reg_trace for "
        "control/register provenance. For TMA producer, mbarrier, and MMA consumer "
        'ordering, request instrument_modes=["reg_trace","tma_trace"] in one '
        "reg_trace experiment so the records describe the same execution. Add "
        "mem_value_trace only when ordinary memory addresses or values are needed. "
        "Trace selectors are optional, data-only, and allowlisted: use only the "
        "pinned kernel identity, non-negative launch IDs, CTA-local warp or warpgroup, "
        "hex PC offsets, mma/tma/sync categories, or an existing triggering/minimized "
        "config. launch_id is the observed global grid_launch_id for the pinned "
        "kernel, not a per-kernel iteration number or an iter0 filename suffix. "
        "Omit launch_id when unknown; do not assume zero. Native delay-config keys "
        "and reduction essential_points.pc_offset are decimal unless explicitly "
        "0x/0X-prefixed. The projected pc_offset_value and pc_offset_hex are "
        "deterministically computed from the original value before sampling. Use "
        "pc_offset_hex for hexadecimal citations and selectors: decimal 14672 "
        "equals 0x3950, not 0x14672. Never just prefix a decimal string with 0x. "
        "Null derived PC fields mean the original value could not be interpreted "
        "as an unsigned 64-bit offset; do not guess it. Preserve original replay "
        "config keys. A config source selects its enabled PCs; it does not replay delay "
        "injection. Never combine explicit PC offsets with a config source. A "
        "minimized delay point is a localization anchor, not a complete causal "
        "capture window or proof of the missing synchronization. Capture the "
        "producer, conflicting consumer, relevant ordering, and iteration boundary "
        "together. Do not automatically trace only the minimized PCs. If a complete "
        "small PC set cannot be established, omit the PC/config selector and capture "
        "the pinned kernel or launch under the existing time and trace-size limits. "
        "Instruction-category filters also remove evidence: omit categories when "
        "the detector requires a full instruction stream. An indeterminate detector "
        "or WARNING candidate cannot prove a missing wait or a RAW mechanism. "
        "Request a sufficiently complete capture when budget permits; otherwise "
        "report inconclusive. Missing EXIT records alone do not prove a truncated "
        "execution, because EXIT may not be included by the capture filters. "
        "Request reduction first when needed, then use the remaining follow-up "
        "round for this diagnostic capture. Do not request reduce_delay_config unless "
        "the session already contains "
        "a triggering random-delay config; reduction never discovers or synthesizes "
        "one. When that config exists, the service will automatically add "
        "reduce_delay_config to a follow-up round. Ground every claim in the evidence "
        "and return inconclusive rather than inventing a root cause. A final report "
        "requires evidence_sufficiency=sufficient; a completed capture with "
        "insufficient mechanism evidence remains inconclusive. Cite artifacts "
        "only by an exact URI present in the projected session. The race_class field "
        "describes the diagnosed race mechanism, not whether a GPU correctness issue "
        "exists. Use none when this report diagnoses no race mechanism; this is not "
        "proof that the kernel is race-free. A sanitizer error_type such as "
        "uninitialized-read describes an observed symptom, not proof of the presence "
        "or absence of a race. Do not infer a race class from the sanitizer tool name "
        "or relabel an unclassified race as none. Negative stress trials and clean "
        "sanitizer checks mean only that no issue was observed within those tests; "
        "they do not exclude a race or prove an observed invalid read harmless. "
        "Say no output mismatch was observed in the stated trials, not that an "
        "invalid read does not affect deterministic correctness. "
        "An unchanged output does not turn a sanitizer finding into a code-quality "
        "issue with no functional risk. Describe cross_validation.agrees as "
        "compatible observations, not independent confirmation of a cause. State "
        "which signal is reproduced: a sanitizer finding and a stress output "
        "mismatch are distinct. For a final report with "
        "race_class=none and nonempty root_cause, cite the log of a successful, "
        "complete sanitizer result with a detailed uninitialized-read, "
        "invalid-memory-access, misaligned-access, or memory-leak finding. A summary "
        "alone or partial/failed evidence does not suffice. Deterministic "
        "trace-analysis evidence embedded in PROJECTED SESSION is the primary "
        "interpretation "
        "of a trace; consult its structured summaries and findings before using raw "
        "trace excerpts as supporting evidence. capture_summaries provide bounded "
        "instruction/source context for planning: static instructions do not prove "
        "execution, and missing requested PCs mean not observed, not necessarily "
        "not executed. Findings are bounded samples. total_findings and per-detector "
        "finding_count give aggregate counts, not exhaustive PC/CTA/warp coverage. "
        "State aggregate counts separately from sampled examples; do not generalize "
        "sampled locations to all findings unless complete supplied evidence "
        "explicitly establishes that scope. An artifact reference or truncated "
        "excerpt does not establish that its full contents were examined. "
        "Projection omissions are explicit; "
        "do not infer facts from omitted history or content. Historical AI turns "
        "are proposals; their guard_notes may record rejection, and only the final "
        "session report is the accepted conclusion. Historical AI explanations "
        "may contain mistakes; prefer structured experiment evidence and computed "
        "PC fields over conflicting historical claims. A nonzero sanitizer exit "
        "with a zero-error footer is inconclusive, not a clean check. Do not infer "
        "a target signal or crash mechanism from the sanitizer return code alone.\n\n"
        "PROJECTED SESSION AND CORRELATED TRIAGE (JSON):\n"
        + projected_state
        + "\n\nBOUNDED ARTIFACT CONTENT (JSON):\n"
        + artifact_context
    )


def build_reasoning_prompt(
    session: AnalysisSession,
    *,
    artifact_loader: ArtifactLoader,
    max_session_bytes: int,
    max_artifact_bytes: int,
) -> str:
    triage = correlate_evidence(
        session.unit.unit_id,
        session.evidence,
        test_id=session.unit.test_id,
    )
    return _prompt(
        project_session_context(
            session,
            triage.to_dict(),
            max_session_bytes,
        ),
        _artifact_context(
            session,
            artifact_loader,
            max_artifact_bytes,
        ),
    )


def _fallback_report(session: AnalysisSession, reason: str) -> ExplainReport:
    reproduced = any(result.has_positive_signal for result in session.evidence.stress)
    refs: List[str] = []
    for result in session.evidence.sanitizer:
        if result.log is not None:
            refs.append(result.log.uri)
    for result in session.evidence.stress:
        if result.triggering_config is not None:
            refs.append(result.triggering_config.artifact.uri)
    for result in session.evidence.traces:
        if result.analysis is not None and result.analysis.artifact is not None:
            refs.append(result.analysis.artifact.uri)
    for result in session.evidence.traces:
        refs.extend(trace.uri for trace in result.traces)
        refs.extend(artifact.uri for artifact in result.supporting_artifacts)
    for result in session.evidence.reductions:
        if result.report is not None:
            refs.append(result.report.uri)
    return ExplainReport(
        root_cause="",
        race_class=RaceClass.UNKNOWN,
        confidence=Confidence.L,
        is_reproduced=reproduced,
        cross_validation=CrossValidation(agrees=False, notes=reason),
        evidence_sufficiency=EvidenceSufficiency.DEGRADED,
        evidence_refs=refs,
        recommended_action="Inspect the preserved evidence or enable Claude reasoning.",
        raw_ai_output="",
    )


def _has_correctness_signal(session: AnalysisSession) -> bool:
    return has_correctness_signal(session.evidence)


def _sanitizer_is_trustworthy_clean(session: AnalysisSession) -> bool:
    return sanitizer_is_trustworthy_clean(session.evidence)


def _stress_is_trustworthy_non_reproduction(session: AnalysisSession) -> bool:
    return stress_is_trustworthy_non_reproduction(session.evidence)


def fallback_decision(session: AnalysisSession, reason: str) -> AnalysisDecision:
    has_signal = _has_correctness_signal(session)
    sanitizer_is_clean = _sanitizer_is_trustworthy_clean(session)
    stress_is_clean = _stress_is_trustworthy_non_reproduction(session)
    if not has_signal and sanitizer_is_clean and stress_is_clean:
        report = _fallback_report(session, reason)
        report.race_class = RaceClass.NONE
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
        max_session_bytes: int = 64 * 1024,
        max_artifact_bytes: int = 64 * 1024,
    ) -> None:
        self._runner = runner
        self._claude_bin = claude_bin
        self._timeout = timeout
        self._available = available
        self._artifact_loader = artifact_loader or _local_artifact_loader
        self._max_session_bytes = max_session_bytes
        self._max_artifact_bytes = max_artifact_bytes

    def analyze(self, session: AnalysisSession) -> AnalysisDecision:
        if initial_evidence_requires_inconclusive(session.evidence):
            # Completeness is a deterministic service invariant. Do not let a
            # model promote legacy or partial initial evidence to clean.
            return fallback_decision(session, "initial evidence is incomplete")
        available = (
            shutil.which(self._claude_bin) is not None
            if self._available is None
            else self._available
        )
        if not available:
            return fallback_decision(session, "Claude is unavailable")
        run = self._runner or _subprocess_runner
        try:
            prompt = build_reasoning_prompt(
                session,
                artifact_loader=self._artifact_loader,
                max_session_bytes=self._max_session_bytes,
                max_artifact_bytes=self._max_artifact_bytes,
            )
        except PromptProjectionError as exc:
            logger.warning("reasoner prompt projection failed: %s", exc)
            return fallback_decision(
                session, f"reasoner prompt projection failed: {exc}"
            )
        try:
            proc = run(
                [
                    self._claude_bin,
                    "-p",
                    "--json-schema",
                    DECISION_SCHEMA,
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
        try:
            return parse_reasoning_output(proc.stdout)
        except ReasoningOutputError as exc:
            return fallback_decision(
                session, f"Claude decision failed validation: {exc}"
            )
