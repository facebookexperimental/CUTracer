"""Deterministic, byte-bounded projections for reasoner prompts."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from cutracer.service.contracts import AnalysisSession, ExperimentSpec

_PROJECTION_SCHEMA_VERSION = 1
_DETAIL_LIMITS: Tuple[Tuple[int, int], ...] = (
    (32, 4096),
    (16, 2048),
    (8, 1024),
    (4, 512),
    (1, 128),
    (0, 128),
)


class PromptProjectionError(ValueError):
    """The configured budget cannot preserve the required prompt context."""


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_size(value: Any) -> int:
    return len(_json_text(value).encode("utf-8"))


def _utf8_prefix(value: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


class _OmissionLog:
    def __init__(self) -> None:
        self._entries: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def add(
        self,
        section: str,
        reason: str,
        *,
        omitted_count: int = 0,
        omitted_bytes: int = 0,
    ) -> None:
        key = (section, reason)
        entry = self._entries.setdefault(
            key,
            {
                "section": section,
                "reason": reason,
                "omitted_count": 0,
                "omitted_bytes": 0,
            },
        )
        entry["omitted_count"] += omitted_count
        entry["omitted_bytes"] += omitted_bytes

    def to_list(self) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for key in sorted(self._entries):
            raw = self._entries[key]
            entry = {
                "section": raw["section"],
                "reason": raw["reason"],
            }
            if raw["omitted_count"]:
                entry["omitted_count"] = raw["omitted_count"]
            if raw["omitted_bytes"]:
                entry["omitted_bytes"] = raw["omitted_bytes"]
            entries.append(entry)
        return entries


def _sample_value(
    value: Any,
    section: str,
    *,
    max_list_items: int,
    max_string_bytes: int,
    omissions: _OmissionLog,
) -> Any:
    if isinstance(value, str):
        encoded_size = len(value.encode("utf-8"))
        if encoded_size <= max_string_bytes:
            return value
        sampled = _utf8_prefix(value, max_string_bytes)
        omissions.add(
            section,
            "string_sampled",
            omitted_bytes=encoded_size - len(sampled.encode("utf-8")),
        )
        return sampled
    if isinstance(value, list):
        included = value[:max_list_items]
        if len(value) > len(included):
            omissions.add(
                section,
                "list_sampled",
                omitted_count=len(value) - len(included),
            )
        item_section = f"{section}[]"
        return [
            _sample_value(
                item,
                item_section,
                max_list_items=max_list_items,
                max_string_bytes=max_string_bytes,
                omissions=omissions,
            )
            for item in included
        ]
    if isinstance(value, dict):
        items = sorted(value.items())
        included = items[:max_list_items]
        if len(items) > len(included):
            omissions.add(
                section,
                "mapping_sampled",
                omitted_count=len(items) - len(included),
            )
        return {
            key: _sample_value(
                child,
                f"{section}.{key}",
                max_list_items=max_list_items,
                max_string_bytes=max_string_bytes,
                omissions=omissions,
            )
            for key, child in included
        }
    return value


def _pending_projection(
    session: AnalysisSession,
    spec: ExperimentSpec,
    omissions: _OmissionLog,
    *,
    max_list_items: int,
    max_string_bytes: int,
) -> Dict[str, Any]:
    raw = spec.to_dict()
    spec_unit = raw.pop("unit", None)
    if spec_unit is not None and spec_unit == session.unit.to_dict():
        raw["unit_ref"] = session.unit.unit_id
        omissions.add(
            "session.pending_experiments[].unit",
            "deduplicated_as_session_unit",
            omitted_count=1,
        )
    elif spec_unit is not None:
        # Preserve a mismatched unit verbatim. Hiding it behind a unit_id-only
        # reference would conceal invalid or stale pending work from the model.
        raw["unit"] = spec_unit
    return {
        key: _sample_value(
            value,
            f"session.pending_experiments[].{key}",
            max_list_items=max_list_items,
            max_string_bytes=max_string_bytes,
            omissions=omissions,
        )
        for key, value in sorted(raw.items())
    }


def _latest_evidence_projection(
    session: AnalysisSession,
    omissions: _OmissionLog,
    *,
    max_list_items: int,
    max_string_bytes: int,
) -> Dict[str, List[Dict[str, Any]]]:
    latest_evidence: Dict[str, List[Dict[str, Any]]] = {}
    for name, results in (
        ("sanitizer", session.evidence.sanitizer),
        ("stress", session.evidence.stress),
        ("traces", session.evidence.traces),
        ("reductions", session.evidence.reductions),
    ):
        if len(results) > 1:
            omissions.add(
                f"session.evidence.{name}",
                "older_evidence",
                omitted_count=len(results) - 1,
            )
        latest_evidence[name] = (
            [
                {
                    key: _sample_value(
                        value,
                        f"session.evidence.{name}[].{key}",
                        max_list_items=max_list_items,
                        max_string_bytes=max_string_bytes,
                        omissions=omissions,
                    )
                    for key, value in sorted(results[-1].to_dict().items())
                }
            ]
            if results
            else []
        )
    return latest_evidence


def _latest_turn_projection(
    session: AnalysisSession,
    omissions: _OmissionLog,
    *,
    max_list_items: int,
    max_string_bytes: int,
) -> List[Dict[str, Any]]:
    if len(session.turns) > 1:
        omissions.add(
            "session.turns",
            "older_decision_history",
            omitted_count=len(session.turns) - 1,
        )
    turns = (
        [
            _sample_value(
                session.turns[-1].to_dict(),
                "session.turns[]",
                max_list_items=max_list_items,
                max_string_bytes=max_string_bytes,
                omissions=omissions,
            )
        ]
        if session.turns
        else []
    )
    return turns


def _pending_experiments_projection(
    session: AnalysisSession,
    omissions: _OmissionLog,
    *,
    max_list_items: int,
    max_string_bytes: int,
) -> List[Dict[str, Any]]:
    pending_experiments = session.pending_experiments[:max_list_items]
    if len(session.pending_experiments) > len(pending_experiments):
        omissions.add(
            "session.pending_experiments",
            "list_sampled",
            omitted_count=len(session.pending_experiments) - len(pending_experiments),
        )
    return [
        _pending_projection(
            session,
            spec,
            omissions,
            max_list_items=max_list_items,
            max_string_bytes=max_string_bytes,
        )
        for spec in pending_experiments
    ]


def _session_projection(
    session: AnalysisSession,
    omissions: _OmissionLog,
    *,
    max_list_items: int,
    max_string_bytes: int,
) -> Dict[str, Any]:
    if session.completed_experiment_ids:
        omissions.add(
            "session.completed_experiment_ids",
            "represented_by_count_and_evidence",
            omitted_count=len(session.completed_experiment_ids),
        )
    if session.experiment_fingerprints:
        omissions.add(
            "session.experiment_fingerprints",
            "represented_by_count",
            omitted_count=len(session.experiment_fingerprints),
        )

    return {
        "schema_version": session.schema_version,
        "session_id": session.session_id,
        "unit": session.unit.to_dict(),
        "status": session.status.value,
        "evidence": _latest_evidence_projection(
            session,
            omissions,
            max_list_items=max_list_items,
            max_string_bytes=max_string_bytes,
        ),
        "budget": session.budget.to_dict(),
        "pending_experiments": _pending_experiments_projection(
            session,
            omissions,
            max_list_items=max_list_items,
            max_string_bytes=max_string_bytes,
        ),
        "completed_experiment_count": len(session.completed_experiment_ids),
        "experiment_fingerprint_count": len(session.experiment_fingerprints),
        "round_index": session.round_index,
        "submitted_experiments": session.submitted_experiments,
        "turns": _latest_turn_projection(
            session,
            omissions,
            max_list_items=max_list_items,
            max_string_bytes=max_string_bytes,
        ),
        "report": (
            None
            if session.report is None
            else _sample_value(
                session.report.to_dict(),
                "session.report",
                max_list_items=max_list_items,
                max_string_bytes=max_string_bytes,
                omissions=omissions,
            )
        ),
        "termination_reason": _sample_value(
            session.termination_reason,
            "session.termination_reason",
            max_list_items=max_list_items,
            max_string_bytes=max_string_bytes,
            omissions=omissions,
        ),
        "state_revision": session.state_revision,
    }


def _session_candidate(
    session: AnalysisSession,
    correlated_triage: Dict[str, Any],
    max_bytes: int,
    detail_level: int,
    max_list_items: int,
    max_string_bytes: int,
) -> Dict[str, Any]:
    omissions = _OmissionLog()
    projected_session = _session_projection(
        session,
        omissions,
        max_list_items=max_list_items,
        max_string_bytes=max_string_bytes,
    )
    projected_triage = _sample_value(
        correlated_triage,
        "correlated_triage",
        max_list_items=max_list_items,
        max_string_bytes=max_string_bytes,
        omissions=omissions,
    )
    return {
        "projection_schema_version": _PROJECTION_SCHEMA_VERSION,
        "budget_bytes": max_bytes,
        "detail_level": detail_level,
        "session": projected_session,
        "correlated_triage": projected_triage,
        "omissions": omissions.to_list(),
    }


def project_session_context(
    session: AnalysisSession,
    correlated_triage: Dict[str, Any],
    max_bytes: int,
) -> str:
    """Return valid JSON containing the richest state that fits ``max_bytes``."""
    if max_bytes <= 0:
        raise PromptProjectionError("session projection budget must be positive")
    smallest_size: Optional[int] = None
    for detail_level, (max_list_items, max_string_bytes) in enumerate(_DETAIL_LIMITS):
        candidate = _session_candidate(
            session,
            correlated_triage,
            max_bytes,
            detail_level,
            max_list_items,
            max_string_bytes,
        )
        serialized = _json_text(candidate)
        size = len(serialized.encode("utf-8"))
        smallest_size = size if smallest_size is None else min(smallest_size, size)
        if size <= max_bytes:
            return serialized
    raise PromptProjectionError(
        "session projection budget "
        f"({max_bytes} bytes) cannot preserve required context "
        f"(minimum {smallest_size or 0} bytes)"
    )
