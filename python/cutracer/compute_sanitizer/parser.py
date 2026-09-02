# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Parse NVIDIA Compute Sanitizer racecheck output into neutral records.

This module is part of the OSS package and deliberately has no dependency on
CUTracer's internal analysis taxonomy.  Consumers adapt the parsed records to
their own finding and reporting types.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class RacecheckSeverity(str, Enum):
    """Severity reported by a racecheck finding header."""

    ERROR = "error"
    WARNING = "warning"


class RacecheckParseDiagnostic(str, Enum):
    """A fact that prevents the parser output from proving a complete run."""

    MISSING_SUMMARY = "missing_summary"
    MALFORMED_SUMMARY = "malformed_summary"
    MULTIPLE_SUMMARIES = "multiple_summaries"
    INCOMPLETE_FINDING = "incomplete_finding"
    UNPARSED_FINDING = "unparsed_finding"
    SUMMARY_COUNT_MISMATCH = "summary_count_mismatch"
    SUMMARY_FINDING_MISMATCH = "summary_finding_mismatch"


@dataclass(frozen=True)
class RacecheckAccess:
    """One access participating in a racecheck finding."""

    role: str
    access_type: str
    function: str
    offset: str
    file: str
    line: int
    hazards: Optional[int] = None


@dataclass(frozen=True)
class RacecheckFinding:
    """One race block, independent of any downstream finding taxonomy."""

    severity: RacecheckSeverity
    accesses: List[RacecheckAccess]
    raw_block: str

    @property
    def max_hazards(self) -> Optional[int]:
        """Largest conflict count reported in this block, if any."""
        counts = [a.hazards for a in self.accesses if a.hazards is not None]
        return max(counts) if counts else None

    @property
    def access_kinds(self) -> List[str]:
        """Access kinds in first-seen order."""
        kinds: List[str] = []
        for access in self.accesses:
            if access.access_type not in kinds:
                kinds.append(access.access_type)
        return kinds

    @property
    def locations(self) -> List[str]:
        """Source locations in first-seen order."""
        locations: List[str] = []
        for access in self.accesses:
            location = f"{access.file}:{access.line}"
            if location not in locations:
                locations.append(location)
        return locations


@dataclass(frozen=True)
class RacecheckSummary:
    """Aggregate counts from the ``RACECHECK SUMMARY`` line."""

    total_hazards: int
    errors: int
    warnings: int
    raw: str


@dataclass(frozen=True)
class RacecheckParseResult:
    """Structured output from one racecheck log."""

    findings: List[RacecheckFinding]
    summary: Optional[RacecheckSummary] = None
    tool: str = "racecheck"
    diagnostics: List[RacecheckParseDiagnostic] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """Whether the log has a single self-consistent summary and findings."""
        return self.summary is not None and not self.diagnostics


# Compute Sanitizer prefixes its own output with this marker. Child process
# output can be interleaved without it and must not terminate an open block.
_PREFIX = "========="

_HEADER_RE = re.compile(
    r"^(?P<level>Error|Warning): Race reported between "
    r"(?P<type>\w+) access at (?P<func>\S+)\+(?P<offset>0x[0-9a-fA-F]+) "
    r"in (?P<file>.+?):(?P<line>\d+)\s*$"
)
_CONFLICT_RE = re.compile(
    r"^and (?P<type>\w+) access at (?P<func>\S+)\+(?P<offset>0x[0-9a-fA-F]+) "
    r"in (?P<file>.+?):(?P<line>\d+)"
    r"(?:\s*\[(?P<hazards>\d+) hazards?\])?\s*$"
)
_SUMMARY_RE = re.compile(
    r"^RACECHECK SUMMARY: (?P<total>\d+) hazards? displayed "
    r"\((?P<errors>\d+) errors?, (?P<warnings>\d+) warnings?\)$"
)


def _required(groups: Dict[str, Optional[str]], name: str) -> str:
    value = groups.get(name)
    if value is None:
        raise ValueError(f"racecheck parser missing required group: {name}")
    return value


def _build_access(groups: Dict[str, Optional[str]], role: str) -> RacecheckAccess:
    hazards = groups.get("hazards")
    return RacecheckAccess(
        role=role,
        access_type=_required(groups, "type"),
        function=_required(groups, "func"),
        offset=_required(groups, "offset"),
        file=_required(groups, "file"),
        line=int(_required(groups, "line")),
        hazards=int(hazards) if hazards is not None else None,
    )


def _build_finding(
    header: Dict[str, Optional[str]],
    conflicts: List[Dict[str, Optional[str]]],
    raw_lines: List[str],
) -> RacecheckFinding:
    severity = (
        RacecheckSeverity.ERROR
        if _required(header, "level") == "Error"
        else RacecheckSeverity.WARNING
    )
    accesses = [_build_access(header, "primary")]
    accesses.extend(_build_access(conflict, "conflict") for conflict in conflicts)
    return RacecheckFinding(
        severity=severity,
        accesses=accesses,
        raw_block="\n".join(raw_lines),
    )


def _summary_consistency_diagnostics(
    summary: RacecheckSummary,
    findings: List[RacecheckFinding],
) -> List[RacecheckParseDiagnostic]:
    """Report contradictions between a parsed summary and finding blocks."""
    diagnostics: List[RacecheckParseDiagnostic] = []
    total_is_zero = summary.total_hazards == 0
    severities_are_zero = summary.errors == 0 and summary.warnings == 0
    if total_is_zero != severities_are_zero:
        diagnostics.append(RacecheckParseDiagnostic.SUMMARY_COUNT_MISMATCH)

    summary_is_positive = not (total_is_zero and severities_are_zero)
    findings_mismatch = summary_is_positive != bool(findings)
    errors_mismatch = (
        any(finding.severity == RacecheckSeverity.ERROR for finding in findings)
        and summary.errors == 0
    )
    warnings_mismatch = (
        any(finding.severity == RacecheckSeverity.WARNING for finding in findings)
        and summary.warnings == 0
    )
    if findings_mismatch or errors_mismatch or warnings_mismatch:
        diagnostics.append(RacecheckParseDiagnostic.SUMMARY_FINDING_MISMATCH)
    return diagnostics


@dataclass
class _RacecheckParser:
    findings: List[RacecheckFinding] = field(default_factory=list)
    diagnostics: List[RacecheckParseDiagnostic] = field(default_factory=list)
    summary: Optional[RacecheckSummary] = None
    summary_markers: int = 0
    header: Optional[Dict[str, Optional[str]]] = None
    conflicts: List[Dict[str, Optional[str]]] = field(default_factory=list)
    raw_lines: List[str] = field(default_factory=list)

    def diagnose(self, issue: RacecheckParseDiagnostic) -> None:
        if issue not in self.diagnostics:
            self.diagnostics.append(issue)

    def flush(self) -> None:
        if self.header is not None:
            if self.conflicts:
                self.findings.append(
                    _build_finding(self.header, self.conflicts, self.raw_lines)
                )
            else:
                self.diagnose(RacecheckParseDiagnostic.INCOMPLETE_FINDING)
        self.header = None
        self.conflicts = []
        self.raw_lines = []

    def consume(self, line: str) -> None:
        if not line.startswith(_PREFIX):
            return
        body = line[len(_PREFIX) :].strip()

        if "RACECHECK SUMMARY:" in body:
            self.summary_markers += 1
            summary_match = _SUMMARY_RE.fullmatch(body)
            if summary_match:
                self.summary = RacecheckSummary(
                    total_hazards=int(summary_match["total"]),
                    errors=int(summary_match["errors"]),
                    warnings=int(summary_match["warnings"]),
                    raw=body,
                )
            else:
                self.diagnose(RacecheckParseDiagnostic.MALFORMED_SUMMARY)
            return

        header_match = _HEADER_RE.match(body)
        if header_match:
            self.flush()
            self.header = header_match.groupdict()
            self.raw_lines = [body]
            return

        if "Race reported between" in body:
            self.flush()
            self.diagnose(RacecheckParseDiagnostic.UNPARSED_FINDING)
            return

        if self.header is None:
            return
        conflict_match = _CONFLICT_RE.match(body)
        if conflict_match:
            self.conflicts.append(conflict_match.groupdict())
            self.raw_lines.append(body)
            return
        if body.startswith("and ") and " access " in body:
            self.diagnose(RacecheckParseDiagnostic.UNPARSED_FINDING)
        self.flush()

    def finish(self) -> RacecheckParseResult:
        self.flush()
        if self.summary_markers == 0:
            self.diagnose(RacecheckParseDiagnostic.MISSING_SUMMARY)
        elif self.summary_markers > 1:
            self.diagnose(RacecheckParseDiagnostic.MULTIPLE_SUMMARIES)

        if self.summary is not None:
            for issue in _summary_consistency_diagnostics(self.summary, self.findings):
                self.diagnose(issue)
        return RacecheckParseResult(
            findings=self.findings,
            summary=self.summary,
            diagnostics=self.diagnostics,
        )


def parse_racecheck_log(text: str) -> RacecheckParseResult:
    """Parse racecheck stdout without depending on internal CUTracer types.

    The parser is intentionally lenient: unrelated and interleaved child
    output is ignored, malformed fragments produce no finding, and a missing
    summary is represented by ``None``. Temporal RAW/WAR/WAW classification is
    not inferred because racecheck output does not establish execution order.
    """
    parser = _RacecheckParser()
    for line in text.splitlines():
        parser.consume(line)
    return parser.finish()
