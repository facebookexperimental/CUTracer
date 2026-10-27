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
class RacecheckInvalidAccess:
    """A ``Potential invalid __shared__`` block.

    Racecheck emits these when it cannot prove the target block has entered.
    They carry a single access rather than a conflicting pair, so they are kept
    out of :attr:`RacecheckParseResult.findings` to leave the paired-access
    contract that downstream race analysis relies on intact.
    """

    message: str
    access_type: str
    size_bytes: int
    access: Optional[RacecheckAccess] = None
    raw_block: str = ""


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
    output_truncated: bool = False
    invalid_accesses: List[RacecheckInvalidAccess] = field(default_factory=list)

    @property
    def has_positive_signal(self) -> bool:
        """Whether the log reported anything at all, of either racecheck kind."""
        return bool(self.findings) or bool(self.invalid_accesses)

    @property
    def is_complete(self) -> bool:
        """Whether the log has a single self-consistent summary and findings.

        Truncated output stays complete: racecheck caps how many hazard blocks
        it prints, but the summary counts remain authoritative, so the run is
        still a proven positive rather than an unknown one.
        """
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
_TRUNCATION_RE = re.compile(r"^Warning: Maximum number of hazards reached")
_INVALID_ACCESS_RE = re.compile(
    r"^Potential invalid __shared__ (?P<type>\w+) of size (?P<size>\d+) bytes\s*$"
)
_INVALID_SITE_RE = re.compile(
    r"^at (?P<func>.+?)\+(?P<offset>0x[0-9a-fA-F]+)"
    r"(?: in (?P<file>.+?):(?P<line>\d+))?\s*$"
)

# How many sanitizer lines after a message may still carry its ``at`` site.
_SITE_LOOKAHEAD = 3


def _sanitizer_body(line: str) -> Optional[str]:
    """Return the sanitizer payload of ``line``, or ``None`` if it has none.

    Compute Sanitizer and the child process write to the same stream without
    coordinating, so the first diagnostic of a run is regularly appended to a
    pytest progress line instead of starting one.  Anchoring on the marker
    rather than the start of the line is what makes those blocks visible.
    """
    marker = line.find(_PREFIX + " ")
    if marker < 0:
        return None
    return line[marker + len(_PREFIX) :].strip()


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


def _build_invalid_access(
    groups: Dict[str, Optional[str]],
    site: Optional[Dict[str, Optional[str]]],
    raw_lines: List[str],
) -> RacecheckInvalidAccess:
    access_type = _required(groups, "type")
    size_bytes = int(_required(groups, "size"))
    access: Optional[RacecheckAccess] = None
    if site is not None:
        line = site.get("line")
        file = site.get("file")
        access = RacecheckAccess(
            role="primary",
            access_type=access_type,
            function=_required(site, "func"),
            offset=_required(site, "offset"),
            file=file if file is not None else "",
            line=int(line) if line is not None else 0,
        )
    return RacecheckInvalidAccess(
        message=(
            f"Potential invalid __shared__ {access_type} of size {size_bytes} bytes"
        ),
        access_type=access_type,
        size_bytes=size_bytes,
        access=access,
        raw_block="\n".join(raw_lines),
    )


def _summary_consistency_diagnostics(
    summary: RacecheckSummary,
    findings: List[RacecheckFinding],
    invalid_accesses: List[RacecheckInvalidAccess],
) -> List[RacecheckParseDiagnostic]:
    """Report contradictions between a parsed summary and finding blocks."""
    diagnostics: List[RacecheckParseDiagnostic] = []
    total_is_zero = summary.total_hazards == 0
    severities_are_zero = summary.errors == 0 and summary.warnings == 0
    if total_is_zero != severities_are_zero:
        diagnostics.append(RacecheckParseDiagnostic.SUMMARY_COUNT_MISMATCH)

    summary_is_positive = not (total_is_zero and severities_are_zero)
    reported_anything = bool(findings) or bool(invalid_accesses)
    errors_mismatch = (
        any(finding.severity == RacecheckSeverity.ERROR for finding in findings)
        and summary.errors == 0
    )
    warnings_mismatch = (
        any(finding.severity == RacecheckSeverity.WARNING for finding in findings)
        and summary.warnings == 0
    )
    missing_report = summary_is_positive and not reported_anything
    unexpected_report = not summary_is_positive and reported_anything
    if missing_report or unexpected_report or errors_mismatch or warnings_mismatch:
        diagnostics.append(RacecheckParseDiagnostic.SUMMARY_FINDING_MISMATCH)
    return diagnostics


@dataclass
class _RacecheckParser:
    findings: List[RacecheckFinding] = field(default_factory=list)
    invalid_accesses: List[RacecheckInvalidAccess] = field(default_factory=list)
    diagnostics: List[RacecheckParseDiagnostic] = field(default_factory=list)
    summary: Optional[RacecheckSummary] = None
    summary_markers: int = 0
    output_truncated: bool = False
    header: Optional[Dict[str, Optional[str]]] = None
    conflicts: List[Dict[str, Optional[str]]] = field(default_factory=list)
    raw_lines: List[str] = field(default_factory=list)
    pending_invalid: Optional[Dict[str, Optional[str]]] = None
    invalid_lookahead: int = 0

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
        if self.pending_invalid is not None:
            self.invalid_accesses.append(
                _build_invalid_access(self.pending_invalid, None, self.raw_lines)
            )
        self.header = None
        self.conflicts = []
        self.raw_lines = []
        self.pending_invalid = None
        self.invalid_lookahead = 0

    def consume_summary(self, body: str) -> bool:
        if "RACECHECK SUMMARY:" in body:
            self.flush()
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
            return True
        return False

    def consume_auxiliary_block(self, body: str) -> bool:
        if _TRUNCATION_RE.match(body):
            # A display cap, not a hazard: the summary counts stay authoritative
            # and must not be contradicted by the smaller number of blocks.
            self.flush()
            self.output_truncated = True
            return True

        invalid_match = _INVALID_ACCESS_RE.match(body)
        if invalid_match:
            self.flush()
            self.pending_invalid = invalid_match.groupdict()
            self.raw_lines = [body]
            return True

        if self.pending_invalid is not None:
            if _HEADER_RE.match(body):
                self.flush()
                return False
            site_match = _INVALID_SITE_RE.match(body)
            if site_match:
                self.raw_lines.append(body)
                self.invalid_accesses.append(
                    _build_invalid_access(
                        self.pending_invalid,
                        site_match.groupdict(),
                        self.raw_lines,
                    )
                )
                self.pending_invalid = None
                self.raw_lines = []
                return True
            # Tolerate the same number of intervening lines as the checker
            # layer before giving up on finding this block's location.
            self.invalid_lookahead += 1
            if self.invalid_lookahead <= _SITE_LOOKAHEAD:
                self.raw_lines.append(body)
                return True
            self.flush()
        return False

    def consume(self, line: str) -> None:
        body = _sanitizer_body(line)
        if body is None:
            return
        if self.consume_summary(body):
            return
        if self.consume_auxiliary_block(body):
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
            for issue in _summary_consistency_diagnostics(
                self.summary,
                self.findings,
                self.invalid_accesses,
            ):
                self.diagnose(issue)
        return RacecheckParseResult(
            findings=self.findings,
            summary=self.summary,
            diagnostics=self.diagnostics,
            output_truncated=self.output_truncated,
            invalid_accesses=self.invalid_accesses,
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
