# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Parse NVIDIA Compute Sanitizer racecheck output into neutral records.

This module is part of the OSS package and deliberately has no dependency on
CUTracer's internal analysis taxonomy.  Consumers adapt the parsed records to
their own finding and reporting types.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class RacecheckSeverity(str, Enum):
    """Severity reported by a racecheck finding header."""

    ERROR = "error"
    WARNING = "warning"


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
    r"RACECHECK SUMMARY: (?P<total>\d+) hazards? displayed "
    r"\((?P<errors>\d+) errors?, (?P<warnings>\d+) warnings?\)"
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


def parse_racecheck_log(text: str) -> RacecheckParseResult:
    """Parse racecheck stdout without depending on internal CUTracer types.

    The parser is intentionally lenient: unrelated and interleaved child
    output is ignored, malformed fragments produce no finding, and a missing
    summary is represented by ``None``. Temporal RAW/WAR/WAW classification is
    not inferred because racecheck output does not establish execution order.
    """
    findings: List[RacecheckFinding] = []
    summary: Optional[RacecheckSummary] = None

    header: Optional[Dict[str, Optional[str]]] = None
    conflicts: List[Dict[str, Optional[str]]] = []
    raw_lines: List[str] = []

    def flush() -> None:
        nonlocal header, conflicts, raw_lines
        if header is not None:
            findings.append(_build_finding(header, conflicts, raw_lines))
        header = None
        conflicts = []
        raw_lines = []

    for line in text.splitlines():
        if not line.startswith(_PREFIX):
            continue
        body = line[len(_PREFIX) :].strip()

        summary_match = _SUMMARY_RE.search(body)
        if summary_match:
            summary = RacecheckSummary(
                total_hazards=int(summary_match["total"]),
                errors=int(summary_match["errors"]),
                warnings=int(summary_match["warnings"]),
                raw=body,
            )
            continue

        header_match = _HEADER_RE.match(body)
        if header_match:
            flush()
            header = header_match.groupdict()
            raw_lines = [body]
            continue

        if header is not None:
            conflict_match = _CONFLICT_RE.match(body)
            if conflict_match:
                conflicts.append(conflict_match.groupdict())
                raw_lines.append(body)
                continue
            flush()

    flush()
    return RacecheckParseResult(findings=findings, summary=summary)
