# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Parse NVIDIA Compute Sanitizer output into neutral records.

This module is part of the OSS package and deliberately has no dependency on
CUTracer's internal analysis taxonomy.  Consumers adapt the parsed records to
their own finding and reporting types.

Two layers are provided:

* the racecheck-specific layer (:func:`parse_racecheck_log`), which recovers
  the paired accesses that only racecheck reports;
* the tool-agnostic checker layer (:func:`parse_checker_log`), which covers all
  four tools with one issue/site/summary shape so that downstream evidence
  conversion does not need a per-tool branch.
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
    INTERNAL_TOOL_ERROR = "internal_tool_error"


class CheckerTool(str, Enum):
    """The four Compute Sanitizer tools this parser understands."""

    INITCHECK = "initcheck"
    MEMCHECK = "memcheck"
    RACECHECK = "racecheck"
    SYNCCHECK = "synccheck"


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

        Once at least one report is parsed, truncated output stays complete:
        racecheck's summary remains authoritative for the omitted remainder.
        A positive summary with no parsed report is still incomplete because
        truncation alone cannot prove what kind of finding was omitted.
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
_INTERNAL_TOOL_ERROR_PREFIX = "Internal Sanitizer Error:"
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
        if body.startswith(_INTERNAL_TOOL_ERROR_PREFIX):
            self.flush()
            self.diagnose(RacecheckParseDiagnostic.INTERNAL_TOOL_ERROR)
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


# ---------------------------------------------------------------------------
# Tool-agnostic checker layer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckerSite:
    """One code location attributed to an issue.

    ``file``/``line`` are absent whenever the sanitizer could not resolve the
    instruction back to source, which is common for kernels compiled without
    line info.  Consumers must treat the symbol as the only always-present
    identity.
    """

    symbol: str
    offset: Optional[str] = None
    file: Optional[str] = None
    line: Optional[int] = None

    @property
    def location(self) -> Optional[str]:
        """``file:line`` when the sanitizer resolved one, else ``None``."""
        if self.file is None or self.line is None:
            return None
        return f"{self.file}:{self.line}"


@dataclass(frozen=True)
class CheckerIssue:
    """One diagnostic block, shaped identically for all four tools."""

    message: str
    sites: List[CheckerSite] = field(default_factory=list)
    severity: RacecheckSeverity = RacecheckSeverity.ERROR
    hazards: Optional[int] = None
    raw_block: str = ""

    @property
    def primary_site(self) -> Optional[CheckerSite]:
        """The site the sanitizer attributed the issue to, if it reported one."""
        return self.sites[0] if self.sites else None


@dataclass(frozen=True)
class CheckerSummary:
    """Aggregate counts from a tool's summary line.

    ``total_hazards``/``warnings`` are racecheck-only; the error-summary tools
    report a single error count.  ``not_printed`` records how many diagnostics
    the tool suppressed, which is what makes the printed block count a lower
    bound rather than the truth.
    """

    errors: int
    warnings: int = 0
    total_hazards: int = 0
    not_printed: Optional[int] = None
    raw: str = ""

    @property
    def is_positive(self) -> bool:
        return self.errors > 0 or self.warnings > 0 or self.total_hazards > 0

    @property
    def is_zero(self) -> bool:
        return not self.is_positive


@dataclass(frozen=True)
class CheckerParseResult:
    """Structured output from one log of any Compute Sanitizer tool."""

    tool: CheckerTool
    issues: List[CheckerIssue] = field(default_factory=list)
    summary: Optional[CheckerSummary] = None
    diagnostics: List[RacecheckParseDiagnostic] = field(default_factory=list)
    output_truncated: bool = False

    @property
    def is_complete(self) -> bool:
        """Whether the log proves what the run did.

        Output truncation does not make a run incomplete: the summary counts
        stay authoritative even when the tool stops printing blocks.  Only a
        missing, malformed, or self-contradictory summary does.
        """
        return self.summary is not None and not self.diagnostics


_ERROR_SUMMARY_RE = re.compile(r"^ERROR SUMMARY: (?P<errors>\d+) errors?$")
_ERROR_NOT_PRINTED_RE = re.compile(
    r"^ERROR SUMMARY: (?P<count>\d+) errors? were not printed\b"
)
_SITE_RE = re.compile(r"^at (?P<site>.+)$")
_SITE_SOURCE_RE = re.compile(r"^(?P<symbol>.*?)\s+in (?P<file>.+?):(?P<line>\d+)$")
_HAZARD_SUFFIX_RE = re.compile(r"\s*\[(?P<hazards>\d+) hazards?\]$")

_MEMCHECK_MESSAGE_RE = re.compile(
    r"^(?:Invalid|Misaligned|Out of bounds|Program hit|Leaked) "
)
# CUDA API errors and leaks are raised on the host, outside any kernel, so the
# sanitizer never attributes them to a device ``at`` site.
_MEMCHECK_HOST_MESSAGE_RE = re.compile(r"^(?:Program hit|Leaked) ")
_MEMCHECK_ADDRESS_SUFFIX_RE = re.compile(r"\s+at 0x[0-9a-fA-F]+$")


def _parse_site(raw: str) -> CheckerSite:
    """Split ``symbol+0xOFFSET [in file:line]`` into its parts.

    The symbol is a demangled C++ name and can itself contain spaces, angle
    brackets, parentheses, and ``+`` characters, so the offset is taken from
    the right rather than by splitting on the first delimiter.
    """
    text = _HAZARD_SUFFIX_RE.sub("", raw.strip())
    file: Optional[str] = None
    line: Optional[int] = None
    source_match = _SITE_SOURCE_RE.match(text)
    if source_match:
        text = source_match["symbol"].strip()
        file = source_match["file"]
        line = int(source_match["line"])

    symbol = text
    offset: Optional[str] = None
    offset_match = re.search(r"\+0x[0-9a-fA-F]+$", text)
    if offset_match is not None and offset_match.start() > 0:
        symbol = text[: offset_match.start()]
        offset = text[offset_match.start() + 1 :]
    return CheckerSite(symbol=symbol, offset=offset, file=file, line=line)


def _checker_message(tool: CheckerTool, body: str) -> Optional[str]:
    """Return the normalized issue message in ``body``, if it starts one."""
    if tool == CheckerTool.INITCHECK:
        return body if body.startswith("Uninitialized ") else None
    if tool == CheckerTool.SYNCCHECK:
        if not body.startswith("Barrier error detected."):
            return None
        return body.removesuffix(".")
    if tool == CheckerTool.MEMCHECK:
        if not _MEMCHECK_MESSAGE_RE.match(body):
            return None
        # Memcheck appends the faulting address to the message; the address is
        # per-run noise and would defeat any grouping by issue kind.
        return _MEMCHECK_ADDRESS_SUFFIX_RE.sub("", body)
    return None


def _checker_issue_at(
    tool: CheckerTool,
    bodies: List[Optional[str]],
    index: int,
) -> Optional[CheckerIssue]:
    body = bodies[index]
    if body is None:
        return None
    message = _checker_message(tool, body)
    if message is None:
        return None

    sites: List[CheckerSite] = []
    raw_lines = [body]
    sanitizer_lines = 0
    for position in range(index + 1, len(bodies)):
        following = bodies[position]
        if following is None:
            continue
        sanitizer_lines += 1
        if sanitizer_lines > _SITE_LOOKAHEAD:
            break
        if (
            _checker_message(tool, following) is not None
            or "ERROR SUMMARY:" in following
        ):
            break
        site_match = _SITE_RE.match(following)
        if site_match:
            sites.append(_parse_site(site_match["site"]))
        raw_lines.append(following)
    return CheckerIssue(
        message=message,
        sites=sites,
        raw_block="\n".join(raw_lines),
    )


@dataclass
class _ErrorCheckerParser:
    tool: CheckerTool
    issues: List[CheckerIssue] = field(default_factory=list)
    diagnostics: List[RacecheckParseDiagnostic] = field(default_factory=list)
    summary: Optional[CheckerSummary] = None
    summary_markers: int = 0
    not_printed: Optional[int] = None
    output_truncated: bool = False

    def diagnose(self, issue: RacecheckParseDiagnostic) -> None:
        if issue not in self.diagnostics:
            self.diagnostics.append(issue)

    def consume_summary(self, body: str) -> bool:
        if "ERROR SUMMARY:" in body:
            truncated_match = _ERROR_NOT_PRINTED_RE.match(body)
            if truncated_match:
                # A companion note to the real summary, not a second summary.
                self.output_truncated = True
                self.not_printed = int(truncated_match["count"])
                return True
            self.summary_markers += 1
            summary_match = _ERROR_SUMMARY_RE.match(body)
            if summary_match:
                self.summary = CheckerSummary(
                    errors=int(summary_match["errors"]), raw=body
                )
            else:
                self.diagnose(RacecheckParseDiagnostic.MALFORMED_SUMMARY)
            return True
        return False

    def consume_issue(
        self,
        bodies: List[Optional[str]],
        index: int,
    ) -> None:
        issue = _checker_issue_at(self.tool, bodies, index)
        if issue is None:
            return
        if not issue.sites and not _MEMCHECK_HOST_MESSAGE_RE.match(issue.message):
            self.diagnose(RacecheckParseDiagnostic.INCOMPLETE_FINDING)
        self.issues.append(issue)

    def finish(self) -> CheckerParseResult:
        if self.summary_markers == 0:
            self.diagnose(RacecheckParseDiagnostic.MISSING_SUMMARY)
        elif self.summary_markers > 1:
            self.diagnose(RacecheckParseDiagnostic.MULTIPLE_SUMMARIES)

        if self.summary is not None:
            if self.not_printed is not None:
                self.summary = CheckerSummary(
                    errors=self.summary.errors,
                    not_printed=self.not_printed,
                    raw=self.summary.raw,
                )
            printed_nothing = (
                self.summary.is_positive
                and not self.issues
                and not self.output_truncated
            )
            if printed_nothing or (self.summary.is_zero and self.issues):
                self.diagnose(RacecheckParseDiagnostic.SUMMARY_FINDING_MISMATCH)

        return CheckerParseResult(
            tool=self.tool,
            issues=self.issues,
            summary=self.summary,
            diagnostics=self.diagnostics,
            output_truncated=self.output_truncated,
        )


def _parse_error_checker_log(tool: CheckerTool, text: str) -> CheckerParseResult:
    """Parse the tools that report a single ``ERROR SUMMARY`` line."""
    bodies: List[Optional[str]] = [_sanitizer_body(line) for line in text.splitlines()]
    parser = _ErrorCheckerParser(tool)
    for index, body in enumerate(bodies):
        if body is None:
            continue
        if body.startswith(_INTERNAL_TOOL_ERROR_PREFIX):
            parser.diagnose(RacecheckParseDiagnostic.INTERNAL_TOOL_ERROR)
            continue
        if parser.consume_summary(body):
            continue
        parser.consume_issue(bodies, index)
    return parser.finish()


def _racecheck_as_checker_result(text: str) -> CheckerParseResult:
    """Project the racecheck-specific result onto the shared checker shape."""
    parsed = parse_racecheck_log(text)
    issues: List[CheckerIssue] = []
    for finding in parsed.findings:
        access_kinds = finding.access_kinds
        sites = [
            CheckerSite(
                symbol=access.function,
                offset=access.offset,
                file=access.file,
                line=access.line,
            )
            for access in finding.accesses
        ]
        issues.append(
            CheckerIssue(
                message=(
                    f"Race reported between {access_kinds[0]} access"
                    if access_kinds
                    else "Race reported"
                ),
                sites=sites,
                severity=finding.severity,
                hazards=finding.max_hazards,
                raw_block=finding.raw_block,
            )
        )
    for invalid in parsed.invalid_accesses:
        access = invalid.access
        issues.append(
            CheckerIssue(
                message=invalid.message,
                sites=(
                    []
                    if access is None
                    else [
                        CheckerSite(
                            symbol=access.function,
                            offset=access.offset,
                            file=access.file or None,
                            line=access.line or None,
                        )
                    ]
                ),
                raw_block=invalid.raw_block,
            )
        )
    summary = (
        None
        if parsed.summary is None
        else CheckerSummary(
            errors=parsed.summary.errors,
            warnings=parsed.summary.warnings,
            total_hazards=parsed.summary.total_hazards,
            raw=parsed.summary.raw,
        )
    )
    return CheckerParseResult(
        tool=CheckerTool.RACECHECK,
        issues=issues,
        summary=summary,
        diagnostics=list(parsed.diagnostics),
        output_truncated=parsed.output_truncated,
    )


def parse_checker_log(tool: CheckerTool, text: str) -> CheckerParseResult:
    """Parse any Compute Sanitizer tool's stdout into one neutral shape.

    Racecheck keeps its richer paired-access parse for consumers that need the
    conflicting accesses; this entry point projects it onto the same
    issue/site/summary shape the other three tools produce so that evidence
    conversion stays tool-agnostic.
    """
    if tool == CheckerTool.RACECHECK:
        return _racecheck_as_checker_result(text)
    return _parse_error_checker_log(tool, text)
