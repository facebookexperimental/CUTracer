# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Versioned suppression of findings that are not this service's to report.

Suppression marks findings rather than discarding them: every suppressed
finding keeps an audit entry naming the rule and rule version that hid it, so a
wrong rule is visible in the report instead of silently shrinking the evidence.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Pattern, Sequence, Tuple

from cutracer.service.contracts import (
    FindingRecord,
    parse_ymd_date,
    SuppressionDecision,
    SuppressionRule,
)

# Sites owned by the framework the kernel under test runs inside, not by the
# Triton/TLX kernel that is the work unit. Measured on the all-fbtriton-UT
# sweep, these account for 265 of 320 recorded findings, every one of them
# under initcheck and every one attributed to an ATen elementwise kernel.
#
# Only patterns with evidence in that corpus are in v1. Adding cutlass, cub,
# thrust, or cuBLAS on suspicion would put unverified matchers in the path of
# real findings; they belong in a v2 raised when a real log demands them.
# The sanitizer adapter canonicalizes finding sources as
# ``compute_sanitizer/<tool>``; bare checker names are intentionally not matched.
FRAMEWORK_SITE_PATTERN: str = r"(?:^|[^A-Za-z0-9_])(?:at|c10)::"

DEFAULT_SUPPRESSION_RULES: Tuple[SuppressionRule, ...] = (
    SuppressionRule(
        rule_id="framework-kernel-site",
        version=1,
        reason=(
            "Finding is attributed to a framework kernel (ATen/c10), not to the "
            "Triton kernel under test."
        ),
        owner="triton_tools",
        kernel_pattern=FRAMEWORK_SITE_PATTERN,
        tools=("compute_sanitizer/initcheck",),
        tracking_ticket="framework-initcheck-noise-20260720",
        reviewed_by="triton_tools",
        reviewed_on="2026-09-05",
        review_due_on="2026-12-04",
        permanent=True,
    ),
)


@dataclass(frozen=True)
class SuppressedFinding:
    """One finding plus the audit entry explaining why it was hidden."""

    finding: FindingRecord
    decision: SuppressionDecision


@dataclass(frozen=True)
class SuppressionOutcome:
    """The partition of one finding set into actionable and suppressed."""

    kept: Tuple[FindingRecord, ...] = ()
    suppressed: Tuple[SuppressedFinding, ...] = ()

    @property
    def suppressed_findings(self) -> List[FindingRecord]:
        return [entry.finding for entry in self.suppressed]

    def audit(self) -> List[Dict[str, object]]:
        """Flat audit trail, one entry per suppressed finding."""
        return [
            {
                "error_type": entry.finding.error_type,
                "kernel_name": entry.finding.kernel_name,
                "decision": entry.decision.to_dict(),
            }
            for entry in self.suppressed
        ]


def _rule_is_active(rule: SuppressionRule, today: Optional[date]) -> bool:
    """Whether ``rule`` may fire.

    An expiring rule with no date context is treated as expired. Losing a
    suppression only produces extra noise, whereas honouring one whose validity
    cannot be established hides findings on an unproven basis.

    The comparison is on parsed dates rather than on the strings, so no rule
    can outlive its expiry through a spelling that happens to sort late.
    """
    if rule.expires_on is None:
        return True
    if today is None:
        return False
    expires = parse_ymd_date(rule.expires_on)
    if expires is None:
        return False
    return today < expires


def _matches(
    rule: SuppressionRule,
    pattern: Optional[Pattern[str]],
    finding: FindingRecord,
) -> bool:
    if rule.kernel_pattern is None and not rule.tools and not rule.error_types:
        # A rule that narrows nothing would suppress everything; refuse rather
        # than let an under-specified rule silence the whole campaign.
        return False
    if rule.tools and finding.source_tool not in rule.tools:
        return False
    if rule.error_types and finding.error_type not in rule.error_types:
        return False
    if rule.kernel_pattern is not None:
        if pattern is None:
            return False
        if finding.kernel_name is None or not pattern.search(finding.kernel_name):
            return False
    return True


class SuppressionPolicy:
    """Applies an ordered, versioned rule set to a finding set."""

    def __init__(
        self, rules: Sequence[SuppressionRule] = DEFAULT_SUPPRESSION_RULES
    ) -> None:
        self._rules: Tuple[SuppressionRule, ...] = tuple(rules)
        # Compiled patterns are held positionally: two rules may legitimately
        # share an id and version in a hand-written policy, and keying by that
        # pair would silently give one of them the other's matcher.
        self._patterns: Tuple[Optional[Pattern[str]], ...] = tuple(
            None if rule.kernel_pattern is None else re.compile(rule.kernel_pattern)
            for rule in self._rules
        )

    @property
    def rules(self) -> Tuple[SuppressionRule, ...]:
        return self._rules

    def apply(
        self,
        findings: Sequence[FindingRecord],
        *,
        today: Optional[str] = None,
    ) -> SuppressionOutcome:
        """Partition ``findings``; ``today`` is a ``YYYY-MM-DD`` date for expiry."""
        current_date = None if today is None else parse_ymd_date(today)
        if today is not None and current_date is None:
            raise ValueError(f"today must be YYYY-MM-DD, got {today!r}")
        kept: List[FindingRecord] = []
        suppressed: List[SuppressedFinding] = []
        for finding in findings:
            decision: Optional[SuppressionDecision] = None
            for rule, pattern in zip(self._rules, self._patterns):
                if not _rule_is_active(rule, current_date):
                    continue
                if _matches(rule, pattern, finding):
                    decision = SuppressionDecision(
                        rule_id=rule.rule_id,
                        rule_version=rule.version,
                        reason=rule.reason,
                        owner=rule.owner,
                        expires_on=rule.expires_on,
                        tracking_ticket=rule.tracking_ticket,
                        reviewed_by=rule.reviewed_by,
                        reviewed_on=rule.reviewed_on,
                        review_due_on=rule.review_due_on,
                        permanent=rule.permanent,
                    )
                    break
            if decision is None:
                kept.append(finding)
            else:
                suppressed.append(SuppressedFinding(finding=finding, decision=decision))
        return SuppressionOutcome(kept=tuple(kept), suppressed=tuple(suppressed))


def load_suppression_policy(path: str) -> SuppressionPolicy:
    """Load an ordered rule set from a JSON ``{"rules": [...]}`` document."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid suppression policy JSON: {error}") from error
    if not isinstance(document, dict):
        raise ValueError("suppression policy must be a JSON object")
    rules = document.get("rules")
    if not isinstance(rules, list):
        raise ValueError("suppression policy rules must be a JSON array")
    try:
        return SuppressionPolicy([SuppressionRule.from_dict(rule) for rule in rules])
    except (KeyError, TypeError, ValueError, re.error) as error:
        raise ValueError(f"invalid suppression rule: {error}") from error
