"""Approval gate for production suppression policies."""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
from collections import Counter

from cutracer.service.contracts import parse_ymd_date, SuppressionRule
from cutracer.service.suppression import SuppressionPolicy


@dataclasses.dataclass(frozen=True)
class SuppressionReviewIssue:
    rule_id: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class SuppressionReviewReport:
    policy_digest: str
    reviewed_on: str
    errors: tuple[SuppressionReviewIssue, ...]

    @property
    def approved(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_digest": self.policy_digest,
            "reviewed_on": self.reviewed_on,
            "approved": self.approved,
            "errors": [error.to_dict() for error in self.errors],
        }


class SuppressionGovernanceError(ValueError):
    """A suppression policy is not eligible for production use."""


def _display_rule_id(rule_id: str) -> str:
    return rule_id if rule_id.strip() else "<missing>"


def _issue(rule: SuppressionRule, code: str, message: str) -> SuppressionReviewIssue:
    return SuppressionReviewIssue(_display_rule_id(rule.rule_id), code, message)


def _review_metadata_issues(
    rule: SuppressionRule,
    today: datetime.date,
    max_review_days: int,
) -> list[SuppressionReviewIssue]:
    issues: list[SuppressionReviewIssue] = []
    missing = [
        name
        for name, value in (
            ("tracking_ticket", rule.tracking_ticket),
            ("reviewed_by", rule.reviewed_by),
            ("reviewed_on", rule.reviewed_on),
            ("review_due_on", rule.review_due_on),
        )
        if not value or not value.strip()
    ]
    if missing:
        issues.append(_issue(rule, "missing_review_metadata", ", ".join(missing)))
        return issues
    reviewed = parse_ymd_date(rule.reviewed_on or "")
    due = parse_ymd_date(rule.review_due_on or "")
    if reviewed is None or due is None or due <= reviewed:
        issues.append(_issue(rule, "invalid_review_window", "review dates are invalid"))
        return issues
    if (due - reviewed).days > max_review_days:
        issues.append(
            _issue(rule, "review_window_too_long", f"maximum is {max_review_days} days")
        )
    if today > due:
        issues.append(
            _issue(rule, "review_overdue", f"review was due {due.isoformat()}")
        )
    return issues


def _rule_issues(
    rule: SuppressionRule,
    today: datetime.date,
    max_review_days: int,
) -> list[SuppressionReviewIssue]:
    issues = _review_metadata_issues(rule, today, max_review_days)
    invalid_identity_fields: list[str] = []
    if not rule.rule_id.strip() or rule.rule_id != rule.rule_id.strip():
        invalid_identity_fields.append("rule_id")
    if rule.version <= 0:
        invalid_identity_fields.append("version")
    if not rule.reason.strip():
        invalid_identity_fields.append("reason")
    if not rule.owner.strip():
        invalid_identity_fields.append("owner")
    if invalid_identity_fields:
        issues.append(
            _issue(
                rule,
                "invalid_identity",
                f"invalid fields: {', '.join(invalid_identity_fields)}",
            )
        )
    if rule.kernel_pattern is None and not rule.tools and not rule.error_types:
        issues.append(
            _issue(rule, "unbounded_matcher", "at least one matcher is required")
        )
    if rule.expires_on is None and not rule.permanent:
        issues.append(
            _issue(rule, "missing_expiry", "non-permanent rule requires expires_on")
        )
    expiry = None if rule.expires_on is None else parse_ymd_date(rule.expires_on)
    if expiry is not None and today >= expiry:
        issues.append(_issue(rule, "expired", f"rule expired {expiry.isoformat()}"))
    return issues


def _policy_digest(policy: SuppressionPolicy) -> str:
    encoded = json.dumps(
        [rule.to_dict() for rule in policy.rules],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def review_suppression_policy(
    policy: SuppressionPolicy,
    *,
    today: str,
    max_review_days: int = 90,
) -> SuppressionReviewReport:
    review_date = parse_ymd_date(today)
    if review_date is None:
        raise ValueError("suppression review today must be YYYY-MM-DD")
    if max_review_days <= 0:
        raise ValueError("suppression max review days must be positive")
    issues: list[SuppressionReviewIssue] = []
    for duplicate, count in sorted(
        Counter(rule.rule_id for rule in policy.rules).items()
    ):
        if count > 1:
            issues.append(
                SuppressionReviewIssue(
                    _display_rule_id(duplicate),
                    "duplicate_rule_id",
                    "rule id must be unique",
                )
            )
    for rule in policy.rules:
        issues.extend(_rule_issues(rule, review_date, max_review_days))
    return SuppressionReviewReport(_policy_digest(policy), today, tuple(issues))


def require_governed_suppression_policy(
    policy: SuppressionPolicy,
    *,
    today: str,
    max_review_days: int = 90,
) -> SuppressionReviewReport:
    report = review_suppression_policy(
        policy,
        today=today,
        max_review_days=max_review_days,
    )
    if report.approved:
        return report
    summary = "; ".join(f"{issue.rule_id}:{issue.code}" for issue in report.errors)
    raise SuppressionGovernanceError(f"suppression policy review failed: {summary}")
