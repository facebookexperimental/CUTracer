# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Per-scope baselines of what a complete campaign already knew about.

The hard invariant is that a partial campaign can never replace a complete
baseline. It is enforced structurally rather than by convention: a
``ScopeBaseline`` can only be produced by :func:`baseline_from_scope`, which
returns ``None`` for anything short of a complete run, so there is no partial
value for a store to accept.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Protocol, Tuple

from cutracer.service.contracts import BaselineVerdict, parse_ymd_date, ScopeId


@dataclass(frozen=True)
class ScopeBaseline:
    """What the last complete campaign for one scope reported.

    Construct through :func:`baseline_from_scope`; the private marker keeps a
    partial campaign from being written directly.
    """

    scope: ScopeId
    correlation_keys: Tuple[str, ...]
    source_revision: str
    recorded_on: str
    _complete: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if not self._complete:
            raise ValueError(
                "ScopeBaseline must be built by baseline_from_scope; a partial "
                "campaign may not establish a baseline"
            )
        if parse_ymd_date(self.recorded_on) is None:
            raise ValueError(
                f"recorded_on must be YYYY-MM-DD, got {self.recorded_on!r}"
            )

    def to_dict(self) -> Dict[str, object]:
        return {
            "scope": self.scope.to_dict(),
            "correlation_keys": list(self.correlation_keys),
            "source_revision": self.source_revision,
            "recorded_on": self.recorded_on,
        }

    @classmethod
    def from_dict(cls, d: object) -> "ScopeBaseline":
        # A stored baseline was complete when it was written; that is the only
        # way it could have been stored.
        if not isinstance(d, dict):
            raise ValueError("baseline must be a JSON object")
        scope = d.get("scope")
        if not isinstance(scope, dict):
            raise ValueError("baseline scope must be a JSON object")
        keys = d.get("correlation_keys")
        if not isinstance(keys, list):
            raise ValueError("baseline correlation_keys must be a JSON array")
        if not all(isinstance(key, str) for key in keys):
            raise ValueError("baseline correlation_keys must contain only strings")
        source_revision = d.get("source_revision")
        if not isinstance(source_revision, str):
            raise ValueError("baseline source_revision must be a string")
        recorded_on = d.get("recorded_on")
        if not isinstance(recorded_on, str):
            raise ValueError("baseline recorded_on must be a string")
        return cls(
            scope=ScopeId.from_dict(scope),
            correlation_keys=tuple(keys),
            source_revision=source_revision,
            recorded_on=recorded_on,
            _complete=True,
        )


def baseline_from_scope(
    scope: ScopeId,
    correlation_keys: Iterable[str],
    *,
    is_complete: bool,
    source_revision: str,
    recorded_on: str,
) -> Optional[ScopeBaseline]:
    """Build a baseline, or ``None`` when the campaign did not cover its scope."""
    if not is_complete:
        return None
    return ScopeBaseline(
        scope=scope,
        correlation_keys=tuple(sorted(set(correlation_keys))),
        source_revision=source_revision,
        recorded_on=recorded_on,
        _complete=True,
    )


@dataclass(frozen=True)
class BaselineDiff:
    """How one scope's current findings compare with its stored baseline."""

    scope: ScopeId
    verdict: BaselineVerdict
    #: Regressions against a known baseline. Empty unless a real comparison
    #: happened, so a consumer may alert on it without re-checking the verdict.
    new_keys: Tuple[str, ...] = ()
    resolved_keys: Tuple[str, ...] = ()
    unchanged_keys: Tuple[str, ...] = ()
    #: What the scope currently reports when there was nothing to compare to.
    observed_keys: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, object]:
        return {
            "scope": self.scope.to_dict(),
            "verdict": self.verdict.value,
            "new_keys": list(self.new_keys),
            "resolved_keys": list(self.resolved_keys),
            "unchanged_keys": list(self.unchanged_keys),
            "observed_keys": list(self.observed_keys),
        }


def diff_against_baseline(
    scope: ScopeId,
    correlation_keys: Iterable[str],
    baseline: Optional[ScopeBaseline],
) -> BaselineDiff:
    """Classify a scope's current findings against what was known before."""
    current = set(correlation_keys)
    if baseline is None:
        # Nothing to compare against, so nothing is proven new. The findings
        # are reported separately from new_keys so that a consumer alerting on
        # regressions cannot mistake a first-ever run for one.
        return BaselineDiff(
            scope=scope,
            verdict=BaselineVerdict.NO_BASELINE,
            observed_keys=tuple(sorted(current)),
        )
    known = set(baseline.correlation_keys)
    new = current - known
    resolved = known - current
    unchanged = current & known
    if new:
        verdict = BaselineVerdict.NEW
    elif resolved:
        verdict = BaselineVerdict.RESOLVED
    else:
        verdict = BaselineVerdict.UNCHANGED
    return BaselineDiff(
        scope=scope,
        verdict=verdict,
        new_keys=tuple(sorted(new)),
        resolved_keys=tuple(sorted(resolved)),
        unchanged_keys=tuple(sorted(unchanged)),
    )


class BaselineStore(Protocol):
    def load(self, scope: ScopeId) -> Optional[ScopeBaseline]: ...

    def record(self, baseline: ScopeBaseline) -> None: ...


class FileBaselineStore:
    """One JSON document per scope, replaced atomically."""

    def __init__(self, root: str) -> None:
        self._root = root

    def _path(self, scope: ScopeId) -> str:
        # Scope keys carry "::" and path separators from pytest ids, so the
        # name is reduced to a portable alphabet; a digest keeps two keys that
        # reduce to the same text from sharing a file.
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", scope.key)[:120]
        digest = hashlib.sha256(scope.key.encode("utf-8")).hexdigest()[:12]
        return os.path.join(self._root, f"{safe}.{digest}.json")

    def load(self, scope: ScopeId) -> Optional[ScopeBaseline]:
        try:
            with open(self._path(scope), "r", encoding="utf-8") as handle:
                return ScopeBaseline.from_dict(json.load(handle))
        except FileNotFoundError:
            return None

    def record(self, baseline: ScopeBaseline) -> None:
        os.makedirs(self._root, exist_ok=True)
        path = self._path(baseline.scope)
        handle = tempfile.NamedTemporaryFile(
            "w", dir=self._root, delete=False, suffix=".tmp", encoding="utf-8"
        )
        # Captured before the try so the cleanup path cannot depend on the
        # creation having succeeded.
        temp_path = handle.name
        try:
            with handle:
                json.dump(baseline.to_dict(), handle, indent=2)
            os.replace(temp_path, path)
        except BaseException:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise


def scopes_for(
    tools: Iterable[str],
    test_family: str,
) -> List[ScopeId]:
    """The scopes a campaign over ``tools`` is accountable for."""
    return [ScopeId(test_family=test_family, tool=tool) for tool in sorted(set(tools))]
