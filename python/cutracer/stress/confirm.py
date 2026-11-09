# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Confirm a ``cutracer stress`` finding by reducing and replaying it.

A search hit is one run. ``confirm_finding`` replays the triggering config
under the same targeting (``cutracer reduce``'s initial check), reduces it to
the delay points that matter with majority voting, and replays the minimal
config once more. Only a finding that survives all three stays a finding.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from cutracer.reduce.reduce import (
    ConfigDoesNotTriggerError,
    reduce_bisect,
    ReduceConfig,
    ReplayConfig,
    ReplayExecutionError,
    ReplayOutcome,
    ReplayResult,
    run_replay,
)
from cutracer.runner import RunTarget
from cutracer.stress.stress import (
    _delay_patterns,
    _remaining_s,
    _run_timeout,
    MIN_RUN_BUDGET_S,
    StressConfig,
    StressResult,
    SYMPTOM_HANG,
    TriggeringConfig,
    Verdict,
    VERDICT_FINDING,
    VERDICT_INCOMPLETE,
    VERDICT_SUSPECT,
)

CONFIRMED = "confirmed"
# A replay of the triggering or the minimal config did not reproduce.
NOT_REPRODUCED = "not_reproduced"
# A replay could not be evaluated, or the time budget ran out.
UNCONFIRMED = "unconfirmed"


@dataclass(frozen=True)
class Confirmation:
    status: str
    reason: str
    replays: int = 0
    minimal_config_path: Optional[str] = None
    # Each point of the minimal config: kernel, PC and SASS.
    essential_points: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "replays": self.replays,
            "minimal_config_path": self.minimal_config_path,
            "essential_points": list(self.essential_points),
        }


class _BudgetExhausted(Exception):
    pass


def replay_config(
    config: StressConfig,
    triggering: TriggeringConfig,
    *,
    cutracer_so: Optional[str],
    output_dir: Optional[str],
) -> ReplayConfig:
    """The replay of the triggering config with the targeting, kernel timeout
    and hang rule the search ran it under (the in-process form of the
    `Reduce:` line `cutracer stress` prints)."""
    target = triggering.target or {}
    warp = target.get("warp")
    rank = target.get("cluster_rank")
    delay_mode: Optional[str] = None
    if warp is not None:
        delay_mode = "fixed" if rank is None else "cluster_fixed"
    return ReplayConfig(
        oracle_argv=list(config.oracle_argv),
        cutracer_so=cutracer_so,
        kernel_filters=config.kernel_filters,
        exact_kernel_name=config.exact_kernel_name,
        exact_kernel_checksum=config.exact_kernel_checksum,
        delay_mode=delay_mode,
        delay_warpgroup_id=None if warp is not None else triggering.warpgroup_id,
        delay_warp_mask=None if warp is None else hex(1 << int(warp)),
        delay_cluster_cta_id=None if warp is None else rank,
        delay_patterns=_delay_patterns(config),
        kernel_timeout_s=config.kernel_timeout_s,
        hang_is_bug=triggering.symptom == SYMPTOM_HANG,
        output_dir=output_dir,
        timeout=config.timeout,
        cwd=config.cwd,
        base_env=config.base_env,
        # A search without an explicit list counts any non-zero exit as "not
        # reproduced"; a replay needs the list, and 1 is the oracle default.
        not_interesting_exit_codes=tuple(config.not_interesting_exit_codes or (1,)),
    )


def confirm_finding(  # noqa: C901
    config: StressConfig,
    result: StressResult,
    *,
    cutracer_so: Optional[str],
    runner: Optional[RunTarget] = None,
    deadline: Optional[float] = None,
    confidence_runs: int = 3,
) -> Confirmation:
    """Reduce the triggering config of `result` (the search `config` ran) and
    replay the minimal config once more. Writes `minimal_config.json` and the
    replays' scratch output under `config.output_dir`. No replay starts with
    less than `MIN_RUN_BUDGET_S` left before `deadline`."""
    triggering = result.triggering
    if triggering is None:
        raise ValueError("there is no triggering config to confirm")
    scratch = os.path.join(config.output_dir, "confirm")
    os.makedirs(scratch, exist_ok=True)
    replay = replay_config(
        config, triggering, cutracer_so=cutracer_so, output_dir=scratch
    )
    replays = 0

    def evaluate(path: str, candidate: ReplayConfig) -> ReplayResult:
        nonlocal replays
        remaining = _remaining_s(deadline)
        if remaining is not None and remaining < MIN_RUN_BUDGET_S:
            raise _BudgetExhausted
        replays += 1
        bounded = dataclasses.replace(
            candidate, timeout=_run_timeout(config, remaining)
        )
        return run_replay(path, bounded, runner=runner)

    minimal_path = os.path.join(config.output_dir, "minimal_config.json")
    try:
        reduced = reduce_bisect(
            ReduceConfig(
                config_path=triggering.config_path,
                output_path=minimal_path,
                replay=replay,
                replay_evaluator=evaluate,
            ),
            confidence_runs=confidence_runs,
        )
    except ConfigDoesNotTriggerError:
        return Confirmation(
            NOT_REPRODUCED,
            "the triggering config did not reproduce on replay",
            replays=replays,
        )
    except ReplayExecutionError as exc:
        return Confirmation(UNCONFIRMED, f"a replay failed: {exc}", replays=replays)
    except _BudgetExhausted:
        return Confirmation(
            UNCONFIRMED,
            "the time budget ran out while confirming",
            replays=replays,
        )
    minimal = reduced.minimal_config_path
    if not reduced.success or minimal is None:
        return Confirmation(
            NOT_REPRODUCED, "reduction kept no delay point", replays=replays
        )
    points = [
        {"kernel_name": p.kernel_name, "pc": str(p.pc_offset), "sass": p.sass}
        for p in reduced.essential_points
    ]

    def reduced_to(status: str, reason: str) -> Confirmation:
        return Confirmation(
            status,
            reason,
            replays=replays,
            minimal_config_path=minimal,
            essential_points=points,
        )

    try:
        final = evaluate(minimal, replay)
    except _BudgetExhausted:
        return reduced_to(
            UNCONFIRMED,
            "the time budget ran out before the minimal config's confirmation replay",
        )
    if final.outcome == ReplayOutcome.NOT_INTERESTING:
        return reduced_to(
            NOT_REPRODUCED,
            "the minimal config did not reproduce on its confirmation replay",
        )
    if final.outcome != ReplayOutcome.INTERESTING:
        return reduced_to(
            UNCONFIRMED,
            f"the confirmation replay failed: {final.error or final.outcome.value}",
        )
    return reduced_to(
        CONFIRMED,
        f"reduced to {len(points)} delay point(s); the minimal config reproduced again",
    )


def confirmed_verdict(
    verdict: Verdict, confirmation: Optional[Confirmation]
) -> Verdict:
    """A finding stays one only if its confirmation reproduced: a config that
    stops reproducing on replay is suspect, and a confirmation that could not
    finish leaves the search incomplete."""
    if confirmation is None or verdict.status != VERDICT_FINDING:
        return verdict
    reasons = [*verdict.reasons, confirmation.reason]
    if confirmation.status == CONFIRMED:
        return Verdict(VERDICT_FINDING, reasons)
    if confirmation.status == NOT_REPRODUCED:
        return Verdict(VERDICT_SUSPECT, reasons)
    return Verdict(VERDICT_INCOMPLETE, reasons)
