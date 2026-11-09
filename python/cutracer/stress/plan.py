# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Run several ``cutracer stress`` modes as one search with one verdict.

Each mode keeps its own delay placement and defaults (``cutracer.stress.modes``)
and writes its own report under ``<output_dir>/<mode>``. Modes that instrument
the same points the same way share one control arm, which runs once. The modes
then run in order under one optional time budget, and by default the search
stops at the first mode that reproduces.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from cutracer.runner import RunTarget
from cutracer.stress.modes import get_mode
from cutracer.stress.stress import (
    _baseline_from_runs,
    _BaselineResult,
    _delay_patterns,
    _remaining_s,
    _run_baseline,
    ATTRIBUTION_BASELINE_INCOMPLETE,
    ATTRIBUTION_NATURAL,
    ATTRIBUTION_NO_BASELINE,
    ATTRIBUTION_NOT_REPRODUCED,
    MIN_RUN_BUDGET_S,
    result_verdict,
    run_stress,
    save_report,
    StressConfig,
    StressResult,
    Verdict,
    VERDICT_CLEAN,
    VERDICT_FINDING,
    VERDICT_INCOMPLETE,
    VERDICT_SUSPECT,
)

# What became of one mode of a plan.
MODE_RAN = "ran"
# The mode does not apply to the instrumented kernels (for example isolate on a
# kernel without mbarrier waits). It does not make the plan incomplete.
MODE_SKIPPED = "skipped"
# The mode applied but did not run: an earlier mode reproduced, the time budget
# ran out, or its control arm produced nothing to discover targets from.
MODE_NOT_RUN = "not_run"


@dataclass
class ModeRun:
    """One mode of a plan, its search config, and what became of it."""

    config: StressConfig
    status: str = MODE_NOT_RUN
    reason: Optional[str] = None
    result: Optional[StressResult] = None

    @property
    def mode(self) -> str:
        return self.config.strategy

    @property
    def report_path(self) -> str:
        return os.path.join(self.config.output_dir, "stress_report.json")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "status": self.status,
            "reason": self.reason,
            "output_dir": self.config.output_dir,
            "report": None if self.result is None else self.result.to_dict(),
        }


@dataclass
class PlanResult:
    runs: List[ModeRun]
    verdict: Verdict
    budget_s: Optional[float]
    elapsed_s: float
    # One entry per control arm: the modes it served and its counts.
    baselines: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def triggering_run(self) -> Optional[ModeRun]:
        for run in self.runs:
            if run.result is not None and run.result.triggering is not None:
                return run
        return None

    def to_dict(self) -> Dict[str, Any]:
        """The plan report: the verdict, each mode's own report, and the same
        top-level keys a single-mode report has (summed over the modes that
        ran), so a reader of either finds `reproduced` and `triggering_config`
        in the same place."""
        results = [run.result for run in self.runs if run.result is not None]
        trigger = self.triggering_run
        triggering = None if trigger is None else trigger.result
        control = triggering or _representative(results)
        return {
            "strategy": ",".join(run.mode for run in self.runs),
            "verdict": self.verdict.to_dict(),
            "budget_s": self.budget_s,
            "elapsed_s": round(self.elapsed_s, 1),
            "reproduced": triggering is not None,
            "triggering_mode": None if trigger is None else trigger.mode,
            "triggering_config": (
                None
                if triggering is None or triggering.triggering is None
                else triggering.triggering.to_dict()
            ),
            "attribution": (
                triggering.attribution
                if triggering is not None
                else _combined_attribution(results)
            ),
            **{
                key: sum(getattr(result, key) for result in results)
                for key in (
                    "completed_trials",
                    "reproductions",
                    "unattributed_reproductions",
                    "infra_errors",
                    "timed_out",
                    "hangs",
                )
            },
            # The control arm of the mode the attribution came from.
            **{
                key: None if control is None else getattr(control, key)
                for key in (
                    "baseline_requested",
                    "baseline_completed",
                    "baseline_reproductions",
                    "baseline_timed_out",
                    "baseline_infra_errors",
                    "baseline_hangs",
                )
            },
            "baselines": list(self.baselines),
            "modes": [run.to_dict() for run in self.runs],
        }


def _representative(results: Sequence[StressResult]) -> Optional[StressResult]:
    """The result whose attribution `_combined_attribution` reports."""
    for attribution in (ATTRIBUTION_NATURAL, ATTRIBUTION_BASELINE_INCOMPLETE):
        for result in results:
            if result.attribution == attribution:
                return result
    return results[0] if results else None


def _combined_attribution(results: Sequence[StressResult]) -> str:
    attributions = {result.attribution for result in results}
    for attribution in (ATTRIBUTION_NATURAL, ATTRIBUTION_BASELINE_INCOMPLETE):
        if attribution in attributions:
            return attribution
    if results and attributions == {ATTRIBUTION_NO_BASELINE}:
        return ATTRIBUTION_NO_BASELINE
    return ATTRIBUTION_NOT_REPRODUCED


def _profile(config: StressConfig) -> tuple[Any, ...]:
    """What makes two modes' control runs the same run: everything that changes
    what is instrumented or how a run is bounded, but not how a delay is
    placed (that is off in a control run) or how its outcome is counted."""
    return (
        _delay_patterns(config),
        config.kernel_filters,
        config.exact_kernel_name,
        config.exact_kernel_checksum,
        config.kernel_timeout_s,
        config.timeout,
        config.baseline_runs,
    )


def _explicit_targets(config: StressConfig) -> bool:
    return config.isolate_ranks is not None and config.isolate_warps is not None


def plan_verdict(runs: Sequence[ModeRun]) -> Verdict:
    """The plan's verdict: a finding in any mode, else a suspect failure in any
    mode, else clean only if every mode that applied ran clean."""
    verdicts = [
        (run.mode, result_verdict(run.result)) for run in runs if run.result is not None
    ]
    for status in (VERDICT_FINDING, VERDICT_SUSPECT):
        found = [
            f"{mode}: {reason}"
            for mode, verdict in verdicts
            if verdict.status == status
            for reason in verdict.reasons
        ]
        if found:
            return Verdict(status, found)
    reasons = [
        f"{mode}: {reason}"
        for mode, verdict in verdicts
        if verdict.status == VERDICT_INCOMPLETE
        for reason in verdict.reasons
    ]
    reasons += [
        f"{run.mode}: did not run ({run.reason})"
        for run in runs
        if run.status == MODE_NOT_RUN
    ]
    if not verdicts:
        reasons.append("no mode ran")
    return Verdict(VERDICT_INCOMPLETE, reasons) if reasons else Verdict(VERDICT_CLEAN)


def _baseline_summary(
    modes: Sequence[str], baseline: _BaselineResult, log_path: Optional[str]
) -> Dict[str, Any]:
    return {
        "modes": list(modes),
        "requested": baseline.requested,
        "runs": len(baseline.runs),
        "log_path": log_path,
        "geometries": [
            {"checksum": g.checksum, "warps": g.warps, "cluster_size": g.cluster_size}
            for g in baseline.geometries
        ],
    }


def run_stress_plan(  # noqa: C901
    configs: Sequence[StressConfig],
    *,
    output_dir: str,
    budget_s: Optional[float] = None,
    stop_on_first: bool = True,
    cutracer_so: Optional[str] = None,
    runner: Optional[RunTarget] = None,
) -> PlanResult:
    """Run one search per config (one per mode), in order.

    Each config names its mode in `strategy` and its own `output_dir`. A
    control arm runs once per instrumentation profile (see `_profile`), the
    first time a mode needs it, and every mode of that profile counts it under
    its own oracle contract. With `stop_on_first`, the modes after the first
    one that reproduces do not run. No run starts with less than
    `MIN_RUN_BUDGET_S` of `budget_s` left.
    """
    if not configs:
        raise ValueError("a stress plan needs at least one mode")
    os.makedirs(output_dir, exist_ok=True)
    started = time.monotonic()
    deadline = None if budget_s is None else started + budget_s
    runs = [ModeRun(config) for config in configs]
    members: Dict[tuple[Any, ...], List[ModeRun]] = {}
    for run in runs:
        members.setdefault(_profile(run.config), []).append(run)
    baselines: Dict[tuple[Any, ...], _BaselineResult] = {}
    summaries: List[Dict[str, Any]] = []
    stopped: Optional[str] = None
    for run in runs:
        if stopped is not None:
            run.reason = stopped
            continue
        remaining = _remaining_s(deadline)
        if remaining is not None and remaining < MIN_RUN_BUDGET_S:
            run.reason = "the time budget ran out before this mode"
            continue
        profile = _profile(run.config)
        if profile not in baselines:
            sharing = members[profile]
            os.makedirs(run.config.output_dir, exist_ok=True)
            baseline = _run_baseline(
                run.config,
                cutracer_so=cutracer_so,
                runner=runner,
                capture_geometry=any(
                    get_mode(member.mode).isolates_targets for member in sharing
                ),
                deadline=deadline,
            )
            log_path = None
            if baseline.logs:
                log_path = os.path.join(output_dir, f"baseline-{len(summaries)}.log")
                with open(log_path, "w") as output:
                    output.write("\n".join(baseline.logs))
            baselines[profile] = baseline
            summaries.append(
                _baseline_summary([m.mode for m in sharing], baseline, log_path)
            )
        baseline = baselines[profile]
        if (
            get_mode(run.mode).isolates_targets
            and not _explicit_targets(run.config)
            and not baseline.geometries
        ):
            if not baseline.requested:
                run.reason = (
                    "it discovers its targets from the control arm, and none ran"
                )
            elif not _baseline_from_runs(
                run.config, baseline.runs, logs=[], geometries=[]
            ).completed:
                # Failed or timed-out control runs launched nothing to learn
                # from; that is not a kernel without mbarrier waits.
                run.reason = (
                    "its control arm produced no sample to discover targets from"
                )
            else:
                run.status = MODE_SKIPPED
                run.reason = (
                    "no instrumented launch has a delay point for this mode "
                    "(for example, the kernels have no mbarrier wait)"
                )
            continue
        run.result = run_stress(
            run.config,
            cutracer_so=cutracer_so,
            runner=runner,
            baseline=baseline,
            deadline=deadline,
        )
        run.status = MODE_RAN
        save_report(run.result, run.report_path)
        if stop_on_first and run.result.triggering is not None:
            stopped = f"stopped after {run.mode} reproduced"
    return PlanResult(
        runs=runs,
        verdict=plan_verdict(runs),
        budget_s=budget_s,
        elapsed_s=time.monotonic() - started,
        baselines=summaries,
    )
