# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Run several ``cutracer stress`` modes as one search with one verdict.

Each mode keeps its own delay placement and defaults (``cutracer.stress.modes``)
and writes its own report under ``<output_dir>/<mode>``. Modes that instrument
the same points the same way share one control arm, which runs once. The modes
then run in order under one optional time budget, and by default the search
stops at the first mode that reproduces. A mode that does not apply to the
program, as its control arm (or, for a mode that needs PDL launches, any
control arm that ran before it) shows, is skipped.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Collection, Dict, List, Optional, Sequence

from cutracer.runner import RunTarget
from cutracer.stress.modes import get_mode, ISOLATE_BY_BARRIER, ISOLATE_BY_WARP
from cutracer.stress.stress import (
    _barrier_targets,
    _baseline_from_runs,
    _BaselineResult,
    _delay_patterns,
    _isolate_by,
    _pdl_hold_reason,
    _pdl_skip_reason,
    _PdlEvidence,
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
# kernel without mbarrier waits, or pdl on a program without PDL launches), or
# it is a coverage gap (see `ModeRun.coverage_gap`). It does not make the plan
# incomplete.
MODE_SKIPPED = "skipped"
# The mode applied but did not run: an earlier mode reproduced, the time budget
# ran out, its control arm produced nothing to discover targets from, or the
# plan cannot test a mode the user named (see `run_stress_plan`).
MODE_NOT_RUN = "not_run"


@dataclass
class ModeRun:
    """One mode of a plan, its search config, and what became of it."""

    config: StressConfig
    status: str = MODE_NOT_RUN
    reason: Optional[str] = None
    result: Optional[StressResult] = None
    # Skipped, although the mode may apply, because the plan cannot test it
    # and the user did not name the mode (see `run_stress_plan`). A clean
    # verdict names it.
    coverage_gap: bool = False

    @property
    def mode(self) -> str:
        return self.config.strategy

    @property
    def report_path(self) -> str:
        return os.path.join(self.config.output_dir, "stress_report.json")

    def to_dict(self) -> Dict[str, Any]:
        entry: Dict[str, Any] = {
            "mode": self.mode,
            "status": self.status,
            "reason": self.reason,
            "output_dir": self.config.output_dir,
            "report": None if self.result is None else self.result.to_dict(),
        }
        if self.coverage_gap:
            entry["coverage_gap"] = True
        return entry


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
        get_mode(config.strategy).nonblocking_launches,
        config.kernel_filters,
        config.exact_kernel_name,
        config.exact_kernel_checksum,
        config.kernel_timeout_s,
        config.exit_quarantine_ns,
        config.timeout,
        config.baseline_runs,
    )


def _launch_scope(config: StressConfig) -> tuple[Any, ...]:
    """What makes two modes' control runs launch the same kernels and
    instrument the same ones, so that one's launch events describe the
    other's (see `_pdl_check`)."""
    env = None if config.base_env is None else tuple(sorted(config.base_env.items()))
    return (
        tuple(config.oracle_argv),
        config.cwd,
        env,
        config.kernel_filters,
        config.exact_kernel_name,
        config.exact_kernel_checksum,
    )


_PDL_UNATTRIBUTED = (
    "the control runs recorded no launch attributes (a CUTracer that does not "
    "record them), so it is unknown whether any launch uses programmatic "
    "dependent launch (PDL); `--strategy pdl` alone runs it anyway"
)
_PDL_UNKNOWN = (
    "no control run completed with a launch, so it is unknown whether any "
    "launch uses programmatic dependent launch (PDL); `--strategy pdl` alone "
    "runs it anyway"
)


def _pdl_check(
    run: ModeRun,
    evidence: Optional[_PdlEvidence],
    *,
    baseline: Optional[_BaselineResult],
) -> Optional[tuple[str, str]]:
    """Why `run`'s mode, if it needs PDL launches, should not search, as
    (status, reason), from the launch events of the control arms so far that
    launched the same kernels; None otherwise, or to search. MODE_SKIPPED
    means the mode does not apply, MODE_NOT_RUN that it may apply but this
    plan cannot test it (see `_record_pdl_check`).

    Without evidence a plan does not search blind: before the mode's own
    control arm (`baseline` None) it waits for that arm, after it the mode
    cannot be tested.
    """
    if not get_mode(run.mode).needs_pdl:
        return None
    if baseline is not None and not baseline.requested:
        return (
            MODE_NOT_RUN,
            "it decides from the control arm whether any launch uses "
            "programmatic dependent launch (PDL), and none ran",
        )
    if evidence is not None:
        skipped = _pdl_skip_reason(evidence)
        if skipped is not None:
            return MODE_SKIPPED, skipped
        held = _pdl_hold_reason(evidence)
        if held is not None:
            return MODE_NOT_RUN, held
        if evidence.attributed:
            return None
        if evidence.launches:
            return MODE_NOT_RUN, _PDL_UNATTRIBUTED
    return None if baseline is None else (MODE_NOT_RUN, _PDL_UNKNOWN)


def _record_pdl_check(
    run: ModeRun, check: tuple[str, str], named: Collection[str]
) -> None:
    """Record a `_pdl_check` outcome on `run`. A mode the plan cannot test is
    not run, which makes the plan incomplete, when the user named it;
    otherwise it is skipped as a coverage gap, which a clean verdict names."""
    run.status, run.reason = check
    if run.status == MODE_NOT_RUN and run.mode not in named:
        run.status, run.coverage_gap = MODE_SKIPPED, True


def _explicit_targets(config: StressConfig) -> bool:
    return (
        _isolate_by(config) == ISOLATE_BY_WARP
        and config.isolate_ranks is not None
        and config.isolate_warps is not None
    )


def _not_searched(run: ModeRun, baseline: _BaselineResult) -> Optional[tuple[str, str]]:
    """Why `run`'s mode should not search, as (status, reason), from what its
    control arm learned about the instrumented launches; None to search."""
    needs_targets = _isolate_by(run.config) is not None
    if needs_targets and _explicit_targets(run.config):
        return None
    if not baseline.requested:
        if needs_targets:
            return (
                MODE_NOT_RUN,
                "it discovers its targets from the control arm, and none ran",
            )
        return None
    if baseline.geometries:
        if _isolate_by(run.config) == ISOLATE_BY_BARRIER and not _barrier_targets(
            run.config, baseline.barriers
        ):
            return (
                MODE_SKIPPED,
                "no instrumented launch has a named barrier to delay at this "
                "mode's delay points (BAR.SYNC or BAR.RED, or BAR.ARV with "
                "barrier_arrives)",
            )
        return None
    if baseline.instrumented_kernels:
        return (
            MODE_SKIPPED,
            "no instrumented launch has a delay point for this mode: its delay "
            "patterns match no instruction of the launched kernels (for example, "
            "isolate on kernels without mbarrier waits, bar on kernels without "
            "named barriers, or lds on kernels without shared-memory loads)",
        )
    completed = _baseline_from_runs(
        run.config, baseline.runs, logs=[], geometries=[]
    ).completed
    if not completed:
        # Failed or timed-out control runs launched nothing to learn from;
        # that is not a kernel without delay points.
        return (
            MODE_NOT_RUN,
            "its control arm produced no sample to learn the instrumented "
            "launches from",
        )
    return (
        MODE_NOT_RUN,
        "its control arm recorded no launch of an instrumented kernel: the kernel "
        "filter matched nothing, or the delay dump or the launch events were "
        "missing or unreadable (see the log)",
    )


def plan_verdict(runs: Sequence[ModeRun]) -> Verdict:
    """The plan's verdict: suspect if any mode saw the failure without an
    attributable delay, since a finding in another mode does not outweigh a
    failure that also happens without one; else a finding in any mode; else
    clean only if every mode that applied ran clean or is a coverage gap,
    which the clean verdict names."""
    verdicts = [
        (run.mode, result_verdict(run.result)) for run in runs if run.result is not None
    ]

    def reasons_of(status: str) -> list[str]:
        return [
            f"{mode}: {reason}"
            for mode, verdict in verdicts
            if verdict.status == status
            for reason in verdict.reasons
        ]

    suspect, finding = reasons_of(VERDICT_SUSPECT), reasons_of(VERDICT_FINDING)
    if suspect:
        return Verdict(VERDICT_SUSPECT, suspect + finding)
    if finding:
        return Verdict(VERDICT_FINDING, finding)
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
    if reasons:
        return Verdict(VERDICT_INCOMPLETE, reasons)
    # A clean mode can still qualify what it covered (an isolate window), and
    # a coverage gap is a mode the plan did not test.
    return Verdict(
        VERDICT_CLEAN,
        [
            f"{mode}: {reason}"
            for mode, verdict in verdicts
            for reason in verdict.reasons
        ]
        + [f"{run.mode}: skipped: {run.reason}" for run in runs if run.coverage_gap],
    )


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
        "barrier_sites": len(baseline.barriers),
        # Launch events of the control runs that completed.
        "launches": baseline.pdl.launches,
        "attributed_launches": baseline.pdl.attributed,
        "pdl_launches": baseline.pdl.direct,
        "testable_pdl_launches": baseline.pdl.testable,
        "covered_pdl_launches": baseline.pdl.covered,
        "graph_pdl_launches": baseline.pdl.graph,
    }


def run_stress_plan(  # noqa: C901
    configs: Sequence[StressConfig],
    *,
    output_dir: str,
    budget_s: Optional[float] = None,
    stop_on_first: bool = True,
    cutracer_so: Optional[str] = None,
    runner: Optional[RunTarget] = None,
    named_modes: Optional[Collection[str]] = None,
) -> PlanResult:
    """Run one search per config (one per mode), in order.

    Each config names its mode in `strategy` and its own `output_dir`. A
    control arm runs once per instrumentation profile (see `_profile`), the
    first time a mode needs it, and every mode of that profile counts it under
    its own oracle contract. A mode that needs PDL launches is held back
    without running its control arm when an earlier control arm that launched
    the same kernels shows it does not apply (skipped) or that the plan cannot
    test it, and without that evidence after its own control arm the plan
    cannot test it either (see `_pdl_check`). A mode skipped because it did
    not apply counts as one the plan cannot test if a later control arm then
    records a PDL launch that can overlap. A mode the plan cannot test is not
    run if `named_modes` (the modes the user listed; None names every mode of
    `configs`) has it, so the plan is incomplete, and otherwise skipped as a
    coverage gap. With `stop_on_first`, the modes after the first one that
    reproduces do not run. No run starts with less than `MIN_RUN_BUDGET_S` of
    `budget_s` left.
    """
    if not configs:
        raise ValueError("a stress plan needs at least one mode")
    os.makedirs(output_dir, exist_ok=True)
    started = time.monotonic()
    deadline = None if budget_s is None else started + budget_s
    runs = [ModeRun(config) for config in configs]
    named = [run.mode for run in runs] if named_modes is None else named_modes
    members: Dict[tuple[Any, ...], List[ModeRun]] = {}
    for run in runs:
        members.setdefault(_profile(run.config), []).append(run)
    baselines: Dict[tuple[Any, ...], _BaselineResult] = {}
    # The PDL launches the control arms of each launch scope recorded so far.
    evidence: Dict[tuple[Any, ...], _PdlEvidence] = {}
    # Modes skipped on an earlier control arm's evidence (see `_pdl_check`).
    preskipped: List[ModeRun] = []
    summaries: List[Dict[str, Any]] = []
    stopped: Optional[str] = None
    for run in runs:
        if stopped is not None:
            run.reason = stopped
            continue
        profile = _profile(run.config)
        scope = _launch_scope(run.config)
        if profile not in baselines:
            early = _pdl_check(run, evidence.get(scope), baseline=None)
            if early is not None:
                _record_pdl_check(run, early, named)
                if run.status == MODE_SKIPPED and not run.coverage_gap:
                    preskipped.append(run)
                continue
        remaining = _remaining_s(deadline)
        if remaining is not None and remaining < MIN_RUN_BUDGET_S:
            run.reason = "the time budget ran out before this mode"
            continue
        if profile not in baselines:
            sharing = members[profile]
            os.makedirs(run.config.output_dir, exist_ok=True)
            baseline = _run_baseline(
                run.config,
                cutracer_so=cutracer_so,
                runner=runner,
                # Every control arm records the instrumented launches and their
                # delay points: isolate takes its targets from them, and a mode
                # whose patterns match nothing in the launched kernels does not
                # apply.
                capture_geometry=True,
                deadline=deadline,
            )
            log_path = None
            if baseline.logs:
                log_path = os.path.join(output_dir, f"baseline-{len(summaries)}.log")
                with open(log_path, "w") as output:
                    output.write("\n".join(baseline.logs))
            baselines[profile] = baseline
            evidence[scope] = baseline.pdl.merge(evidence.get(scope, _PdlEvidence()))
            summaries.append(
                _baseline_summary([m.mode for m in sharing], baseline, log_path)
            )
        baseline = baselines[profile]
        held = _pdl_check(run, evidence.get(scope), baseline=baseline)
        if held is not None:
            _record_pdl_check(run, held, named)
            continue
        not_searched = _not_searched(run, baseline)
        if not_searched is not None:
            run.status, run.reason = not_searched
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
    for run in preskipped:
        final = evidence.get(_launch_scope(run.config))
        if final is not None and _pdl_skip_reason(final) is None:
            later = (
                "a later control arm recorded a launch with programmatic "
                "dependent launch (PDL) that can overlap the kernel before it, "
                "which the earlier ones did not"
            )
            _record_pdl_check(run, (MODE_NOT_RUN, later), named)
    return PlanResult(
        runs=runs,
        verdict=plan_verdict(runs),
        budget_s=budget_s,
        elapsed_s=time.monotonic() - started,
        baselines=summaries,
    )
