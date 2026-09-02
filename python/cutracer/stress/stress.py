# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Core random-delay stress search for ``cutracer stress``.

Runs a correctness oracle repeatedly under random delay injection to discover a
delay configuration that triggers a (data-race) bug, then keeps that config for
deterministic replay / reduction. This is the *discovery* counterpart to
``cutracer reduce`` (which minimizes an already-known triggering config); both
own their oracle-driven delay loops inside CUTracer.

The loop reuses CUTracer's public instrumented-target runner, so a single
attempt is exactly what ``cutracer trace -i random_delay`` would run -- no
self-subprocessing.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from cutracer.reduce.config_mutator import DelayConfigMutator
from cutracer.runner import InstrumentationConfig, run_instrumented_target, RunTarget


@dataclass
class StressConfig:
    """Inputs for a random-delay stress search."""

    oracle_argv: List[str]
    delay_ladder_ns: List[int]
    enable_prob: float = 1.0
    warpgroup_ids: List[int] = field(default_factory=list)
    attempts_per_delay: int = 3
    stop_on_first: bool = True
    # Non-zero oracle exit codes that mean "ran, but did not reproduce". Other
    # non-zero codes are treated as infra errors. Empty => any non-zero code is
    # a clean "did not reproduce" (the standalone default; the service pins [1]).
    not_interesting_exit_codes: List[int] = field(default_factory=list)
    kernel_filters: Optional[str] = None
    # Control runs performed BEFORE the search, with injection instrumented but
    # every point disabled. Without them a reproduction cannot be attributed to
    # the delay on any target that also fails on its own.
    #
    # Defaults to 0 (off) here but to 3 on the `cutracer stress` CLI: the CLI is
    # where a human reads a verdict, while programmatic callers (the service)
    # own their own budget and should opt in explicitly.
    baseline_runs: int = 0
    output_dir: str = "."
    timeout: int = 1800
    cwd: Optional[str] = None
    base_env: Optional[Mapping[str, str]] = None

    def __post_init__(self) -> None:
        # The `cutracer stress` CLI rejects an empty ladder with a UsageError,
        # but programmatic callers (the service, which builds this straight from
        # a spec) reach the search loop directly. An empty ladder means the
        # search would sweep nothing, which is a caller mistake, not a clean
        # zero-trial answer. ValueError, not IndexError-from-somewhere-inside:
        # the service classifies ValueError as an INFRA_ERROR.
        if not self.delay_ladder_ns:
            raise ValueError("delay_ladder_ns must contain at least one delay")
        if self.baseline_runs < 0:
            raise ValueError(f"baseline_runs must be >= 0, got {self.baseline_runs}")


@dataclass
class TriggeringConfig:
    """The delay config that reproduced the bug (pins the winning attempt)."""

    config_path: str
    delay_ns: int
    warpgroup_id: Optional[int]
    enable_prob: float
    attempt_index: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_path": self.config_path,
            "delay_ns": self.delay_ns,
            "warpgroup_id": self.warpgroup_id,
            "enable_prob": self.enable_prob,
            "attempt_index": self.attempt_index,
        }


# How a reproduction relates to the target's own failure rate.
ATTRIBUTION_NOT_REPRODUCED = "not_reproduced"
# No control runs were REQUESTED, so the natural rate is unknown and attribution
# is unproven. This is the caller opting out, not a failure.
ATTRIBUTION_NO_BASELINE = "no_baseline"
# A control arm WAS requested but did not produce the samples asked for (runs
# timed out or died with an unexpected exit code). Distinct from `no_baseline`
# on purpose: "the control crashed" must never read as "the user declined a
# control", and unlike `no_baseline` it corroborates nothing.
ATTRIBUTION_BASELINE_INCOMPLETE = "baseline_incomplete"
# The target reproduced with injection disabled: a stress hit may just be that.
ATTRIBUTION_NATURAL = "natural"
# The control never reproduced, so the injected delay is the distinguishing
# variable. Still only as strong as `baseline_completed` samples.
ATTRIBUTION_ATTRIBUTED = "attributed"


@dataclass
class StressResult:
    reproduced: bool
    completed_trials: int
    reproductions: int
    infra_errors: int
    reproduction_rate: float
    triggering: Optional[TriggeringConfig]
    log_path: Optional[str] = None
    # Attempts whose repro command hung (timed out). Counted separately from
    # ``infra_errors`` so a caller can tell "the workload hangs" apart from a
    # generic launch/tooling failure.
    timed_out: int = 0
    # Attempts where the oracle reported the bug but the delay config it ran
    # under enabled NO injection point. The failure is real, but nothing this
    # search did can have caused it -- see `_enabled_point_count`.
    unattributed_reproductions: int = 0
    # Control arm: runs with injection instrumented but every point disabled.
    #
    # `requested` and `completed` are tracked separately because they answer
    # different questions. `requested` is what the caller asked for;
    # `completed` is how many runs produced a usable sample. Collapsing them
    # would let three control runs that all timed out report as
    # `attribution=no_baseline`, i.e. "no control was asked for".
    baseline_requested: int = 0
    baseline_completed: int = 0
    baseline_reproductions: int = 0
    # Control runs that hung / failed to run, counted apart from each other for
    # the same reason the search arm keeps `timed_out` and `infra_errors` apart.
    baseline_timed_out: int = 0
    baseline_infra_errors: int = 0
    attribution: str = ATTRIBUTION_NO_BASELINE

    @property
    def baseline_rate(self) -> float:
        """Reproductions per *usable* control sample.

        Divides by `baseline_completed`, not `baseline_requested`: a run that
        timed out measured nothing, and counting it as a clean control run
        would understate the target's natural failure rate.
        """
        return (
            (self.baseline_reproductions / self.baseline_completed)
            if self.baseline_completed
            else 0.0
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reproduced": self.reproduced,
            "completed_trials": self.completed_trials,
            "reproductions": self.reproductions,
            "unattributed_reproductions": self.unattributed_reproductions,
            "infra_errors": self.infra_errors,
            "timed_out": self.timed_out,
            "reproduction_rate": self.reproduction_rate,
            "baseline_requested": self.baseline_requested,
            "baseline_completed": self.baseline_completed,
            "baseline_reproductions": self.baseline_reproductions,
            "baseline_timed_out": self.baseline_timed_out,
            "baseline_infra_errors": self.baseline_infra_errors,
            "baseline_rate": self.baseline_rate,
            "attribution": self.attribution,
            "triggering_config": (
                None if self.triggering is None else self.triggering.to_dict()
            ),
        }


@dataclass(frozen=True)
class _AttemptResult:
    completed_trials: int
    reproductions: int
    infra_errors: int
    triggering: Optional[TriggeringConfig]
    log: str
    timed_out: int = 0
    unattributed: int = 0


def _enabled_point_count(config_path: str) -> int:
    """How many injection points the dumped delay config actually turned on.

    A delay config records every candidate point with an ``on`` flag, and only
    the enabled ones stall. With a low ``--enable-prob`` and a large candidate
    set, a whole attempt can enable nothing: on a target that also fails on its
    own, the oracle then reports the bug during a run that injected no delay at
    all, and attributing that to the config would be wrong.

    Reuses the reduce-side config model rather than re-parsing, with validation
    off and every error swallowed: an unreadable config means "cannot attribute",
    which is the same conservative answer as zero points.
    """
    try:
        return len(DelayConfigMutator(config_path, validate=False).enabled_points)
    except Exception:  # noqa: BLE001 - any parse failure means "cannot attribute"
        return 0


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _attempt_log(
    *,
    delay_ns: int,
    warpgroup: Optional[int],
    attempt: int,
    stdout: str = "",
    stderr: str = "",
    error: str = "",
) -> str:
    target = "all" if warpgroup is None else str(warpgroup)
    sections = [
        f"=== delay_ns={delay_ns} warpgroup={target} attempt={attempt} ===",
    ]
    if stdout:
        sections.extend(("--- stdout ---", stdout))
    if stderr:
        sections.extend(("--- stderr ---", stderr))
    if error:
        sections.extend(("--- error ---", error))
    return "\n".join(sections)


def _run_attempt(
    config: StressConfig,
    *,
    delay_ns: int,
    warpgroup: Optional[int],
    attempt: int,
    cutracer_so: Optional[str],
    runner: Optional[RunTarget],
) -> _AttemptResult:
    config_path = os.path.join(
        config.output_dir,
        f"delay-{delay_ns}-wg-{warpgroup}-attempt-{attempt}.json",
    )
    try:
        os.unlink(config_path)
    except FileNotFoundError:
        pass
    try:
        proc = run_instrumented_target(
            config.oracle_argv,
            InstrumentationConfig(
                cutracer_so=cutracer_so,
                instrument="random_delay",
                analysis="random_delay",
                kernel_filters=config.kernel_filters,
                output_dir=config.output_dir,
                delay_ns=delay_ns,
                delay_enable_prob=config.enable_prob,
                delay_mode="random",
                delay_warpgroup_id=warpgroup,
                delay_dump_path=config_path,
                cwd=config.cwd,
                timeout=config.timeout,
                base_env=config.base_env,
            ),
            runner=runner,
        )
    except subprocess.TimeoutExpired as exc:
        _unlink(config_path)
        return _AttemptResult(
            0,
            0,
            0,
            None,
            _attempt_log(
                delay_ns=delay_ns,
                warpgroup=warpgroup,
                attempt=attempt,
                error=f"timed out: {exc}",
            ),
            timed_out=1,
        )

    log = _attempt_log(
        delay_ns=delay_ns,
        warpgroup=warpgroup,
        attempt=attempt,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )
    if proc.returncode == 0 and os.path.isfile(config_path):
        if _enabled_point_count(config_path) == 0:
            # Real failure, but this attempt injected nothing, so it is evidence
            # about the target -- not about any injection point. Keep it out of
            # the reproduction count so it can never become a triggering config.
            _unlink(config_path)
            return _AttemptResult(
                1,
                0,
                0,
                None,
                _attempt_log(
                    delay_ns=delay_ns,
                    warpgroup=warpgroup,
                    attempt=attempt,
                    stdout=proc.stdout or "",
                    stderr=proc.stderr or "",
                    error=(
                        "oracle reported the bug but the delay config enabled 0 "
                        "injection points: the target reproduces without injected "
                        "delay, so this attempt is not attributable"
                    ),
                ),
                unattributed=1,
            )
        return _AttemptResult(
            1,
            1,
            0,
            TriggeringConfig(
                config_path=config_path,
                delay_ns=delay_ns,
                warpgroup_id=warpgroup,
                enable_prob=config.enable_prob,
                attempt_index=attempt,
            ),
            log,
        )
    if proc.returncode == 0:
        # The oracle flagged the run interesting but CUTracer wrote no config,
        # so there is nothing deterministic to replay or reduce.
        return _AttemptResult(0, 0, 1, None, log)

    _unlink(config_path)
    if (
        not config.not_interesting_exit_codes
        or proc.returncode in config.not_interesting_exit_codes
    ):
        return _AttemptResult(1, 0, 0, None, log)
    return _AttemptResult(0, 0, 1, None, log)


@dataclass(frozen=True)
class _BaselineResult:
    """Outcome of the control arm.

    ``requested`` and ``completed`` differ whenever a control run timed out or
    exited with an unexpected code; see ``StressResult`` for why that gap must
    stay visible.
    """

    requested: int
    completed: int
    reproductions: int
    timed_out: int
    infra_errors: int
    logs: List[str]


def _run_baseline(
    config: StressConfig,
    *,
    cutracer_so: Optional[str],
    runner: Optional[RunTarget],
) -> _BaselineResult:
    """Run the control arm: instrumented, but with every delay point disabled.

    A *matched* control on purpose. Running the oracle bare would also measure a
    natural rate, but it would differ from the search arm in two variables at
    once (instrumentation overhead and injected stalls), so it could not tell
    them apart. `--delay-enable-prob 0` keeps the instrumentation and removes
    only the stalls, which is the variable the search manipulates.

    No dump path is passed: an all-disabled config is not worth keeping, and not
    writing one keeps the baseline out of the search's config namespace.
    """
    completed = 0
    reproduced = 0
    timed_out = 0
    infra_errors = 0
    logs: List[str] = []
    if config.baseline_runs <= 0:
        # A caller that opted out of the control arm must not pay for anything
        # here -- including reading `delay_ladder_ns[0]`.
        return _BaselineResult(0, 0, 0, 0, 0, logs)
    delay_ns = config.delay_ladder_ns[0]
    for attempt in range(config.baseline_runs):
        try:
            proc = run_instrumented_target(
                config.oracle_argv,
                InstrumentationConfig(
                    cutracer_so=cutracer_so,
                    instrument="random_delay",
                    analysis="random_delay",
                    kernel_filters=config.kernel_filters,
                    output_dir=config.output_dir,
                    delay_ns=delay_ns,
                    delay_enable_prob=0.0,
                    delay_mode="random",
                    cwd=config.cwd,
                    timeout=config.timeout,
                    base_env=config.base_env,
                ),
                runner=runner,
            )
        except subprocess.TimeoutExpired as exc:
            timed_out += 1
            logs.append(
                _attempt_log(
                    delay_ns=delay_ns,
                    warpgroup=None,
                    attempt=attempt,
                    error=f"baseline timed out: {exc}",
                )
            )
            continue
        logs.append(
            _attempt_log(
                delay_ns=delay_ns,
                warpgroup=None,
                attempt=attempt,
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                error="baseline (injection disabled)",
            )
        )
        if proc.returncode == 0:
            completed += 1
            reproduced += 1
        elif (
            not config.not_interesting_exit_codes
            or proc.returncode in config.not_interesting_exit_codes
        ):
            completed += 1
        else:
            # Same rule as the search arm: an exit code the oracle contract does
            # not cover means the control run did not measure anything.
            infra_errors += 1
    return _BaselineResult(
        requested=config.baseline_runs,
        completed=completed,
        reproductions=reproduced,
        timed_out=timed_out,
        infra_errors=infra_errors,
        logs=logs,
    )


def _attribution(
    *,
    baseline_requested: int,
    baseline_completed: int,
    baseline_reproductions: int,
    triggering: object,
) -> str:
    """Classify what a reproduction can be credited to."""
    if baseline_requested == 0:
        return ATTRIBUTION_NO_BASELINE
    if baseline_reproductions > 0:
        # The target fails with injection disabled, so a search hit is not
        # distinguishable from that. Reported even when the search found a
        # config, because the config may merely have coincided with a failure.
        # Checked before the completeness gate on purpose: a control run that
        # DID reproduce is positive evidence, and stays the stronger answer even
        # if its siblings never finished.
        return ATTRIBUTION_NATURAL
    if baseline_completed < baseline_requested:
        # A control arm was asked for and did not deliver. "0 of 3 control runs
        # produced a sample" is not the same claim as "no control was wanted",
        # and it must not be reported as one.
        return ATTRIBUTION_BASELINE_INCOMPLETE
    if triggering is None:
        return ATTRIBUTION_NOT_REPRODUCED
    return ATTRIBUTION_ATTRIBUTED


def run_stress(
    config: StressConfig,
    *,
    cutracer_so: Optional[str] = None,
    runner: Optional[RunTarget] = None,
) -> StressResult:
    """Sweep the delay ladder under random injection until the oracle reproduces.

    A "reproduction" requires an interesting oracle exit (0), a dumped delay
    config (otherwise there is nothing to replay/reduce, so the attempt is
    counted as an infra error), AND at least one enabled injection point in that
    config (otherwise the attempt injected nothing and cannot have caused the
    failure -- counted as an unattributed reproduction).
    """
    os.makedirs(config.output_dir, exist_ok=True)
    completed = 0
    reproductions = 0
    infra_errors = 0
    timed_out = 0
    unattributed = 0
    baseline = _run_baseline(config, cutracer_so=cutracer_so, runner=runner)
    triggering: Optional[TriggeringConfig] = None
    log_parts: List[str] = list(baseline.logs)
    stop = False
    warp_targets: List[Optional[int]] = (
        [int(x) for x in config.warpgroup_ids] if config.warpgroup_ids else [None]
    )

    for delay_ns in config.delay_ladder_ns:
        for warpgroup in warp_targets:
            for attempt in range(config.attempts_per_delay):
                outcome = _run_attempt(
                    config,
                    delay_ns=delay_ns,
                    warpgroup=warpgroup,
                    attempt=attempt,
                    cutracer_so=cutracer_so,
                    runner=runner,
                )
                completed += outcome.completed_trials
                reproductions += outcome.reproductions
                infra_errors += outcome.infra_errors
                timed_out += outcome.timed_out
                unattributed += outcome.unattributed
                log_parts.append(outcome.log)
                if outcome.triggering is not None:
                    if triggering is None:
                        triggering = outcome.triggering
                    if config.stop_on_first:
                        stop = True
                if stop:
                    break
            if stop:
                break
        if stop:
            break

    log_path: Optional[str] = None
    if log_parts:
        log_path = os.path.join(config.output_dir, "stress.log")
        try:
            with open(log_path, "w") as fh:
                fh.write("\n".join(log_parts))
        except OSError:
            log_path = None

    rate = (reproductions / completed) if completed else 0.0
    return StressResult(
        reproduced=triggering is not None,
        completed_trials=completed,
        reproductions=reproductions,
        infra_errors=infra_errors,
        reproduction_rate=rate,
        triggering=triggering,
        log_path=log_path,
        timed_out=timed_out,
        unattributed_reproductions=unattributed,
        baseline_requested=baseline.requested,
        baseline_completed=baseline.completed,
        baseline_reproductions=baseline.reproductions,
        baseline_timed_out=baseline.timed_out,
        baseline_infra_errors=baseline.infra_errors,
        attribution=_attribution(
            baseline_requested=baseline.requested,
            baseline_completed=baseline.completed,
            baseline_reproductions=baseline.reproductions,
            triggering=triggering,
        ),
    )


def save_report(result: StressResult, path: str) -> None:
    """Atomically write the JSON stress report."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(result.to_dict(), fh, indent=2)
    os.replace(tmp, path)
