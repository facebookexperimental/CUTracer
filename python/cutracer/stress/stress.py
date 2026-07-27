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
    output_dir: str = "."
    timeout: int = 1800
    cwd: Optional[str] = None
    base_env: Optional[Mapping[str, str]] = None


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


@dataclass
class StressResult:
    reproduced: bool
    completed_trials: int
    reproductions: int
    infra_errors: int
    reproduction_rate: float
    triggering: Optional[TriggeringConfig]
    log_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reproduced": self.reproduced,
            "completed_trials": self.completed_trials,
            "reproductions": self.reproductions,
            "infra_errors": self.infra_errors,
            "reproduction_rate": self.reproduction_rate,
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
            1,
            None,
            _attempt_log(
                delay_ns=delay_ns,
                warpgroup=warpgroup,
                attempt=attempt,
                error=f"timed out: {exc}",
            ),
        )

    log = _attempt_log(
        delay_ns=delay_ns,
        warpgroup=warpgroup,
        attempt=attempt,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )
    if proc.returncode == 0 and os.path.isfile(config_path):
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


def run_stress(
    config: StressConfig,
    *,
    cutracer_so: str,
    runner: Optional[RunTarget] = None,
) -> StressResult:
    """Sweep the delay ladder under random injection until the oracle reproduces.

    A "reproduction" requires both an interesting oracle exit (0) AND a dumped
    delay config (otherwise there is nothing to replay/reduce, so the attempt is
    counted as an infra error rather than a reproduction).
    """
    os.makedirs(config.output_dir, exist_ok=True)
    completed = 0
    reproductions = 0
    infra_errors = 0
    triggering: Optional[TriggeringConfig] = None
    log_parts: List[str] = []
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
    )


def save_report(result: StressResult, path: str) -> None:
    """Atomically write the JSON stress report."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(result.to_dict(), fh, indent=2)
    os.replace(tmp, path)
