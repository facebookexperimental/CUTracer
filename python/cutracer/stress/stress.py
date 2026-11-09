# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Core random-delay stress search for ``cutracer stress``.

Runs a correctness oracle repeatedly under delay injection to discover a delay
configuration that triggers a (data-race) bug, then keeps that config for
deterministic replay / reduction. This is the *discovery* counterpart to
``cutracer reduce`` (which minimizes an already-known triggering config); both
own their oracle-driven delay loops inside CUTracer.

Two strategies: ``random`` injects random per-thread delays at every candidate
point (optionally one warpgroup at a time); ``isolate`` stalls one (cluster
rank, CTA-local warp) at a time with a fixed delay while every other warp runs
at full speed (see ``StressConfig.strategy``).

The loop reuses CUTracer's public instrumented-target runner, so a single
attempt is exactly what ``cutracer trace -i random_delay`` would run -- no
self-subprocessing.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import secrets
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import click
from cutracer.runner import (
    InstrumentationConfig,
    run_instrumented_target,
    run_process_group,
    RunTarget,
)
from cutracer.stress.modes import get_mode, get_modes, ISOLATE, RANDOM
from cutracer.stress.retention import (
    MAX_REJECTED_CONFIGS as _MAX_REJECTED_CONFIGS,
    retain_oldest_and_newest,
)
from cutracer.watchdog import main_logs, watchdog_terminations


logger: logging.Logger = logging.getLogger(__name__)
MAX_REJECTED_CONFIGS = _MAX_REJECTED_CONFIGS
# CUTracer's default no-data watchdog, kept when no kernel timeout is set.
_DEFAULT_NO_DATA_TIMEOUT_S = 15
# A run is not started with less time than this left before a deadline.
MIN_RUN_BUDGET_S = 30

# What the oracle observed in a reproduced attempt.
SYMPTOM_FAILURE = "failure"
# CUTracer's watchdog ended a hung kernel (requires `hang_is_bug`).
SYMPTOM_HANG = "hang"

STRATEGY_RANDOM: str = RANDOM.name
STRATEGY_ISOLATE: str = ISOLATE.name
ISOLATE_DEFAULT_PATTERNS: str | None = ISOLATE.delay_patterns
# CUTracer warp masks are 32 bits wide (CUDA allows 32 warps per CTA).
_MAX_WARPS = 32


@dataclass(frozen=True)
class DelayTarget:
    """One actor of an isolate search: a CTA-local warp on one cluster rank.

    ``cluster_rank`` is None for launches without clusters, where the warp is
    delayed in every CTA.
    """

    warp: int
    cluster_rank: Optional[int] = None

    @property
    def warp_mask(self) -> str:
        return hex(1 << self.warp)

    def label(self) -> str:
        if self.cluster_rank is None:
            return f"warp {self.warp}"
        return f"cluster rank {self.cluster_rank}, warp {self.warp}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_rank": self.cluster_rank,
            "warp": self.warp,
            "warp_mask": self.warp_mask,
        }


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
    # CUTracer's per-kernel timeout for every run (0 = off). random_delay sends
    # no trace data, so otherwise CUTracer's no-data watchdog ends any kernel
    # that runs for 15 s; a positive value turns that watchdog off and bounds
    # each kernel by this timeout instead.
    kernel_timeout_s: int = 0
    # Count a run whose kernel CUTracer's watchdog ended (a hang) as a
    # reproduction, whatever the oracle's exit code. Off by default: a large
    # delay can also slow a correct kernel past the timeout.
    hang_is_bug: bool = False
    # A mode of `cutracer.stress.modes`. STRATEGY_RANDOM: random per-thread
    # delays at every candidate point, all warps at once or one warpgroup at a
    # time (`warpgroup_ids`). STRATEGY_ISOLATE: one target (cluster rank,
    # CTA-local warp) per attempt, with a fixed delay before each matching
    # instruction (`enable_prob` is ignored) while every other warp runs at
    # full speed. Random delays slow a producer down together with its
    # consumers, so they cannot show that a consumer is missing from the
    # release protocol; isolating that consumer lets the producer lap it.
    strategy: str = STRATEGY_RANDOM
    # Isolate targets: cluster ranks (tried high to low) and CTA-local warps.
    # None = every rank / warp of the instrumented launches the control arm
    # observed, which needs `baseline_runs >= 1`.
    isolate_ranks: Optional[List[int]] = None
    isolate_warps: Optional[List[int]] = None
    # SASS substrings to delay at (CUTRACER_DELAY_PATTERNS). None = mbarrier
    # waits for isolate, CUTracer's built-in patterns for random.
    delay_patterns: Optional[str] = None
    cwd: Optional[str] = None
    base_env: Optional[Mapping[str, str]] = None
    resume_result: Optional["StressResult"] = None
    progress_callback: Optional[Callable[["StressResult"], None]] = None
    use_bundled_cuda_tools: bool = True
    expected_kernel_name: Optional[str] = None
    expected_kernel_checksum: Optional[str] = None
    exact_kernel_name: Optional[str] = None
    exact_kernel_checksum: Optional[str] = None

    def __post_init__(self) -> None:
        # The `cutracer stress` CLI rejects an empty ladder with a UsageError,
        # but programmatic callers (the service, which builds this straight from
        # a spec) reach the search loop directly. An empty ladder means the
        # search would sweep nothing, which is a caller mistake, not a clean
        # zero-trial answer. ValueError, not IndexError-from-somewhere-inside:
        # the service classifies ValueError as an INFRA_ERROR.
        if not self.delay_ladder_ns:
            raise ValueError("delay_ladder_ns must contain at least one delay")
        if any(delay_ns < 1 for delay_ns in self.delay_ladder_ns):
            raise ValueError("delay_ladder_ns entries must be positive")
        if self.attempts_per_delay < 1:
            raise ValueError(
                f"attempts_per_delay must be positive, got {self.attempts_per_delay}"
            )
        if not 0.0 < self.enable_prob <= 1.0:
            raise ValueError(f"enable_prob must be in (0, 1], got {self.enable_prob}")
        if any(not 0 <= group < 8 for group in self.warpgroup_ids):
            raise ValueError("warpgroup_ids must be in [0, 8)")
        if self.baseline_runs < 0:
            raise ValueError(f"baseline_runs must be >= 0, got {self.baseline_runs}")
        if self.kernel_timeout_s < 0:
            raise ValueError(
                f"kernel_timeout_s must be >= 0, got {self.kernel_timeout_s}"
            )
        if self.strategy not in get_modes():
            raise ValueError(f"unknown stress strategy {self.strategy!r}")
        if get_mode(self.strategy).isolates_targets:
            if self.warpgroup_ids:
                raise ValueError("warpgroup_ids do not apply to an isolate search")
        elif self.isolate_ranks is not None or self.isolate_warps is not None:
            raise ValueError("isolate_ranks and isolate_warps need strategy isolate")
        if self.isolate_ranks is not None and not self.isolate_ranks:
            raise ValueError("isolate_ranks must name at least one cluster rank")
        if self.isolate_warps is not None and not self.isolate_warps:
            raise ValueError("isolate_warps must name at least one warp")
        if any(rank < 0 for rank in self.isolate_ranks or []):
            raise ValueError("isolate_ranks must be >= 0")
        if any(not 0 <= warp < _MAX_WARPS for warp in self.isolate_warps or []):
            raise ValueError(f"isolate_warps must be in [0, {_MAX_WARPS})")


@dataclass
class TriggeringConfig:
    """The delay config that reproduced the bug (pins the winning attempt)."""

    config_path: str
    delay_ns: int
    warpgroup_id: Optional[int]
    enable_prob: float
    attempt_index: int
    enabled_point_count: Optional[int] = None
    symptom: str = SYMPTOM_FAILURE
    # The isolate target (`DelayTarget.to_dict()`); None for random searches.
    target: Optional[Dict[str, Any]] = None
    # `cutracer trace` options that replay this config with the targeting it
    # was recorded under; add `--delay-load-path <config> -- <oracle>`.
    replay_args: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_path": self.config_path,
            "delay_ns": self.delay_ns,
            "warpgroup_id": self.warpgroup_id,
            "enable_prob": self.enable_prob,
            "attempt_index": self.attempt_index,
            "enabled_point_count": self.enabled_point_count,
            "symptom": self.symptom,
            "target": self.target,
            "replay_args": list(self.replay_args),
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
    # Attempts in which CUTracer's watchdog ended a hung kernel. They are also
    # counted by outcome: as reproductions with `hang_is_bug`, otherwise by the
    # oracle's exit code (or as `timed_out`).
    hangs: int = 0
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
    baseline_hangs: int = 0
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

    # Exact durable prefix represented by ``log_path``. ``None`` supports old
    # in-memory callers that did not record a boundary; zero is an explicit
    # empty durable prefix. This lets resume discard an uncheckpointed tail.
    log_size_bytes: Optional[int] = None
    # Delay configs of attempts rejected by validation, newest last. Retained
    # so a rejection can be explained from published artifacts alone.
    rejected_config_paths: List[str] = field(default_factory=list)
    strategy: str = STRATEGY_RANDOM
    # The isolate targets in search order (`DelayTarget.to_dict()`).
    targets: List[Dict[str, Any]] = field(default_factory=list)
    # Schedule shape the counts were accumulated against, recorded by
    # `run_stress`. A resume locates the remaining trials by counting the
    # ones already run, so it must reuse this shape; None for results
    # reconstructed from older reports, which skip the check.
    delay_ladder_ns: Optional[List[int]] = None
    attempts_per_delay: Optional[int] = None
    # Attempts the schedule held, and whether a deadline stopped the search
    # before it ran them all (see `result_verdict`).
    planned_trials: Optional[int] = None
    budget_exhausted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "verdict": result_verdict(self).to_dict(),
            "planned_trials": self.planned_trials,
            "budget_exhausted": self.budget_exhausted,
            "targets": list(self.targets),
            "delay_ladder_ns": (
                None if self.delay_ladder_ns is None else list(self.delay_ladder_ns)
            ),
            "attempts_per_delay": self.attempts_per_delay,
            "reproduced": self.reproduced,
            "completed_trials": self.completed_trials,
            "reproductions": self.reproductions,
            "unattributed_reproductions": self.unattributed_reproductions,
            "infra_errors": self.infra_errors,
            "timed_out": self.timed_out,
            "hangs": self.hangs,
            "log_size_bytes": self.log_size_bytes,
            "reproduction_rate": self.reproduction_rate,
            "baseline_requested": self.baseline_requested,
            "baseline_completed": self.baseline_completed,
            "baseline_reproductions": self.baseline_reproductions,
            "baseline_timed_out": self.baseline_timed_out,
            "baseline_infra_errors": self.baseline_infra_errors,
            "baseline_hangs": self.baseline_hangs,
            "baseline_rate": self.baseline_rate,
            "attribution": self.attribution,
            "triggering_config": (
                None if self.triggering is None else self.triggering.to_dict()
            ),
            "rejected_config_paths": list(self.rejected_config_paths),
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
    hangs: int = 0
    # Delay config of an attempt that was rejected by validation. Kept so the
    # rejection can be diagnosed after the fact; a remote worker cannot be
    # attached to, and deleting the evidence at the moment validation fails
    # makes the failure unexplainable from the published artifacts.
    rejected_config_path: Optional[str] = None


def _enabled_point_count(config_path: str) -> Optional[int]:
    """How many injection points the dumped delay config actually turned on.

    A delay config records every candidate point with an ``on`` flag, and only
    the enabled ones stall. With a low ``--enable-prob`` and a large candidate
    set, a whole attempt can enable nothing: on a target that also fails on its
    own, the oracle then reports the bug during a run that injected no delay at
    all, and attributing that to the config would be wrong.

    ``None`` means the config could not be inspected. That is infrastructure
    failure rather than proof that no delay was injected.
    """
    try:
        with open(config_path) as source:
            payload = json.load(source)
        kernels = payload.get("kernels", {})
        return sum(
            raw_point.get("on") is True
            for raw_kernel in kernels.values()
            if isinstance(raw_kernel, dict)
            for raw_point in raw_kernel.get("instrumentation_points", {}).values()
            if isinstance(raw_point, dict)
        )
    except (AttributeError, OSError, UnicodeError, json.JSONDecodeError):
        return None


def _preserve_rejected_config(config_path: str) -> Optional[str]:
    """Move a rejected delay config aside under a unique diagnostic name.

    Returns the retained path, or ``None`` when there was nothing to keep.
    Renaming (instead of leaving it in place) keeps the per-attempt config
    namespace free for the next attempt while preserving the evidence.
    """
    if not os.path.isfile(config_path):
        return None
    directory, name = os.path.split(config_path)
    stem, suffix = os.path.splitext(name)
    descriptor = -1
    rejected: Optional[str] = None
    try:
        descriptor, rejected = tempfile.mkstemp(
            prefix=f"rejected-{stem}-",
            suffix=suffix,
            dir=directory,
        )
        os.close(descriptor)
        descriptor = -1
        os.replace(config_path, rejected)
    except OSError as exc:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if rejected is not None:
            _unlink(rejected)
        logger.warning(
            "failed to preserve rejected delay config %s: %s",
            config_path,
            exc,
        )
        return None
    return rejected


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _is_owned_path(path: str, owned_root: str) -> bool:
    root = os.path.abspath(owned_root)
    absolute = os.path.abspath(path)
    try:
        return os.path.commonpath((root, absolute)) == root
    except ValueError:
        return False


def _bounded_rejected_config_paths(
    paths: List[str],
    *,
    owned_root: Optional[str] = None,
) -> List[str]:
    unique = list(dict.fromkeys(paths))
    retained = retain_oldest_and_newest(unique)
    if owned_root is not None:
        retained_set = set(retained)
        for path in unique:
            if path not in retained_set and _is_owned_path(path, owned_root):
                _unlink(os.path.abspath(path))
    return retained


def _record_rejected_config(
    paths: List[str],
    rejected: Optional[str],
    *,
    owned_root: str,
) -> None:
    if rejected is None or rejected in paths:
        return
    candidates = [*paths, rejected]
    retained = retain_oldest_and_newest(candidates)
    retained_set = set(retained)
    for path in candidates:
        if path not in retained_set and _is_owned_path(path, owned_root):
            _unlink(path)
    paths[:] = retained


def _attempt_log(
    *,
    delay_ns: int,
    warpgroup: Optional[int],
    attempt: int,
    returncode: Optional[int] = None,
    stdout: str = "",
    stderr: str = "",
    error: str = "",
    watchdog: Sequence[str] = (),
    target: Optional[DelayTarget] = None,
) -> str:
    if target is not None:
        placement = f"target={target.label()}"
    else:
        placement = f"warpgroup={'all' if warpgroup is None else warpgroup}"
    sections = [
        f"=== delay_ns={delay_ns} {placement} attempt={attempt} ===",
    ]
    if returncode is not None:
        sections.append(f"returncode={returncode}")
    if stdout:
        sections.extend(("--- stdout ---", stdout))
    if stderr:
        sections.extend(("--- stderr ---", stderr))
    if watchdog:
        sections.extend(("--- CUTracer watchdog ---", *watchdog))
    if error:
        sections.extend(("--- error ---", error))
    return "\n".join(sections)


def _no_data_timeout_s(config: StressConfig) -> int:
    """The no-data watchdog setting: off whenever the kernel timeout bounds kernels."""
    return 0 if config.kernel_timeout_s > 0 else _DEFAULT_NO_DATA_TIMEOUT_S


def _delay_patterns(config: StressConfig) -> Optional[str]:
    return config.delay_patterns or get_mode(config.strategy).delay_patterns


def _replay_args(
    config: StressConfig,
    *,
    delay_ns: int,
    warpgroup: Optional[int],
    target: Optional[DelayTarget],
) -> List[str]:
    """`cutracer trace` options that replay an attempt's config as it ran.

    The delay config records which points were on, not how the delay was
    targeted, so a replay must pass the targeting again.
    """
    args = ["-i", "random_delay", "-a", "random_delay", "--delay-ns", str(delay_ns)]
    args += _targeting_args(config, warpgroup=warpgroup, target=target)
    if config.kernel_timeout_s > 0:
        args += ["--no-data-timeout-s", "0"]
    return args


def _targeting_args(
    config: StressConfig,
    *,
    warpgroup: Optional[int],
    target: Optional[DelayTarget],
) -> List[str]:
    """Options that `cutracer trace` and `cutracer reduce` share to restore an
    attempt's targeting and kernel timeout."""
    args: List[str] = []
    if config.kernel_filters:
        args += ["-k", config.kernel_filters]
    if target is not None:
        if target.cluster_rank is None:
            args += ["--delay-mode", "fixed"]
        else:
            args += ["--delay-mode", "cluster_fixed"]
            args += ["--delay-cluster-cta-id", str(target.cluster_rank)]
        args += ["--delay-warp-mask", target.warp_mask]
    elif warpgroup is not None:
        args += ["--delay-warpgroup-id", str(warpgroup)]
    patterns = _delay_patterns(config)
    if patterns:
        args += ["--delay-patterns", patterns]
    if config.kernel_timeout_s > 0:
        args += ["--kernel-timeout-s", str(config.kernel_timeout_s)]
    return args


def reduce_args(config: StressConfig, triggering: TriggeringConfig) -> List[str]:
    """`cutracer reduce` options that replay the triggering config the way the
    search ran it; add `-c <config> -t <oracle>`."""
    target = None
    if triggering.target is not None:
        warp = triggering.target.get("warp")
        if warp is not None:
            target = DelayTarget(
                warp=int(warp),
                cluster_rank=triggering.target.get("cluster_rank"),
            )
    args = _targeting_args(config, warpgroup=triggering.warpgroup_id, target=target)
    # Reduce's per-replay outer timeout defaults to 300 s while a search runs
    # its attempts with `config.timeout` (1800 s by default): a replay killed
    # by the outer timeout before CUTracer's kernel timeout fires cannot be
    # recognized as the hang the search found.
    args += ["--timeout", str(config.timeout)]
    if triggering.symptom == SYMPTOM_HANG:
        args.append("--hang-is-bug")
    return args


@dataclass(frozen=True)
class _LaunchGeometry:
    """One instrumented kernel's largest launch shape a control run observed."""

    warps: int
    cluster_size: int
    checksum: str = ""

    def merge(self, other: Optional["_LaunchGeometry"]) -> "_LaunchGeometry":
        if other is None:
            return self
        return _LaunchGeometry(
            max(self.warps, other.warps),
            max(self.cluster_size, other.cluster_size),
            self.checksum or other.checksum,
        )


def _kernel_launch_events(output_dir: str) -> List[Dict[str, Any]]:
    """Every readable `kernel_launch` event in the kernel-events files of a run."""
    events: List[Dict[str, Any]] = []
    try:
        names = sorted(os.listdir(output_dir))
    except OSError:
        return events
    for name in names:
        if not name.startswith("cutracer_kernel_events_") or not name.endswith(
            ".ndjson"
        ):
            continue
        try:
            with open(os.path.join(output_dir, name)) as source:
                for line in source:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(event, dict) and event.get("type") == "kernel_launch":
                        events.append(event)
        except (OSError, UnicodeError):
            continue
    return events


def _block_warps(event: Mapping[str, Any]) -> Optional[int]:
    block = event.get("block")
    if (
        not isinstance(block, list)
        or len(block) != 3
        or any(type(dimension) is not int or dimension <= 0 for dimension in block)
    ):
        return None
    return (block[0] * block[1] * block[2] + 31) // 32


def _cluster_size(event: Mapping[str, Any]) -> int:
    """The launch's cluster size; 1 without clusters or from a CUTracer that did not record it."""
    size = event.get("cluster_size", 1)
    return size if type(size) is int and size >= 1 else 1


def _instrumented_geometries(
    output_dir: str, config_path: str
) -> List[_LaunchGeometry]:
    """Per-kernel launch shapes of the launches whose kernels have delay points.

    `config_path` is the run's delay dump: it lists every instrumented kernel
    and its candidate points, so launches of kernels that have nothing to delay
    (or were not instrumented at all) do not widen the search. Shapes stay
    per kernel (in first-launch order): merging the warps of one kernel with
    the cluster size of another could invent targets no launch has, and
    keeping only the widest kernel would drop the other kernels' targets.
    """
    try:
        with open(config_path) as source:
            kernels = json.load(source).get("kernels", {})
        checksums = {
            kernel.get("kernel_checksum")
            for kernel in kernels.values()
            if isinstance(kernel, dict) and kernel.get("instrumentation_points")
        }
    except (AttributeError, OSError, UnicodeError, json.JSONDecodeError):
        return []
    by_checksum: Dict[str, _LaunchGeometry] = {}
    for event in _kernel_launch_events(output_dir):
        warps = _block_warps(event)
        checksum = event.get("kernel_checksum")
        if warps is None or not isinstance(checksum, str) or checksum not in checksums:
            continue
        launch = _LaunchGeometry(warps, _cluster_size(event), checksum)
        by_checksum[checksum] = launch.merge(by_checksum.get(checksum))
    return list(by_checksum.values())


def _merge_geometries(
    geometries: List[_LaunchGeometry], new: List[_LaunchGeometry]
) -> List[_LaunchGeometry]:
    """Fold another control run's per-kernel shapes into `geometries`."""
    by_checksum = {geometry.checksum: geometry for geometry in geometries}
    order = [geometry.checksum for geometry in geometries]
    for geometry in new:
        if geometry.checksum in by_checksum:
            by_checksum[geometry.checksum] = geometry.merge(
                by_checksum[geometry.checksum]
            )
        else:
            by_checksum[geometry.checksum] = geometry
            order.append(geometry.checksum)
    return [by_checksum[checksum] for checksum in order]


def _isolate_targets(  # noqa: C901
    config: StressConfig, geometries: Sequence[_LaunchGeometry]
) -> List[DelayTarget]:
    """The isolate targets in search order.

    Targets are grouped by instrumented kernel in first-launch order; within
    each kernel, cluster ranks run high to low and warps ascending. Higher
    ranks come first: in a multi-CTA kernel rank 0 usually leads, and a
    warp of a peer CTA whose work is predicated off is the one most easily left
    out of the release protocol. Targets are collected from every instrumented
    launch shape, without combining one kernel's warps with another's cluster
    ranks; a target named for several kernels is tried once, because one
    attempt delays that warp of every kernel it launches in.
    """
    if config.isolate_ranks is None or config.isolate_warps is None:
        if not geometries:
            if config.baseline_runs <= 0:
                raise ValueError(
                    "an isolate search that discovers its targets from the "
                    "control arm needs baseline_runs >= 1; pass "
                    "baseline_runs >= 1, or the isolate_ranks and "
                    "isolate_warps explicitly"
                )
            raise ValueError(
                "the control arm recorded no instrumented launch with delay "
                "points, so there is nothing to isolate; check the kernel filter "
                "and the delay patterns, or pass the ranks and warps explicitly"
            )
    if config.isolate_ranks is not None and config.isolate_warps is not None:
        warps = sorted(set(config.isolate_warps))
        ranks: Sequence[Optional[int]] = sorted(set(config.isolate_ranks), reverse=True)
        return [DelayTarget(warp, rank) for rank in ranks for warp in warps]
    targets: List[DelayTarget] = []
    seen: set[DelayTarget] = set()
    for geometry in geometries:
        warps = (
            sorted(set(config.isolate_warps))
            if config.isolate_warps is not None
            else list(range(min(_MAX_WARPS, geometry.warps)))
        )
        if config.isolate_ranks is not None:
            # An explicit rank names a CTA of a multi-CTA cluster (see
            # `_validate_target_launch`); pairing it with a launch that has
            # no such rank would only schedule attempts validation rejects.
            shape_ranks: Sequence[Optional[int]] = sorted(
                {
                    rank
                    for rank in config.isolate_ranks
                    if geometry.cluster_size > 1 and rank < geometry.cluster_size
                },
                reverse=True,
            )
        elif geometry.cluster_size > 1:
            shape_ranks = range(geometry.cluster_size - 1, -1, -1)
        else:
            shape_ranks = (None,)
        for rank in shape_ranks:
            for warp in warps:
                target = DelayTarget(warp, rank)
                if target not in seen:
                    seen.add(target)
                    targets.append(target)
    if not targets:
        if config.isolate_ranks is not None:
            raise ValueError(
                "the explicit isolate_ranks name no cluster rank of a "
                "multi-CTA instrumented launch, so there is nothing to isolate"
            )
        raise ValueError(
            "the instrumented launches have no warp to isolate, so there is "
            "nothing to isolate"
        )
    return targets


def _restored_targets(checkpoint: Sequence[Mapping[str, Any]]) -> List[DelayTarget]:
    """The isolate targets a checkpoint recorded, in their original order.

    A discovered target list is a per-kernel union, so it is not always the
    Cartesian product an explicit `isolate_ranks` / `isolate_warps` pair
    reconstructs (and a launch without clusters contributes targets whose
    cluster rank is None, which no explicit rank names). A resume therefore
    restores the recorded list rather than rebuilding it; the entries are
    still checked, because the resumed schedule is located by counting the
    trials the checkpoint already ran against this list.
    """
    targets: List[DelayTarget] = []
    seen: set[DelayTarget] = set()
    for entry in checkpoint:
        if not isinstance(entry, Mapping):
            raise ValueError("resumed isolate targets must be target records")
        warp = entry.get("warp")
        rank = entry.get("cluster_rank")
        if type(warp) is not int or not 0 <= warp < _MAX_WARPS:
            raise ValueError("resumed isolate targets name a warp outside the CTA")
        if rank is not None and (type(rank) is not int or rank < 0):
            raise ValueError("resumed isolate targets name an invalid cluster rank")
        target = DelayTarget(warp, rank)
        if "warp_mask" in entry and entry["warp_mask"] != target.warp_mask:
            raise ValueError("resumed isolate targets name an inconsistent warp mask")
        if target in seen:
            raise ValueError("resumed isolate targets contain a duplicate target")
        seen.add(target)
        targets.append(target)
    if not targets:
        raise ValueError("resumed isolate targets must not be empty")
    return targets


@dataclass(frozen=True)
class _Trial:
    delay_ns: int
    attempt: int
    warpgroup: Optional[int] = None
    target: Optional[DelayTarget] = None


def _trial_schedule(config: StressConfig, targets: List[DelayTarget]) -> List[_Trial]:
    """Every planned attempt, in run order.

    A random search sweeps the delay ladder outermost, each delay over every
    warpgroup. An isolate search gives each target every delay before moving
    to the next target, so a finding on an early target does not wait for the
    whole ladder to run on every other target first.
    """
    attempts = range(config.attempts_per_delay)
    if get_mode(config.strategy).isolates_targets:
        return [
            _Trial(delay_ns, attempt, target=target)
            for target in targets
            for delay_ns in config.delay_ladder_ns
            for attempt in attempts
        ]
    warpgroups: List[Optional[int]] = [int(x) for x in config.warpgroup_ids] or [None]
    return [
        _Trial(delay_ns, attempt, warpgroup=warpgroup)
        for delay_ns in config.delay_ladder_ns
        for warpgroup in warpgroups
        for attempt in attempts
    ]


def _validate_delay_config(  # noqa: C901
    path: str,
    *,
    delay_ns: int,
    expected_kernel_name: Optional[str],
    expected_kernel_checksum: Optional[str],
) -> tuple[Optional[str], List[tuple[str, str]], Optional[int]]:
    """Return validated kernel identities and enabled-PC count for a config.

    Every eligible kernel with enabled points is returned, not just the
    first: an attempt's target may belong to any of them, so target
    validation must not bind itself to one kernel's launch shape.
    """
    if not os.path.isfile(path):
        return "CUTracer wrote no delay config", [], None
    try:
        with open(path) as source:
            payload = json.load(source)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return f"delay config is unreadable: {exc}", [], None
    if not isinstance(payload, dict) or payload.get("version") != "1.0":
        return "delay config has an unsupported or missing version", [], None
    configured_delay = payload.get("delay_ns")
    if type(configured_delay) is not int or configured_delay != delay_ns:
        return (
            "delay config does not match the attempted delay "
            f"({configured_delay!r} != {delay_ns})",
            [],
            None,
        )
    kernels = payload.get("kernels")
    if not isinstance(kernels, dict) or not kernels:
        return "delay config contains no instrumented kernels", [], None

    identities: List[tuple[str, str]] = []
    enabled_count = 0
    saw_eligible_kernel = False
    for raw_kernel in kernels.values():
        if not isinstance(raw_kernel, dict):
            continue
        kernel_name = raw_kernel.get("kernel_name")
        kernel_checksum = raw_kernel.get("kernel_checksum")
        if not isinstance(kernel_name, str) or not kernel_name:
            continue
        if not isinstance(kernel_checksum, str) or not kernel_checksum:
            continue
        if expected_kernel_checksum is not None:
            if kernel_checksum != expected_kernel_checksum:
                continue
        elif expected_kernel_name is not None and kernel_name != expected_kernel_name:
            continue
        saw_eligible_kernel = True
        points = raw_kernel.get("instrumentation_points")
        if not isinstance(points, dict):
            continue
        enabled_pcs: set[int] = set()
        for raw_point in points.values():
            if not isinstance(raw_point, dict) or raw_point.get("on") is not True:
                continue
            point_delay = raw_point.get("delay")
            point_pc = raw_point.get("pc")
            point_sass = raw_point.get("sass")
            if (
                type(point_delay) is int
                and point_delay == delay_ns
                and type(point_pc) is int
                and point_pc >= 0
                and isinstance(point_sass, str)
                and point_sass
            ):
                enabled_pcs.add(point_pc)
        if enabled_pcs:
            identities.append((kernel_name, kernel_checksum))
            enabled_count += len(enabled_pcs)

    if identities:
        return None, identities, enabled_count
    if not saw_eligible_kernel and (
        expected_kernel_name is not None or expected_kernel_checksum is not None
    ):
        return (
            "delay config does not contain the expected kernel identity",
            [],
            None,
        )
    return (
        "delay config contains no enabled point for the expected delay",
        [],
        None,
    )


def _validate_warpgroup_launch(
    output_dir: str,
    *,
    warpgroup: int,
    kernel_identities: Sequence[tuple[str, str]],
) -> Optional[str]:
    """Verify that a targeted warpgroup has at least one live warp at launch."""
    return _validate_target_launch(
        output_dir,
        kernel_identities=kernel_identities,
        warp=4 * warpgroup,
        cluster_rank=None,
        label=f"warpgroup {warpgroup}",
    )


def _validate_target_launch(  # noqa: C901
    output_dir: str,
    *,
    kernel_identities: Sequence[tuple[str, str]],
    warp: int,
    cluster_rank: Optional[int],
    label: str,
) -> Optional[str]:
    """Verify that a matching launch has CTA-local `warp` (and cluster rank).

    A target outside every launch makes the attempt inject nothing, so its
    "not reproduced" would be meaningless. Any eligible kernel may provide
    the matching launch. An explicit cluster rank additionally needs a
    multi-CTA cluster: cluster targeting delays no CTA of a one-CTA cluster,
    so rank 0 of such a launch is not a perturbed target.
    """
    kernel_checksums = {checksum for _name, checksum in kernel_identities}
    matched_launch = False
    matched_without_cluster = False
    try:
        names = os.listdir(output_dir)
    except OSError as exc:
        return f"kernel launch evidence is unreadable: {exc}"
    read_errors: list[str] = []
    for name in names:
        if not name.startswith("cutracer_kernel_events_") or not name.endswith(
            ".ndjson"
        ):
            continue
        path = os.path.join(output_dir, name)
        try:
            with open(path) as source:
                for line in source:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if (
                        not isinstance(event, dict)
                        or event.get("type") != "kernel_launch"
                        or event.get("kernel_checksum") not in kernel_checksums
                    ):
                        continue
                    matched_launch = True
                    block_warps = _block_warps(event)
                    if block_warps is None or warp >= block_warps:
                        continue
                    if cluster_rank is None:
                        return None
                    if _cluster_size(event) > 1 and cluster_rank < _cluster_size(event):
                        return None
                    if _cluster_size(event) <= 1:
                        matched_without_cluster = True
        except (OSError, UnicodeError) as exc:
            read_errors.append(f"{name}: {exc}")
    if read_errors:
        return "kernel launch evidence is unreadable: " + "; ".join(read_errors)
    if matched_without_cluster:
        return (
            f"{label} needs a multi-CTA cluster launch, but the matching "
            "kernel launched with one CTA per cluster"
        )
    if matched_launch:
        return f"{label} has no live warp in the matching kernel launch"
    return f"no matching kernel launch evidence was recorded for {label} targeting"


def _attempt_targeting(
    config: StressConfig,
    *,
    warpgroup: Optional[int],
    target: Optional[DelayTarget],
) -> Dict[str, Any]:
    """The InstrumentationConfig fields that place an attempt's delays."""
    if target is None:
        return {
            "delay_mode": "random",
            "delay_enable_prob": config.enable_prob,
            "delay_warpgroup_id": warpgroup,
        }
    return {
        "delay_mode": "fixed" if target.cluster_rank is None else "cluster_fixed",
        "delay_enable_prob": 1.0,
        "delay_cluster_cta_id": target.cluster_rank,
        "delay_warp_mask": target.warp_mask,
    }


def _attempt_config_name(
    delay_ns: int, warpgroup: Optional[int], target: Optional[DelayTarget], attempt: int
) -> str:
    if target is None:
        return f"delay-{delay_ns}-wg-{warpgroup}-attempt-{attempt}.json"
    rank = "all" if target.cluster_rank is None else target.cluster_rank
    return f"delay-{delay_ns}-rank-{rank}-warp-{target.warp}-attempt-{attempt}.json"


def _run_attempt_in_output_dir(  # noqa: C901
    config: StressConfig,
    *,
    delay_ns: int,
    warpgroup: Optional[int],
    attempt: int,
    cutracer_so: Optional[str],
    runner: Optional[RunTarget],
    instrumentation_output_dir: str,
    target: Optional[DelayTarget] = None,
) -> _AttemptResult:
    config_path = os.path.join(
        config.output_dir,
        _attempt_config_name(delay_ns, warpgroup, target, attempt),
    )
    try:
        os.unlink(config_path)
    except FileNotFoundError:
        pass
    logs_before = main_logs(instrumentation_output_dir)
    proc: Optional[subprocess.CompletedProcess[str]] = None
    timeout_error = ""
    try:
        proc = run_instrumented_target(
            config.oracle_argv,
            InstrumentationConfig(
                cutracer_so=cutracer_so,
                instrument="random_delay",
                analysis="random_delay",
                kernel_filters=config.kernel_filters,
                exact_kernel_name=config.exact_kernel_name,
                exact_kernel_checksum=config.exact_kernel_checksum,
                output_dir=instrumentation_output_dir,
                delay_ns=delay_ns,
                delay_patterns=_delay_patterns(config),
                delay_dump_path=config_path,
                # A fresh seed per attempt, even when the caller's environment
                # pins one: every process of the attempt shares it.
                delay_seed=secrets.randbits(64),
                kernel_timeout_s=config.kernel_timeout_s,
                no_data_timeout_s=_no_data_timeout_s(config),
                cwd=config.cwd,
                timeout=config.timeout,
                base_env=config.base_env,
                use_bundled_cuda_tools=config.use_bundled_cuda_tools,
                kernel_events=(
                    "nostack" if warpgroup is not None or target is not None else None
                ),
                **_attempt_targeting(config, warpgroup=warpgroup, target=target),
            ),
            runner=runner or run_process_group,
        )
    except subprocess.TimeoutExpired as exc:
        timeout_error = f"timed out: {exc}"
    except (click.ClickException, OSError, ValueError) as exc:
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
                error=f"launch failed: {type(exc).__name__}: {exc}",
                target=target,
            ),
        )
    finally:
        # The attempt's processes serialize their config merges on this lock;
        # none are left once the target has returned.
        _unlink(config_path + ".lock")

    watchdog = watchdog_terminations(instrumentation_output_dir, before=logs_before)
    hangs = 1 if watchdog else 0
    # A hang ended by CUTracer's watchdog is the symptom itself; the oracle's
    # exit code (or its absence after an outer timeout) then carries nothing.
    hung = bool(watchdog) and config.hang_is_bug
    returncode = None if proc is None else proc.returncode
    log = _attempt_log(
        delay_ns=delay_ns,
        warpgroup=warpgroup,
        attempt=attempt,
        returncode=returncode,
        stdout="" if proc is None else proc.stdout or "",
        stderr="" if proc is None else proc.stderr or "",
        error=timeout_error,
        watchdog=watchdog,
        target=target,
    )
    if returncode is None and not hung:
        _unlink(config_path)
        return _AttemptResult(0, 0, 0, None, log, timed_out=1, hangs=hangs)

    is_interesting = hung or returncode == 0
    is_clean = not is_interesting and (
        not config.not_interesting_exit_codes
        or returncode in config.not_interesting_exit_codes
    )
    if not is_interesting and not is_clean:
        rejected = _preserve_rejected_config(config_path)
        return _AttemptResult(
            0,
            0,
            1,
            None,
            log + f"\n--- error ---\nunexpected oracle returncode {returncode}",
            rejected_config_path=rejected,
            hangs=hangs,
        )

    if (
        is_interesting
        and os.path.isfile(config_path)
        and _enabled_point_count(config_path) == 0
    ):
        # Real failure, but this attempt injected nothing (or its config cannot
        # prove otherwise), so it is evidence about the target rather than an
        # attributable triggering configuration.
        _unlink(config_path)
        return _AttemptResult(
            1,
            0,
            0,
            None,
            log + "\n--- error ---\noracle reported the bug but the delay config "
            "enabled 0 verifiable injection points: the target reproduces "
            "without attributable injected delay",
            unattributed=1,
            hangs=hangs,
        )

    config_error, kernel_identities, enabled_point_count = _validate_delay_config(
        config_path,
        delay_ns=delay_ns,
        expected_kernel_name=config.expected_kernel_name,
        expected_kernel_checksum=config.expected_kernel_checksum,
    )
    if config_error is not None:
        rejected = _preserve_rejected_config(config_path)
        return _AttemptResult(
            0,
            0,
            1,
            None,
            log + f"\n--- error ---\n{config_error}",
            rejected_config_path=rejected,
            hangs=hangs,
        )
    assert kernel_identities
    assert enabled_point_count is not None
    targeting_error: Optional[str] = None
    if target is not None:
        targeting_error = _validate_target_launch(
            instrumentation_output_dir,
            kernel_identities=kernel_identities,
            warp=target.warp,
            cluster_rank=target.cluster_rank,
            label=target.label(),
        )
    elif warpgroup is not None:
        targeting_error = _validate_warpgroup_launch(
            instrumentation_output_dir,
            warpgroup=warpgroup,
            kernel_identities=kernel_identities,
        )
    if targeting_error is not None:
        rejected = _preserve_rejected_config(config_path)
        return _AttemptResult(
            0,
            0,
            1,
            None,
            log + f"\n--- error ---\n{targeting_error}",
            rejected_config_path=rejected,
            hangs=hangs,
        )

    if is_interesting:
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
                enabled_point_count=enabled_point_count,
                symptom=SYMPTOM_HANG if hung else SYMPTOM_FAILURE,
                target=None if target is None else target.to_dict(),
                replay_args=_replay_args(
                    config, delay_ns=delay_ns, warpgroup=warpgroup, target=target
                ),
            ),
            log,
            hangs=hangs,
        )
    _unlink(config_path)
    return _AttemptResult(1, 0, 0, None, log, hangs=hangs)


def _run_attempt(
    config: StressConfig,
    *,
    delay_ns: int,
    warpgroup: Optional[int],
    attempt: int,
    cutracer_so: Optional[str],
    runner: Optional[RunTarget],
    target: Optional[DelayTarget] = None,
) -> _AttemptResult:
    event_dir = (
        tempfile.mkdtemp(prefix=".stress_events_", dir=config.output_dir)
        if warpgroup is not None or target is not None
        else None
    )
    try:
        return _run_attempt_in_output_dir(
            config,
            delay_ns=delay_ns,
            warpgroup=warpgroup,
            attempt=attempt,
            cutracer_so=cutracer_so,
            runner=runner,
            instrumentation_output_dir=event_dir or config.output_dir,
            target=target,
        )
    finally:
        if event_dir is not None:
            shutil.rmtree(event_dir, ignore_errors=True)


def _fsync_directory(path: str) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_stress_log(
    output_dir: str,
    log_part: str,
    *,
    append: bool,
) -> tuple[str, int]:
    """Durably append one trial, rolling back a failed partial write."""
    log_path = os.path.join(output_dir, "stress.log")
    previous_size = 0
    try:
        if append:
            try:
                previous_size = os.path.getsize(log_path)
            except FileNotFoundError as exc:
                raise OSError(
                    "durable stress log disappeared before the next append"
                ) from exc
        payload = (("\n" if previous_size else "") + log_part).encode("utf-8")
        with open(log_path, "ab" if append else "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        if not append:
            _fsync_directory(output_dir)
        return log_path, previous_size + len(payload)
    except OSError:
        if append:
            try:
                with open(log_path, "r+b") as output:
                    output.truncate(previous_size)
                    output.flush()
                    os.fsync(output.fileno())
            except OSError:
                pass
        else:
            _unlink(log_path)
        raise


def _prepare_stress_log(
    output_dir: str,
    result: Optional[StressResult],
) -> tuple[Optional[str], int]:
    """Start a live append-only log from the last durable checkpoint prefix."""
    log_path = os.path.join(output_dir, "stress.log")
    if result is None:
        _unlink(log_path)
        return None, 0
    attempted = result.completed_trials + result.infra_errors + result.timed_out
    if attempted == 0:
        _unlink(log_path)
        return None, 0
    if result.log_path is None:
        raise ValueError("resumed stress progress requires a persisted log")
    source_size = os.path.getsize(result.log_path)
    expected_size = (
        source_size if result.log_size_bytes is None else result.log_size_bytes
    )
    if expected_size < 0 or source_size < expected_size:
        raise ValueError("resumed stress log is shorter than its durable prefix")

    if os.path.abspath(result.log_path) == os.path.abspath(log_path):
        if source_size > expected_size:
            with open(log_path, "r+b") as output:
                output.truncate(expected_size)
                output.flush()
                os.fsync(output.fileno())
        return log_path, expected_size

    descriptor, temporary = tempfile.mkstemp(prefix=".stress.log.", dir=output_dir)
    try:
        with (
            open(result.log_path, "rb") as source,
            os.fdopen(descriptor, "wb") as output,
        ):
            descriptor = -1
            remaining = expected_size
            while remaining:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError(
                        "resumed stress log ended before its durable prefix"
                    )
                output.write(chunk)
                remaining -= len(chunk)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, log_path)
        temporary = ""
        _fsync_directory(output_dir)
        return log_path, expected_size
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary:
            _unlink(temporary)


@dataclass(frozen=True)
class _BaselineRun:
    """One control run: the oracle's exit code (None when the run timed out)
    and whether CUTracer's watchdog ended a hung kernel in it."""

    returncode: Optional[int]
    hung: bool


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
    hangs: int = 0
    # Per-kernel instrumented launch shapes, recorded for isolate searches only.
    geometries: List[_LaunchGeometry] = field(default_factory=list)
    # The raw runs, so that a control arm shared by several modes can be
    # counted under each mode's own oracle contract (see `_baseline_from_runs`).
    runs: List[_BaselineRun] = field(default_factory=list)


def _baseline_from_runs(
    config: StressConfig,
    runs: Sequence[_BaselineRun],
    *,
    logs: List[str],
    geometries: List[_LaunchGeometry],
    requested: Optional[int] = None,
) -> _BaselineResult:
    """Count control runs under `config`'s oracle contract and hang rule."""
    completed = reproduced = timed_out = infra_errors = hangs = 0
    for run in runs:
        hangs += 1 if run.hung else 0
        if run.hung and config.hang_is_bug:
            # The target hangs without any injected delay.
            completed += 1
            reproduced += 1
        elif run.returncode is None:
            timed_out += 1
        elif run.returncode == 0:
            completed += 1
            reproduced += 1
        elif (
            not config.not_interesting_exit_codes
            or run.returncode in config.not_interesting_exit_codes
        ):
            completed += 1
        else:
            # Same rule as the search arm: an exit code the oracle contract does
            # not cover means the control run did not measure anything.
            infra_errors += 1
    return _BaselineResult(
        requested=config.baseline_runs if requested is None else requested,
        completed=completed,
        reproductions=reproduced,
        timed_out=timed_out,
        infra_errors=infra_errors,
        logs=logs,
        hangs=hangs,
        geometries=geometries,
        runs=list(runs),
    )


def _remaining_s(deadline: Optional[float]) -> Optional[float]:
    return None if deadline is None else deadline - time.monotonic()


def _run_timeout(config: StressConfig, remaining: Optional[float]) -> int:
    """The per-run timeout, cut short so that a run ends by the deadline."""
    if remaining is None:
        return config.timeout
    return max(1, min(config.timeout, int(remaining)))


def _run_baseline(
    config: StressConfig,
    *,
    cutracer_so: Optional[str],
    runner: Optional[RunTarget],
    capture_geometry: Optional[bool] = None,
    deadline: Optional[float] = None,
) -> _BaselineResult:
    """Run the control arm: instrumented, but with every delay point disabled.

    A *matched* control on purpose. Running the oracle bare would also measure a
    natural rate, but it would differ from the search arm in two variables at
    once (instrumentation overhead and injected stalls), so it could not tell
    them apart. `--delay-enable-prob 0` keeps the instrumentation and removes
    only the stalls, which is the variable the search manipulates.

    No dump path is passed for a random search: an all-disabled config is not
    worth keeping, and not writing one keeps the baseline out of the search's
    config namespace. An isolate search runs each control run in a scratch
    directory with a dump and kernel events, to learn which warps and cluster
    ranks the instrumented launches have; `capture_geometry` does the same for
    a control arm that several modes share.

    No run starts with less than `MIN_RUN_BUDGET_S` left before `deadline`;
    the runs that do not start leave the control arm incomplete.
    """
    runs: List[_BaselineRun] = []
    geometries: List[_LaunchGeometry] = []
    logs: List[str] = []
    if config.baseline_runs <= 0:
        # A caller that opted out of the control arm must not pay for anything
        # here -- including reading `delay_ladder_ns[0]`.
        return _BaselineResult(0, 0, 0, 0, 0, logs)
    delay_ns = config.delay_ladder_ns[0]
    capture = (
        get_mode(config.strategy).isolates_targets
        if capture_geometry is None
        else capture_geometry
    )
    for attempt in range(config.baseline_runs):
        remaining = _remaining_s(deadline)
        if remaining is not None and remaining < MIN_RUN_BUDGET_S:
            break
        run_dir = (
            tempfile.mkdtemp(prefix=".stress_baseline_", dir=config.output_dir)
            if capture
            else config.output_dir
        )
        dump_path = os.path.join(run_dir, "delay_config.json") if capture else None
        logs_before = main_logs(run_dir)
        proc: Optional[subprocess.CompletedProcess[str]] = None
        error = "baseline (injection disabled)"
        try:
            proc = run_instrumented_target(
                config.oracle_argv,
                InstrumentationConfig(
                    cutracer_so=cutracer_so,
                    instrument="random_delay",
                    analysis="random_delay",
                    kernel_filters=config.kernel_filters,
                    exact_kernel_name=config.exact_kernel_name,
                    exact_kernel_checksum=config.exact_kernel_checksum,
                    output_dir=run_dir,
                    delay_ns=delay_ns,
                    delay_enable_prob=0.0,
                    delay_mode="random",
                    delay_patterns=_delay_patterns(config),
                    delay_dump_path=dump_path,
                    kernel_events="nostack" if capture else None,
                    kernel_timeout_s=config.kernel_timeout_s,
                    no_data_timeout_s=_no_data_timeout_s(config),
                    cwd=config.cwd,
                    timeout=_run_timeout(config, remaining),
                    base_env=config.base_env,
                    use_bundled_cuda_tools=config.use_bundled_cuda_tools,
                ),
                runner=runner or run_process_group,
            )
        except subprocess.TimeoutExpired as exc:
            error = f"baseline timed out: {exc}"
        watchdog = watchdog_terminations(run_dir, before=logs_before)
        if dump_path is not None:
            geometries = _merge_geometries(
                geometries, _instrumented_geometries(run_dir, dump_path)
            )
            shutil.rmtree(run_dir, ignore_errors=True)
        logs.append(
            _attempt_log(
                delay_ns=delay_ns,
                warpgroup=None,
                attempt=attempt,
                stdout="" if proc is None else proc.stdout or "",
                stderr="" if proc is None else proc.stderr or "",
                error=error,
                watchdog=watchdog,
            )
        )
        runs.append(
            _BaselineRun(None if proc is None else proc.returncode, bool(watchdog))
        )
    return _baseline_from_runs(config, runs, logs=logs, geometries=geometries)


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


# What a search shows (see `result_verdict`).
VERDICT_FINDING = "finding"
VERDICT_SUSPECT = "suspect"
VERDICT_CLEAN = "clean"
VERDICT_INCOMPLETE = "incomplete"


@dataclass(frozen=True)
class Verdict:
    status: str
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, "reasons": list(self.reasons)}


def _unattributed_reason(result: StressResult) -> str:
    if result.baseline_reproductions:
        return (
            f"the control arm reproduced {result.baseline_reproductions} of "
            f"{result.baseline_completed} run(s) with injection disabled"
        )
    if result.attribution == ATTRIBUTION_NO_BASELINE:
        return "reproduced, but no control arm ran"
    return (
        f"reproduced, but only {result.baseline_completed} of "
        f"{result.baseline_requested} control run(s) produced a sample"
    )


def _incomplete_reasons(result: StressResult) -> List[str]:
    reasons: List[str] = []
    if result.baseline_completed < result.baseline_requested:
        reasons.append(
            f"only {result.baseline_completed} of {result.baseline_requested} "
            "control run(s) produced a sample"
        )
    if result.infra_errors:
        reasons.append(f"{result.infra_errors} attempt(s) were infra errors")
    if result.timed_out:
        reasons.append(f"{result.timed_out} attempt(s) timed out")
    if result.hangs:
        reasons.append(
            f"CUTracer's watchdog ended a hung kernel in {result.hangs} attempt(s)"
        )
    attempted = result.completed_trials + result.infra_errors + result.timed_out
    if result.budget_exhausted:
        reasons.append(
            f"the time budget ran out after {attempted} of "
            f"{result.planned_trials} planned attempt(s)"
        )
    elif result.planned_trials is not None and attempted < result.planned_trials:
        reasons.append(
            f"only {attempted} of {result.planned_trials} planned attempt(s) ran"
        )
    return reasons


def result_verdict(result: StressResult) -> Verdict:
    """Classify one search.

    - finding: an attempt reproduced the bug and the control arm never did, so
      the injected delay is what made the difference.
    - suspect: the bug reproduced, but not attributably to an injected delay:
      it also reproduced with injection disabled, or in an attempt that
      injected nothing, or without a complete control arm.
    - clean: every planned attempt ran and measured something (no infra error,
      timeout or unexplained hang), after a complete control arm if one was
      requested, and nothing reproduced. It covers only the attempts made.
    - incomplete: anything else.
    """
    triggering = result.triggering
    if triggering is not None:
        if result.attribution == ATTRIBUTION_ATTRIBUTED:
            return Verdict(
                VERDICT_FINDING,
                [
                    f"reproduced ({triggering.symptom}) at "
                    f"delay_ns={triggering.delay_ns}"
                ],
            )
        return Verdict(VERDICT_SUSPECT, [_unattributed_reason(result)])
    if result.baseline_reproductions:
        return Verdict(VERDICT_SUSPECT, [_unattributed_reason(result)])
    if result.unattributed_reproductions:
        return Verdict(
            VERDICT_SUSPECT,
            [
                f"{result.unattributed_reproductions} attempt(s) reported the bug "
                "while injecting no delay"
            ],
        )
    reasons = _incomplete_reasons(result)
    return Verdict(VERDICT_INCOMPLETE, reasons) if reasons else Verdict(VERDICT_CLEAN)


def run_stress(  # noqa: C901
    config: StressConfig,
    *,
    cutracer_so: Optional[str] = None,
    runner: Optional[RunTarget] = None,
    baseline: Optional[_BaselineResult] = None,
    deadline: Optional[float] = None,
) -> StressResult:
    """Run the planned attempts (see `_trial_schedule`) until the oracle reproduces.

    A "reproduction" requires an interesting oracle exit (0), a dumped delay
    config (otherwise there is nothing to replay/reduce, so the attempt is
    counted as an infra error), AND at least one enabled injection point in that
    config (otherwise the attempt injected nothing and cannot have caused the
    failure -- counted as an unattributed reproduction).

    `baseline` is a control arm that already ran with this search's
    instrumentation (see `cutracer.stress.plan`); it is counted under this
    search's oracle contract instead of running the control arm again. No
    attempt starts with less than `MIN_RUN_BUDGET_S` left before `deadline`
    (a `time.monotonic()` value); the result then says `budget_exhausted`.
    """
    os.makedirs(config.output_dir, exist_ok=True)
    resume = config.resume_result
    completed = 0 if resume is None else resume.completed_trials
    reproductions = 0 if resume is None else resume.reproductions
    infra_errors = 0 if resume is None else resume.infra_errors
    timed_out = 0 if resume is None else resume.timed_out
    hangs = 0 if resume is None else resume.hangs
    unattributed = 0 if resume is None else resume.unattributed_reproductions
    if baseline is not None:
        if resume is not None:
            raise ValueError("a shared control arm cannot resume a search")
        baseline = _baseline_from_runs(
            config,
            baseline.runs,
            logs=[],
            geometries=baseline.geometries,
            requested=baseline.requested,
        )
    elif resume is None:
        baseline = _run_baseline(
            config, cutracer_so=cutracer_so, runner=runner, deadline=deadline
        )
    else:
        baseline = _BaselineResult(
            requested=resume.baseline_requested,
            completed=resume.baseline_completed,
            reproductions=resume.baseline_reproductions,
            timed_out=resume.baseline_timed_out,
            infra_errors=resume.baseline_infra_errors,
            logs=[],
            hangs=resume.baseline_hangs,
        )
    rejected_config_paths: List[str] = _bounded_rejected_config_paths(
        [] if resume is None else list(resume.rejected_config_paths),
        owned_root=config.output_dir,
    )
    triggering = None if resume is None else resume.triggering
    stop = False
    targets: List[DelayTarget] = []
    restored_targets = False
    if get_mode(config.strategy).isolates_targets:
        if resume is not None and resume.targets:
            # The checkpoint's recorded targets are the original schedule;
            # see `_restored_targets` for why they cannot be rebuilt from
            # explicit ranks and warps.
            targets = _restored_targets(resume.targets)
            restored_targets = True
            if (
                config.isolate_ranks is not None
                and any(
                    target.cluster_rank is not None
                    and target.cluster_rank not in config.isolate_ranks
                    for target in targets
                )
            ) or (
                config.isolate_warps is not None
                and any(target.warp not in config.isolate_warps for target in targets)
            ):
                raise ValueError(
                    "resuming an isolate search needs explicit isolate_ranks "
                    "and isolate_warps that cover the checkpoint targets"
                )
        else:
            if resume is not None and (
                config.isolate_ranks is None or config.isolate_warps is None
            ):
                # The control arm that chose the targets does not run again,
                # and an older checkpoint recorded no targets to restore.
                raise ValueError(
                    "resuming an isolate search needs explicit isolate_ranks and "
                    "isolate_warps"
                )
            targets = _isolate_targets(config, baseline.geometries)
    schedule = _trial_schedule(config, targets)
    planned_trials = len(schedule)
    if resume is not None:
        # The remaining trials are located by counting the ones already run,
        # which is only the same position under the original schedule shape.
        if (
            resume.delay_ladder_ns is not None
            and list(resume.delay_ladder_ns) != list(config.delay_ladder_ns)
        ) or (
            resume.attempts_per_delay is not None
            and resume.attempts_per_delay != config.attempts_per_delay
        ):
            raise ValueError(
                "resuming a stress search needs the original delay_ladder_ns "
                "and attempts_per_delay; the remaining trials are located by "
                "counting the ones already run"
            )
        if (
            not restored_targets
            and resume.targets
            and resume.targets != [target.to_dict() for target in targets]
        ):
            raise ValueError(
                "resuming an isolate search needs the original targets; the "
                "remaining trials are located by counting the ones already run"
            )
    attempted_trials = completed + infra_errors + timed_out
    if min(completed, reproductions, unattributed, infra_errors, timed_out) < 0:
        raise ValueError("resumed stress progress contains negative counts")
    if reproductions + unattributed > completed or attempted_trials > planned_trials:
        raise ValueError("resumed stress progress is outside the campaign plan")
    if triggering is not None and reproductions == 0:
        raise ValueError("resumed triggering config has no reproduced trial")
    if triggering is not None and not os.path.isfile(triggering.config_path):
        raise ValueError("resumed triggering config is unavailable")
    if reproductions > 0 and triggering is None:
        raise ValueError("resumed reproduction is missing its triggering config")
    log_path, log_size_bytes = _prepare_stress_log(config.output_dir, resume)
    if baseline.logs:
        log_path, log_size_bytes = _write_stress_log(
            config.output_dir,
            "\n".join(baseline.logs),
            append=log_path is not None,
        )
    if triggering is not None and config.stop_on_first:
        stop = True
    budget_exhausted = False

    def progress_result(log_path: Optional[str]) -> StressResult:
        return StressResult(
            reproduced=triggering is not None,
            completed_trials=completed,
            reproductions=reproductions,
            infra_errors=infra_errors,
            reproduction_rate=reproductions / completed if completed else 0.0,
            triggering=triggering,
            log_path=log_path,
            timed_out=timed_out,
            hangs=hangs,
            unattributed_reproductions=unattributed,
            baseline_requested=baseline.requested,
            baseline_completed=baseline.completed,
            baseline_reproductions=baseline.reproductions,
            baseline_timed_out=baseline.timed_out,
            baseline_infra_errors=baseline.infra_errors,
            baseline_hangs=baseline.hangs,
            attribution=_attribution(
                baseline_requested=baseline.requested,
                baseline_completed=baseline.completed,
                baseline_reproductions=baseline.reproductions,
                triggering=triggering,
            ),
            log_size_bytes=log_size_bytes if log_path is not None else None,
            rejected_config_paths=list(rejected_config_paths),
            strategy=config.strategy,
            targets=[target.to_dict() for target in targets],
            delay_ladder_ns=list(config.delay_ladder_ns),
            attempts_per_delay=config.attempts_per_delay,
            planned_trials=planned_trials,
            budget_exhausted=budget_exhausted,
        )

    for trial in schedule[attempted_trials:]:
        if stop:
            break
        remaining = _remaining_s(deadline)
        if remaining is not None and remaining < MIN_RUN_BUDGET_S:
            budget_exhausted = True
            break
        outcome = _run_attempt(
            (
                config
                if remaining is None or remaining >= config.timeout
                else dataclasses.replace(
                    config, timeout=_run_timeout(config, remaining)
                )
            ),
            delay_ns=trial.delay_ns,
            warpgroup=trial.warpgroup,
            attempt=trial.attempt,
            cutracer_so=cutracer_so,
            runner=runner,
            target=trial.target,
        )
        completed += outcome.completed_trials
        reproductions += outcome.reproductions
        infra_errors += outcome.infra_errors
        timed_out += outcome.timed_out
        hangs += outcome.hangs
        unattributed += outcome.unattributed
        _record_rejected_config(
            rejected_config_paths,
            outcome.rejected_config_path,
            owned_root=config.output_dir,
        )
        if outcome.triggering is not None:
            if triggering is None:
                triggering = outcome.triggering
            if config.stop_on_first:
                stop = True

        try:
            log_path, log_size_bytes = _write_stress_log(
                config.output_dir,
                outcome.log,
                append=log_path is not None,
            )
        except OSError:
            # Surface the just-completed attempt to an adapter before
            # propagating persistence failure. The adapter can retain
            # exact counts even though no durable log is available.
            if config.progress_callback is not None:
                config.progress_callback(progress_result(None))
            raise
        if config.progress_callback is not None:
            config.progress_callback(progress_result(log_path))

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
        hangs=hangs,
        unattributed_reproductions=unattributed,
        baseline_requested=baseline.requested,
        baseline_completed=baseline.completed,
        baseline_reproductions=baseline.reproductions,
        baseline_timed_out=baseline.timed_out,
        baseline_infra_errors=baseline.infra_errors,
        baseline_hangs=baseline.hangs,
        attribution=_attribution(
            baseline_requested=baseline.requested,
            baseline_completed=baseline.completed,
            baseline_reproductions=baseline.reproductions,
            triggering=triggering,
        ),
        log_size_bytes=log_size_bytes if log_path is not None else None,
        rejected_config_paths=list(rejected_config_paths),
        strategy=config.strategy,
        targets=[target.to_dict() for target in targets],
        delay_ladder_ns=list(config.delay_ladder_ns),
        attempts_per_delay=config.attempts_per_delay,
        planned_trials=planned_trials,
        budget_exhausted=budget_exhausted,
    )


def save_report(
    result: StressResult, path: str, extra: Optional[Mapping[str, Any]] = None
) -> None:
    """Atomically write the JSON stress report; `extra` keys override its own."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump({**result.to_dict(), **(extra or {})}, fh, indent=2)
    os.replace(tmp, path)
