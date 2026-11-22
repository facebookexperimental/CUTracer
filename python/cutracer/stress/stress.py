# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Core random-delay stress search for ``cutracer stress``.

Runs a correctness oracle repeatedly under delay injection to discover a delay
configuration that triggers a (data-race) bug, then keeps that config for
deterministic replay / reduction. This is the *discovery* counterpart to
``cutracer reduce`` (which minimizes an already-known triggering config); both
own their oracle-driven delay loops inside CUTracer.

Each mode of ``cutracer.stress.modes`` either delays every warp (or one
warpgroup) at its delay points, or isolates one target per attempt with a fixed
delay (see ``StressConfig.strategy`` and ``StressConfig.isolate_by``).

The loop reuses CUTracer's public instrumented-target runner, so a single
attempt is exactly what ``cutracer trace -i random_delay`` would run -- no
self-subprocessing.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import click
from cutracer.runner import (
    InstrumentationConfig,
    run_instrumented_target,
    run_process_group,
    RunTarget,
)
from cutracer.stress.modes import (
    get_mode,
    get_modes,
    ISOLATE,
    ISOLATE_BY_BARRIER,
    ISOLATE_BY_WARP,
    ISOLATE_KINDS,
    PDL,
    RANDOM,
)
from cutracer.stress.retention import (
    MAX_REJECTED_CONFIGS as _MAX_REJECTED_CONFIGS,
    retain_oldest_and_newest,
)
from cutracer.watchdog import (
    exit_quarantine_changes,
    exit_quarantine_failures,
    main_logs,
    watchdog_terminations,
)


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
# CUTracer's exit quarantine saw a CTA's shared memory change after its last
# thread exited (requires `exit_quarantine_ns`), while the oracle passed.
SYMPTOM_LATE_WRITE = "late_write"

STRATEGY_RANDOM: str = RANDOM.name
STRATEGY_ISOLATE: str = ISOLATE.name
STRATEGY_PDL: str = PDL.name
ISOLATE_DEFAULT_PATTERNS: str | None = ISOLATE.delay_patterns
# CUTracer warp masks are 32 bits wide (CUDA allows 32 warps per CTA).
_MAX_WARPS = 32

# Opcode, kind and barrier operand of `@P0 BAR.SYNC.DEFER_BLOCKING 0x5, 0x40 ;`.
_BARRIER_SASS = re.compile(
    r"^\s*(?:@!?\w+\s+)?(BAR\.(SYNC|ARV|RED)(?:\.\w+)*)\s+([^\s,;]+)"
)
_IMMEDIATE = re.compile(r"^(?:0x[0-9a-fA-F]+|0|[1-9][0-9]*)$")
_CHECKSUM = re.compile(r"^[0-9a-f]+$")


def _barrier_operation(sass: str) -> Optional[Tuple[str, str, str]]:
    """(opcode, kind, barrier operand) of a named-barrier instruction; None
    for any other instruction."""
    match = _BARRIER_SASS.match(sass)
    if match is None:
        return None
    return match.group(1), match.group(2), match.group(3)


def _pattern_list(patterns: Optional[str]) -> List[str]:
    """`CUTRACER_DELAY_PATTERNS` as CUTracer splits it."""
    return [p.strip() for p in (patterns or "").split(",") if p.strip()]


def _selects_named_barriers(patterns: Optional[str], arrives: bool) -> bool:
    """Whether delay patterns can match a named barrier a target delays: a
    pattern that is part of an opcode it delays (`BAR`, `BAR.SYNC`) or that
    names one (`@P0 BAR.SYNC.DEFER_BLOCKING`), but not `MEMBAR`."""
    kinds = ("SYNC", "RED", "ARV") if arrives else ("SYNC", "RED")
    opcode = re.compile(rf"(?<![A-Z_.])BAR\.(?:{'|'.join(kinds)})")
    return any(
        pattern == "*"
        or any(pattern in f"BAR.{kind}" for kind in kinds)
        or opcode.search(pattern) is not None
        for pattern in _pattern_list(patterns)
    )


@dataclass(frozen=True)
class BarrierSite:
    """One named-barrier instruction of an instrumented kernel."""

    kernel_name: str
    kernel_checksum: str
    pc: int
    sass: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kernel_name": self.kernel_name,
            "kernel_checksum": self.kernel_checksum,
            "pc": self.pc,
            "sass": self.sass,
        }

    @classmethod
    def from_dict(cls, entry: object) -> "BarrierSite":
        if not isinstance(entry, Mapping):
            raise ValueError("a named-barrier site must be a site record")
        name, checksum = entry.get("kernel_name"), entry.get("kernel_checksum")
        pc, sass = entry.get("pc"), entry.get("sass")
        if not isinstance(name, str) or not name:
            raise ValueError("a named-barrier site names no kernel")
        if not isinstance(checksum, str) or not _CHECKSUM.match(checksum):
            raise ValueError("a named-barrier site has an invalid kernel checksum")
        if type(pc) is not int or pc < 0:
            raise ValueError("a named-barrier site has an invalid pc")
        if not isinstance(sass, str) or _barrier_operation(sass) is None:
            raise ValueError("a named-barrier site is not a named-barrier instruction")
        return cls(name, checksum, pc, sass)


@dataclass(frozen=True)
class NamedBarrier:
    """The points one named-barrier target delays.

    An immediate barrier id (``barrier_id``) covers every instruction that
    names it, in every instrumented kernel. A barrier whose id is in a
    register is a single site (``barrier_id`` None): two sites that read the
    same register need not name the same barrier. ``patterns`` are SASS
    substrings that select exactly ``sites``; without them, an attempt
    replays a delay config that turns on only ``sites``.
    """

    barrier_id: Optional[int]
    sites: Tuple[BarrierSite, ...]
    patterns: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.sites:
            raise ValueError("a named barrier has at least one site")
        operations = [_barrier_operation(site.sass) for site in self.sites]
        if any(operation is None for operation in operations):
            raise ValueError("a named-barrier site is not a named-barrier instruction")
        operands = {operation[2] for operation in operations if operation}
        if self.barrier_id is None:
            if len(self.sites) != 1 or _IMMEDIATE.match(operands.pop()):
                raise ValueError("a register-id named barrier is one register site")
            if self.patterns:
                raise ValueError("a register-id named barrier has no patterns")
        elif any(
            not _IMMEDIATE.match(operand) or int(operand, 0) != self.barrier_id
            for operand in operands
        ):
            raise ValueError("every site of a named barrier names its id")
        allowed = {f"{op[0]} {op[2]}" for op in operations if op}
        if any(pattern not in allowed for pattern in self.patterns):
            raise ValueError(
                "named-barrier patterns are the opcodes and id of its sites"
            )

    def label(self) -> str:
        if self.barrier_id is not None:
            return f"named barrier {self.barrier_id:#x}"
        site = self.sites[0]
        operation = _barrier_operation(site.sass)
        operand = "?" if operation is None else operation[2]
        return f"named barrier {operand} at pc {site.pc:#x}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.barrier_id,
            "patterns": list(self.patterns),
            "sites": [site.to_dict() for site in self.sites],
        }

    @classmethod
    def from_dict(cls, entry: object) -> "NamedBarrier":
        if not isinstance(entry, Mapping):
            raise ValueError("a named barrier must be a barrier record")
        barrier_id, sites = entry.get("id"), entry.get("sites")
        patterns = entry.get("patterns", [])
        if barrier_id is not None and type(barrier_id) is not int:
            raise ValueError("a named barrier has an invalid id")
        if not isinstance(sites, list) or not isinstance(patterns, list):
            raise ValueError("a named barrier lists its sites and patterns")
        if not all(isinstance(pattern, str) for pattern in patterns):
            raise ValueError("named-barrier patterns must be strings")
        return cls(
            barrier_id,
            tuple(BarrierSite.from_dict(site) for site in sites),
            tuple(patterns),
        )


@dataclass(frozen=True)
class DelayTarget:
    """What one attempt of a mode that isolates targets stalls, with a fixed
    delay: one CTA-local ``warp`` before the mode's delay points, or every
    warp before the points of one named ``barrier`` only.

    ``cluster_rank`` is None for every CTA: launches without clusters, where
    the warp is delayed in every CTA, and named-barrier targets.
    """

    warp: Optional[int] = None
    cluster_rank: Optional[int] = None
    barrier: Optional[NamedBarrier] = None

    def __post_init__(self) -> None:
        if (self.warp is None) == (self.barrier is None):
            raise ValueError("a delay target names one warp or one named barrier")
        if self.barrier is not None and self.cluster_rank is not None:
            raise ValueError("a named-barrier target delays every CTA")

    @property
    def warp_mask(self) -> Optional[str]:
        """The CUTracer warp mask of the target; None for every warp."""
        return None if self.warp is None else hex(1 << self.warp)

    def label(self) -> str:
        if self.barrier is not None:
            return self.barrier.label()
        if self.cluster_rank is None:
            return f"warp {self.warp}"
        return f"cluster rank {self.cluster_rank}, warp {self.warp}"

    def to_dict(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "cluster_rank": self.cluster_rank,
            "warp": self.warp,
            "warp_mask": self.warp_mask,
        }
        if self.barrier is not None:
            record["barrier"] = self.barrier.to_dict()
        return record

    @classmethod
    def from_dict(cls, entry: object) -> "DelayTarget":
        """The target a `to_dict()` record (or an older report's) names."""
        if not isinstance(entry, Mapping):
            raise ValueError("a delay target must be a target record")
        rank = entry.get("cluster_rank")
        if rank is not None and (type(rank) is not int or rank < 0):
            raise ValueError("a delay target names an invalid cluster rank")
        if entry.get("barrier") is not None:
            if entry.get("warp") is not None or entry.get("warp_mask") is not None:
                raise ValueError("a named-barrier target names no warp")
            return cls(
                cluster_rank=rank, barrier=NamedBarrier.from_dict(entry["barrier"])
            )
        warp = entry.get("warp")
        if type(warp) is not int or not 0 <= warp < _MAX_WARPS:
            raise ValueError("a delay target names a warp outside the CTA")
        target = cls(warp, rank)
        if "warp_mask" in entry and entry["warp_mask"] != target.warp_mask:
            raise ValueError("a delay target names an inconsistent warp mask")
        return target


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
    # A command that runs the workload without CUTracer and exits like the
    # oracle (0 when the bug reproduced). It runs once, before the control
    # arm, and can loop over many launches in one process, which control
    # runs (one instrumented process each) cannot afford: an oracle that
    # compares outputs bitwise fails on a kernel whose result varies from
    # launch to launch (an unordered floating-point reduction) at a rate a
    # few control runs miss, and the search would credit those failures to
    # the delay. None: no plain run.
    plain_oracle_argv: Optional[List[str]] = None
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
    # CUTracer's exit quarantine in every run (ns, 0 = off): the last thread of
    # each instrumented CTA poisons its shared memory and watches it this long.
    # A run in which a word changed is a reproduction, whatever the oracle's
    # exit code: a write landed in a CTA that had finished.
    exit_quarantine_ns: int = 0
    # A mode of `cutracer.stress.modes`. STRATEGY_RANDOM: random per-thread
    # delays at every candidate point, all warps at once or one warpgroup at a
    # time (`warpgroup_ids`). A mode that isolates targets (STRATEGY_ISOLATE:
    # one cluster rank and CTA-local warp) stalls one target per attempt, with
    # a fixed delay before each matching instruction (`enable_prob` is
    # ignored) while everything else runs at full speed. Random delays slow a
    # producer down together with its consumers, so they cannot show that a
    # consumer is missing from the release protocol; isolating that consumer
    # lets the producer lap it.
    # STRATEGY_PDL: a fixed delay on every warp before each global store, with
    # non-blocking launches in every run, for a kernel launched with PDL that
    # reads before an earlier kernel's stores.
    strategy: str = STRATEGY_RANDOM
    # What each attempt of a mode that isolates targets stalls (one of
    # `cutracer.stress.modes.ISOLATE_KINDS`); None = the mode's own.
    isolate_by: Optional[str] = None
    # Isolate targets: cluster ranks (tried high to low) and CTA-local warps.
    # None = every rank / warp of the instrumented launches the control arm
    # observed, which needs `baseline_runs >= 1`.
    isolate_ranks: Optional[List[int]] = None
    isolate_warps: Optional[List[int]] = None
    # Modes that isolate targets: try at most this many targets, a window of
    # the full target list that `isolate_rotation` moves (window k starts at
    # target k * isolate_max_targets and wraps around), so that successive
    # searches with an increasing rotation cover every target.
    isolate_max_targets: Optional[int] = None
    isolate_rotation: int = 0
    # Named-barrier targets: also delay the barrier's arrives (`BAR.ARV`),
    # not only its syncs and reductions.
    barrier_arrives: bool = False
    # SASS substrings to delay at (CUTRACER_DELAY_PATTERNS). None = the
    # mode's own (mbarrier waits for isolate, CUTracer's built-in patterns
    # for random).
    delay_patterns: Optional[str] = None
    cwd: Optional[str] = None
    base_env: Optional[Mapping[str, str]] = None
    resume_result: Optional["StressResult"] = None
    progress_callback: Optional[Callable[["StressResult"], None]] = None
    # Called once the targets are known, before the first attempt, with the
    # strategy, the number of targets (0 without targets) and of attempts.
    schedule_callback: Optional[Callable[[str, int, int], None]] = None
    use_bundled_cuda_tools: bool = True
    expected_kernel_name: Optional[str] = None
    expected_kernel_checksum: Optional[str] = None
    exact_kernel_name: Optional[str] = None
    exact_kernel_checksum: Optional[str] = None

    def __post_init__(self) -> None:  # noqa: C901
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
        if self.plain_oracle_argv is not None and not self.plain_oracle_argv:
            raise ValueError("plain_oracle_argv must name a command")
        if self.kernel_timeout_s < 0:
            raise ValueError(
                f"kernel_timeout_s must be >= 0, got {self.kernel_timeout_s}"
            )
        if not 0 <= self.exit_quarantine_ns <= 0xFFFFFFFF:
            raise ValueError("exit_quarantine_ns must be an unsigned 32-bit ns value")
        if self.strategy not in get_modes():
            raise ValueError(f"unknown stress strategy {self.strategy!r}")
        if self.isolate_by is not None and self.isolate_by not in ISOLATE_KINDS:
            raise ValueError(
                f"isolate_by must be one of {', '.join(ISOLATE_KINDS)}, got "
                f"{self.isolate_by!r}"
            )
        if get_mode(self.strategy).isolates_targets:
            if self.warpgroup_ids:
                raise ValueError("warpgroup_ids do not apply to an isolate search")
        elif (
            self.isolate_by is not None
            or self.isolate_max_targets is not None
            or self.isolate_rotation
        ):
            raise ValueError(
                "isolate_by, isolate_max_targets and isolate_rotation need a mode "
                "that isolates targets"
            )
        if _isolate_by(self) != ISOLATE_BY_WARP and (
            self.isolate_ranks is not None or self.isolate_warps is not None
        ):
            raise ValueError(
                "isolate_ranks and isolate_warps need strategy isolate (one warp "
                "of one cluster rank per attempt)"
            )
        if _isolate_by(self) == ISOLATE_BY_BARRIER:
            if not _selects_named_barriers(_delay_patterns(self), self.barrier_arrives):
                raise ValueError(
                    "a named-barrier search needs delay patterns that match "
                    "BAR.SYNC or BAR.RED (BAR.ARV with barrier_arrives), e.g. "
                    "'BAR.SYNC,BAR.ARV,BAR.RED'"
                )
        elif self.barrier_arrives:
            raise ValueError("barrier_arrives needs isolate_by barrier")
        if self.isolate_max_targets is not None and self.isolate_max_targets < 1:
            raise ValueError("isolate_max_targets must be >= 1")
        if self.isolate_rotation < 0:
            raise ValueError("isolate_rotation must be >= 0")
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
    # The plain oracle's run (`StressConfig.plain_oracle_argv`): how the oracle
    # contract counted it (a `_CONTROL_*` outcome) and its exit code (None when
    # it timed out or did not start); all unset without a plain oracle.
    plain_requested: bool = False
    plain_outcome: Optional[str] = None
    plain_returncode: Optional[int] = None

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
    # How many targets the kernels had, when `isolate_max_targets` limited the
    # search to a window of them; None otherwise.
    targets_total: Optional[int] = None
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
    # What each attempt isolated (`StressConfig.isolate_by` and
    # `barrier_arrives`); a resume must reuse them. None for results
    # reconstructed from older reports, which skip the check.
    isolate_by: Optional[str] = None
    barrier_arrives: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": self.strategy,
            "verdict": result_verdict(self).to_dict(),
            "planned_trials": self.planned_trials,
            "isolate_by": self.isolate_by,
            "barrier_arrives": self.barrier_arrives,
            "budget_exhausted": self.budget_exhausted,
            "targets": list(self.targets),
            "targets_total": self.targets_total,
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
            "plain_requested": self.plain_requested,
            "plain_outcome": self.plain_outcome,
            "plain_returncode": self.plain_returncode,
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


def _run_log(
    header: str,
    *,
    returncode: Optional[int] = None,
    stdout: str = "",
    stderr: str = "",
    error: str = "",
    watchdog: Sequence[str] = (),
    late_writes: Sequence[str] = (),
) -> str:
    sections = [header]
    if returncode is not None:
        sections.append(f"returncode={returncode}")
    if stdout:
        sections.extend(("--- stdout ---", stdout))
    if stderr:
        sections.extend(("--- stderr ---", stderr))
    if watchdog:
        sections.extend(("--- CUTracer watchdog ---", *watchdog))
    if late_writes:
        sections.extend(("--- CUTracer exit quarantine ---", *late_writes))
    if error:
        sections.extend(("--- error ---", error))
    return "\n".join(sections)


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
    late_writes: Sequence[str] = (),
) -> str:
    if target is not None:
        placement = f"target={target.label()}"
    else:
        placement = f"warpgroup={'all' if warpgroup is None else warpgroup}"
    return _run_log(
        f"=== delay_ns={delay_ns} {placement} attempt={attempt} ===",
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        error=error,
        watchdog=watchdog,
        late_writes=late_writes,
    )


def _no_data_timeout_s(config: StressConfig) -> int:
    """The no-data watchdog setting: off whenever the kernel timeout bounds kernels."""
    return 0 if config.kernel_timeout_s > 0 else _DEFAULT_NO_DATA_TIMEOUT_S


def _delay_patterns(config: StressConfig) -> Optional[str]:
    return config.delay_patterns or get_mode(config.strategy).delay_patterns


def _isolate_by(config: StressConfig) -> Optional[str]:
    """What each attempt of `config`'s search stalls (`ISOLATE_KINDS`); None
    for a mode that does not isolate targets."""
    mode = get_mode(config.strategy)
    if not mode.isolates_targets:
        return None
    return config.isolate_by or mode.isolate_by


def _target_patterns(config: StressConfig, target: DelayTarget) -> Optional[str]:
    """The delay patterns of an attempt on `target`: its named barrier's own,
    when patterns select exactly its sites, else the mode's."""
    if target.barrier is not None and target.barrier.patterns:
        return ",".join(target.barrier.patterns)
    return _delay_patterns(config)


def _replays_a_config(target: Optional[DelayTarget]) -> bool:
    """Whether an attempt on `target` replays a delay config written for it
    (`_write_barrier_config`) rather than dumping the points CUTracer chose."""
    return (
        target is not None
        and target.barrier is not None
        and not target.barrier.patterns
    )


def _target_fields(config: StressConfig, target: DelayTarget) -> Dict[str, Any]:
    """The `InstrumentationConfig` / `ReplayConfig` fields that restore the
    targeting of an attempt on `target`."""
    return {
        "delay_mode": "fixed" if target.cluster_rank is None else "cluster_fixed",
        "delay_cluster_cta_id": target.cluster_rank,
        "delay_warp_mask": target.warp_mask,
        "delay_patterns": _target_patterns(config, target),
    }


def _triggering_target(triggering: TriggeringConfig) -> Optional[DelayTarget]:
    """The target a triggering config was recorded under; None for a random
    search, or for a partial record that names no target (which then loses
    the targeting rather than failing the summary)."""
    if triggering.target is None:
        return None
    try:
        return DelayTarget.from_dict(triggering.target)
    except ValueError:
        return None


def _launch_options(config: StressConfig) -> Dict[str, Any]:
    """The InstrumentationConfig fields that set how every run of `config`'s
    mode launches kernels, control runs and replays included."""
    if not get_mode(config.strategy).nonblocking_launches:
        return {}
    return {"nonblocking_launches": True, "cpu_callstack": "0"}


def _launch_args(config: StressConfig) -> List[str]:
    """`_launch_options` as `cutracer trace` / `cutracer reduce` options, with
    the kernel events every run of a non-blocking mode records, so that a
    replay does the same host work per launch (see `_records_launches`)."""
    if not get_mode(config.strategy).nonblocking_launches:
        return []
    return [
        "--nonblocking-launches",
        "--cpu-callstack",
        "0",
        "--kernel-events",
        "nostack",
    ]


def _records_launches(
    config: StressConfig,
    *,
    warpgroup: Optional[int],
    target: Optional[DelayTarget],
) -> bool:
    """Whether an attempt records kernel launch events: to validate its
    targeting, or, with non-blocking launches, to do the same host work per
    launch as its control arm and its replays (whether a consumer launches
    while a producer's stores are pending depends on it)."""
    return (
        warpgroup is not None
        or target is not None
        or get_mode(config.strategy).nonblocking_launches
    )


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
    attempt's targeting, kernel timeout and launch handling."""
    args: List[str] = []
    if config.kernel_filters:
        args += ["-k", config.kernel_filters]
    patterns = _delay_patterns(config)
    if target is not None:
        fields = _target_fields(config, target)
        args += ["--delay-mode", fields["delay_mode"]]
        if target.cluster_rank is not None:
            args += ["--delay-cluster-cta-id", str(target.cluster_rank)]
        if target.warp_mask is not None:
            args += ["--delay-warp-mask", target.warp_mask]
        patterns = fields["delay_patterns"]
    else:
        delay_mode = get_mode(config.strategy).delay_mode
        if delay_mode != "random":
            args += ["--delay-mode", delay_mode]
        if warpgroup is not None:
            args += ["--delay-warpgroup-id", str(warpgroup)]
    if patterns:
        args += ["--delay-patterns", patterns]
    if config.kernel_timeout_s > 0:
        args += ["--kernel-timeout-s", str(config.kernel_timeout_s)]
    if config.exit_quarantine_ns > 0:
        args += ["--exit-quarantine-ns", str(config.exit_quarantine_ns)]
    return args + _launch_args(config)


def _late_writes(config: StressConfig, directory: str, before: set[str]) -> list[str]:
    if config.exit_quarantine_ns <= 0:
        return []
    return exit_quarantine_changes(directory, before=before)


def reduce_args(config: StressConfig, triggering: TriggeringConfig) -> List[str]:
    """`cutracer reduce` options that replay the triggering config the way the
    search ran it; add `-c <config> -t <oracle>`."""
    args = _targeting_args(
        config,
        warpgroup=triggering.warpgroup_id,
        target=_triggering_target(triggering),
    )
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


def _kernel_launch_event_files(output_dir: str) -> List[List[Dict[str, Any]]]:
    """The readable `kernel_launch` events of a run, one list per kernel-events
    file (one file per process, in launch order)."""
    files: List[List[Dict[str, Any]]] = []
    try:
        names = sorted(os.listdir(output_dir))
    except OSError:
        return files
    for name in names:
        if not name.startswith("cutracer_kernel_events_") or not name.endswith(
            ".ndjson"
        ):
            continue
        events: List[Dict[str, Any]] = []
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
        files.append(events)
    return files


def _kernel_launch_events(output_dir: str) -> List[Dict[str, Any]]:
    """Every readable `kernel_launch` event in the kernel-events files of a run."""
    return [
        event for events in _kernel_launch_event_files(output_dir) for event in events
    ]


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


def _listed_kernels(config_path: str) -> set[str]:
    """Checksums of the kernels a run's delay dump lists: every kernel CUTracer
    instrumented, whether or not the delay patterns matched any of its
    instructions."""
    try:
        with open(config_path) as source:
            kernels = json.load(source).get("kernels", {})
        return {
            checksum
            for kernel in kernels.values()
            if isinstance(kernel, dict)
            and isinstance(checksum := kernel.get("kernel_checksum"), str)
        }
    except (AttributeError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        logger.warning(
            "cannot read the control run's delay dump %s: %s", config_path, exc
        )
        return set()


def _launched_instrumented_kernels(output_dir: str, listed: set[str]) -> set[str]:
    """Checksums of the launched kernels among `listed` (`_listed_kernels`)."""
    return {
        checksum
        for event in _kernel_launch_events(output_dir)
        if isinstance(checksum := event.get("kernel_checksum"), str)
        and checksum in listed
    }


def _barrier_sites(output_dir: str, config_path: str) -> List[BarrierSite]:
    """The named-barrier points of the launched kernels in a run's delay
    dump, by kernel in first-launch order, then by PC."""
    try:
        with open(config_path) as source:
            kernels = json.load(source).get("kernels", {})
    except (AttributeError, OSError, UnicodeError, json.JSONDecodeError):
        # `_launched_instrumented_kernels` warns about the same dump.
        return []
    by_checksum: Dict[str, List[BarrierSite]] = {}
    malformed = 0
    for kernel in kernels.values() if isinstance(kernels, dict) else ():
        if not isinstance(kernel, dict):
            continue
        name, checksum = kernel.get("kernel_name"), kernel.get("kernel_checksum")
        points = kernel.get("instrumentation_points")
        if not isinstance(points, dict):
            continue
        sites: List[BarrierSite] = []
        for point in points.values():
            sass = point.get("sass") if isinstance(point, dict) else None
            if isinstance(sass, str) and _barrier_operation(sass) is None:
                # Another instruction the delay patterns matched.
                continue
            try:
                sites.append(
                    BarrierSite.from_dict(
                        {**point, "kernel_name": name, "kernel_checksum": checksum}
                    )
                )
            except (TypeError, ValueError):
                malformed += 1
        by_checksum[str(checksum)] = sorted(sites, key=lambda site: site.pc)
    if malformed:
        logger.warning(
            "skipped %d malformed named-barrier point(s) of the delay dump %s",
            malformed,
            config_path,
        )
    launched = dict.fromkeys(
        checksum
        for event in _kernel_launch_events(output_dir)
        if isinstance(checksum := event.get("kernel_checksum"), str)
    )
    return [site for checksum in launched for site in by_checksum.get(checksum, [])]


def _merge_barrier_sites(
    sites: List[BarrierSite], new: List[BarrierSite]
) -> List[BarrierSite]:
    """Fold another control run's named-barrier sites into `sites`."""
    seen = {(site.kernel_checksum, site.pc) for site in sites}
    merged = list(sites)
    for site in new:
        if (site.kernel_checksum, site.pc) not in seen:
            seen.add((site.kernel_checksum, site.pc))
            merged.append(site)
    return merged


# How many kernel names a reason quotes.
_NAMED_KERNELS = 3


def _first_names(*groups: Sequence[str]) -> Tuple[str, ...]:
    """The first `_NAMED_KERNELS` distinct names of `groups`, in order."""
    names = dict.fromkeys(name for group in groups for name in group)
    return tuple(names)[:_NAMED_KERNELS]


@dataclass(frozen=True)
class _PdlEvidence:
    """What the kernel launch events of completed control runs say about
    programmatic dependent launch (PDL).

    `launches` counts the launches they recorded and `attributed` those
    recorded by a CUTracer that records launch attributes. `graph` counts the
    CUDA graph node launches captured with PDL and `direct` the direct
    launches that used it. Of the direct ones, `first` were the first launch
    of an instrumented kernel and `unpaired` followed no direct launch on
    their stream, so neither overlaps the kernel before it (`_pdl_kind`); the
    rest are `testable`. `covered` counts the testable ones whose producers
    are all instrumented, so that a delay can hold back every store they may
    read early; `uninstrumented` names a few of the producers that are not.
    """

    launches: int = 0
    attributed: int = 0
    direct: int = 0
    graph: int = 0
    first: int = 0
    unpaired: int = 0
    testable: int = 0
    covered: int = 0
    uninstrumented: Tuple[str, ...] = ()

    def merge(self, other: "_PdlEvidence") -> "_PdlEvidence":
        return _PdlEvidence(
            launches=self.launches + other.launches,
            attributed=self.attributed + other.attributed,
            direct=self.direct + other.direct,
            graph=self.graph + other.graph,
            first=self.first + other.first,
            unpaired=self.unpaired + other.unpaired,
            testable=self.testable + other.testable,
            covered=self.covered + other.covered,
            uninstrumented=_first_names(self.uninstrumented, other.uninstrumented),
        )


@dataclass(frozen=True)
class _Producers:
    """The kernels a launch with programmatic stream serialization may read
    early from: the launch before it on its stream and, while each of those
    used stream serialization itself, the launch before that. `instrumented`
    says whether CUTracer instrumented all of them; `uninstrumented` names a
    few that it did not."""

    instrumented: bool
    uninstrumented: Tuple[str, ...] = ()


def _uses_pdl(attributes: Mapping[str, Any]) -> bool:
    """Whether recorded launch attributes ask for PDL: programmatic stream
    serialization (a consumer on the same stream) or a programmatic event (a
    consumer on another stream waits on it)."""
    return _serializes(attributes) or isinstance(
        attributes.get("programmatic_event"), dict
    )


def _serializes(attributes: object) -> bool:
    """Whether launch attributes ask for programmatic stream serialization."""
    return (
        isinstance(attributes, dict)
        and attributes.get("programmatic_stream_serialization") is True
    )


def _producers_after(
    event: Mapping[str, Any], listed: set[str], before: Optional[_Producers]
) -> _Producers:
    """The producers of a launch with stream serialization right after the
    direct launch or graph node launch `event`: `event`, and `before` (the
    producers of `event` itself) when `event` was a direct launch with stream
    serialization."""
    instrumented = event.get("kernel_checksum") in listed
    names = () if instrumented else (str(event.get("kernel_name")),)
    if before is None or not _serializes(event.get("launch_attributes")):
        return _Producers(instrumented, names)
    return _Producers(
        instrumented and before.instrumented,
        _first_names(names, before.uninstrumented),
    )


@dataclass(frozen=True)
class _LaunchContext:
    """A launch event with what came before it in its process: the launch
    before it on its stream, that launch's `_Producers`, and whether it was
    its kernel's first launch."""

    event: Mapping[str, Any]
    before: Optional[Mapping[str, Any]]
    producers: Optional[_Producers]
    first_launch: bool


def _launch_contexts(
    events: Sequence[Mapping[str, Any]], listed: set[str]
) -> Iterator[_LaunchContext]:
    """Each of one process's launch events, in launch order, in context.

    Launches pair up by the stream handle CUTracer records, so the per-thread
    default streams of different threads, or the NULL streams of different
    contexts, look like one stream: a launch can then pair up with another
    thread's or context's launch.
    """
    last: Dict[Any, Tuple[Mapping[str, Any], _Producers]] = {}
    launched: set[object] = set()
    for event in events:
        stream = event.get("stream_id")
        before, producers = last.get(stream, (None, None))
        kernel = event.get("kernel_checksum")
        yield _LaunchContext(event, before, producers, kernel not in launched)
        launched.add(kernel)
        last[stream] = (event, _producers_after(event, listed, producers))


_PDL_FIRST = "first"
_PDL_UNPAIRED = "unpaired"
_PDL_COVERED = "covered"
_PDL_UNCOVERED = "uncovered"


def _pdl_kind(
    context: _LaunchContext, listed: set[str]
) -> Optional[Tuple[str, Optional[_Producers]]]:
    """How a direct launch that uses PDL counts (`_PdlEvidence`), with the
    producers it may read early from; None for any other launch.

    A launch on another stream may wait on a programmatic event and read this
    launch's output early. Stream serialization lets this launch read the
    launches before it early, which overlaps under CUTracer only after a
    direct launch, and not on the first launch of an instrumented kernel,
    which waits for the running kernels while CUTracer instruments it.
    """
    event = context.event
    attributes = event.get("launch_attributes")
    if not isinstance(attributes, dict) or not _uses_pdl(attributes):
        return None
    if isinstance(attributes.get("programmatic_event"), dict):
        producers = _producers_after(event, listed, context.producers)
    elif context.before is None or "launch_attributes" not in context.before:
        return _PDL_UNPAIRED, None
    elif context.first_launch and event.get("kernel_checksum") in listed:
        return _PDL_FIRST, None
    else:
        producers = context.producers
    assert producers is not None
    return (_PDL_COVERED if producers.instrumented else _PDL_UNCOVERED), producers


def _process_pdl_evidence(
    events: Sequence[Mapping[str, Any]], listed: set[str]
) -> _PdlEvidence:
    """`_pdl_evidence` for one process's launch events, in launch order."""
    kinds = dict.fromkeys((_PDL_FIRST, _PDL_UNPAIRED, _PDL_COVERED, _PDL_UNCOVERED), 0)
    attributed = graph = 0
    uninstrumented: Tuple[str, ...] = ()
    for context in _launch_contexts(events, listed):
        if "nonblocking_launches" not in context.event:
            continue
        attributed += 1
        captured = context.event.get("captured_launch_attributes")
        if isinstance(captured, dict) and _uses_pdl(captured):
            graph += 1
        kind = _pdl_kind(context, listed)
        if kind is None:
            continue
        kinds[kind[0]] += 1
        if kind[1] is not None and not kind[1].instrumented:
            uninstrumented = _first_names(uninstrumented, kind[1].uninstrumented)
    testable = kinds[_PDL_COVERED] + kinds[_PDL_UNCOVERED]
    return _PdlEvidence(
        launches=len(events),
        attributed=attributed,
        direct=testable + kinds[_PDL_FIRST] + kinds[_PDL_UNPAIRED],
        graph=graph,
        first=kinds[_PDL_FIRST],
        unpaired=kinds[_PDL_UNPAIRED],
        testable=testable,
        covered=kinds[_PDL_COVERED],
        uninstrumented=uninstrumented,
    )


def _pdl_evidence(output_dir: str, listed: set[str]) -> _PdlEvidence:
    """The PDL launches among a run's kernel launch events.

    The events cover every launch, whatever the kernel filter. A direct launch
    carries `launch_attributes`, a graph node launch carries
    `captured_launch_attributes` when its captured launch is known, and every
    event of a CUTracer that records them carries `nonblocking_launches`.
    `listed` holds the kernels the run instrumented (`_listed_kernels`).
    """
    evidence = _PdlEvidence()
    for events in _kernel_launch_event_files(output_dir):
        evidence = evidence.merge(_process_pdl_evidence(events, listed))
    return evidence


def _pdl_skip_reason(evidence: _PdlEvidence) -> Optional[str]:
    """Why a mode that needs PDL launches does not apply: no launch that uses
    PDL can overlap the kernel before it. None when one can, or when the
    events cannot tell (there are none, or their CUTracer records no launch
    attributes)."""
    if evidence.testable or not evidence.attributed:
        return None
    if not evidence.direct and not evidence.graph:
        return "no launch in the control runs used programmatic dependent launch (PDL)"
    causes = []
    if evidence.graph:
        causes.append("PDL inside a CUDA graph does not overlap under CUTracer")
    if evidence.first:
        causes.append(
            "a kernel's first launch waits for the running kernels while "
            "CUTracer instruments it"
        )
    if evidence.unpaired:
        causes.append(
            "a launch that follows a CUDA graph launch, or no launch, on its "
            "stream has no direct launch to overlap"
        )
    return (
        "no launch that used programmatic dependent launch (PDL) can overlap "
        "the kernel before it: " + "; ".join(causes)
    )


def _pdl_hold_reason(evidence: _PdlEvidence) -> Optional[str]:
    """Why a mode that needs PDL launches applies but cannot test them: the
    kernel filter leaves a kernel every testable launch may read early from
    uninstrumented, so no delay can hold back all of its producers' stores.
    None otherwise."""
    if not evidence.testable or evidence.covered:
        return None
    return (
        "every launch that uses programmatic dependent launch (PDL) may read "
        "early from a kernel the kernel filter leaves uninstrumented (for "
        f"example {', '.join(evidence.uninstrumented)}), so no delay can hold "
        "back that kernel's stores; widen -k to include it"
    )


_BLOCKING_FALLBACK_MARKER = "--nonblocking-launches has no effect"


def _blocking_fallbacks(directory: str, *, before: set[str]) -> List[str]:
    """Lines of the main logs in `directory` that are not in `before` in which
    CUTracer reports draining launches on the launching thread despite
    --nonblocking-launches: that process's launches did not overlap."""
    lines: List[str] = []
    for name in sorted(main_logs(directory) - before):
        try:
            with open(os.path.join(directory, name), errors="replace") as log:
                lines.extend(
                    line.rstrip("\n")
                    for line in log
                    if _BLOCKING_FALLBACK_MARKER in line
                )
        except OSError:
            continue
    return lines


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


def _barrier_patterns(
    group: Sequence[BarrierSite],
    sites: Sequence[BarrierSite],
    mode_patterns: Optional[str],
) -> Tuple[str, ...]:
    """SASS substrings (opcode and barrier id) that select exactly the sites
    of `group`; () when they could select another instruction, so that the
    group needs a delay config that names its sites.

    An attempt's patterns replace the mode's, so they may only match
    instructions the control arm saw: the mode's patterns must match every
    instruction of each kind in `group`, and no pattern another site.
    """
    patterns = set()
    for site in group:
        operation = _barrier_operation(site.sass)
        if operation is None:
            continue
        opcode, kind, operand = operation
        if not any(
            pattern == "*" or pattern in f"BAR.{kind}"
            for pattern in _pattern_list(mode_patterns)
        ):
            return ()
        patterns.add(f"{opcode} {operand}")
    members = set(group)
    if any(
        pattern in site.sass
        for site in sites
        if site not in members
        for pattern in patterns
    ):
        return ()
    return tuple(sorted(patterns))


def _barrier_targets(
    config: StressConfig, sites: Sequence[BarrierSite]
) -> List[DelayTarget]:
    """One target per named barrier, in search order.

    Sites group by immediate barrier id across the instrumented kernels; a
    site whose barrier id is in a register is a target of its own. A target
    delays its barrier's syncs and reductions, and its arrives with
    `barrier_arrives`. Barriers that some warps only arrive at come first: a
    late sync there lets the arriving warps run a phase ahead. Register-id
    sites follow, then the barriers every participant syncs on, whose arrival
    order a delay on every warp keeps.
    """
    delayed = {"SYNC", "RED", "ARV"} if config.barrier_arrives else {"SYNC", "RED"}
    by_id: Dict[int, List[BarrierSite]] = {}
    arrived: set[int] = set()
    by_site: List[BarrierSite] = []
    for site in sites:
        operation = _barrier_operation(site.sass)
        if operation is None:
            continue
        _opcode, kind, operand = operation
        if not _IMMEDIATE.match(operand):
            if kind in delayed:
                by_site.append(site)
            continue
        barrier_id = int(operand, 0)
        if kind == "ARV":
            arrived.add(barrier_id)
        if kind in delayed:
            by_id.setdefault(barrier_id, []).append(site)
    mode_patterns = _delay_patterns(config)
    immediate = [
        NamedBarrier(
            barrier_id, tuple(group), _barrier_patterns(group, sites, mode_patterns)
        )
        for barrier_id, group in sorted(by_id.items())
    ]
    ordered = [
        *(barrier for barrier in immediate if barrier.barrier_id in arrived),
        *(NamedBarrier(None, (site,)) for site in by_site),
        *(barrier for barrier in immediate if barrier.barrier_id not in arrived),
    ]
    return [DelayTarget(barrier=barrier) for barrier in ordered]


def _isolate_targets(  # noqa: C901
    config: StressConfig,
    geometries: Sequence[_LaunchGeometry],
    barriers: Sequence[BarrierSite] = (),
) -> List[DelayTarget]:
    """The targets of a mode that isolates them, in search order.

    Named-barrier targets come from the `barriers` the control arm's delay
    dumps list (see `_barrier_targets`). Warp targets are grouped by
    instrumented kernel in first-launch order; within each kernel, cluster
    ranks run high to low and warps ascending. Higher ranks come first: in a
    multi-CTA kernel rank 0 usually leads, and a warp of a peer CTA whose work
    is predicated off is the one most easily left out of the release
    protocol. Targets are collected from every instrumented launch shape,
    without combining one kernel's warps with another's cluster ranks; a
    target named for several kernels is tried once, because one attempt
    delays that warp of every kernel it launches in.
    """
    if _isolate_by(config) == ISOLATE_BY_BARRIER:
        targets = _barrier_targets(config, barriers)
        if targets:
            return targets
        if config.baseline_runs <= 0:
            raise ValueError(
                "a named-barrier search discovers its targets from the control "
                "arm and needs baseline_runs >= 1"
            )
        raise ValueError(
            "the control arm recorded no named barrier (BAR.SYNC, BAR.ARV or "
            "BAR.RED) in the instrumented launches, so there is nothing to "
            "isolate; check the kernel filter and the delay patterns"
        )
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


def _target_window(
    config: StressConfig, targets: List[DelayTarget]
) -> List[DelayTarget]:
    """The rotating window of `targets` a search with `isolate_max_targets`
    tries; all of them without a cap or when they fit."""
    limit = config.isolate_max_targets
    if limit is None or len(targets) <= limit:
        return targets
    start = config.isolate_rotation * limit % len(targets)
    return [targets[(start + i) % len(targets)] for i in range(limit)]


def _delays_its_sites(config: StressConfig, target: DelayTarget) -> bool:
    """Whether every site of a named-barrier target is a point this search
    instruments and delays: CUTracer only looks up a replayed config's points
    among the instructions the delay patterns match."""
    assert target.barrier is not None
    patterns = _pattern_list(_delay_patterns(config))
    for site in target.barrier.sites:
        operation = _barrier_operation(site.sass)
        if operation is None or (operation[1] == "ARV" and not config.barrier_arrives):
            return False
        if not any(pattern == "*" or pattern in site.sass for pattern in patterns):
            return False
    return True


def _restored_targets(checkpoint: Sequence[Mapping[str, Any]]) -> List[DelayTarget]:
    """The targets a checkpoint recorded, in their original order.

    A discovered target list is a per-kernel union, so it is not always the
    Cartesian product an explicit `isolate_ranks` / `isolate_warps` pair
    reconstructs (and a launch without clusters contributes targets whose
    cluster rank is None, which no explicit rank names); named-barrier targets
    have no explicit form at all. A resume therefore restores the recorded
    list rather than rebuilding it; the entries are still checked, because the
    resumed schedule is located by counting the trials the checkpoint already
    ran against this list.
    """
    targets: List[DelayTarget] = []
    seen: set[DelayTarget] = set()
    for entry in checkpoint:
        try:
            target = DelayTarget.from_dict(entry)
        except ValueError as exc:
            raise ValueError(f"resumed isolate targets: {exc}") from exc
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
    warp: Optional[int],
    cluster_rank: Optional[int],
    label: str,
) -> Optional[str]:
    """Verify that a matching launch has CTA-local `warp` (any warp when None)
    and cluster rank.

    A target outside every launch makes the attempt inject nothing, so its
    "not reproduced" would be meaningless. Any eligible kernel may provide
    the matching launch. An explicit cluster rank additionally needs a
    multi-CTA cluster: cluster targeting delays no CTA of a one-CTA cluster,
    so rank 0 of such a launch is not a perturbed target.
    """
    kernel_checksums = {checksum for _name, checksum in kernel_identities}
    matched_launch = False
    matched_without_cluster = False
    multi_cta_without_warp = False
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
                    if block_warps is None or (
                        warp is not None and warp >= block_warps
                    ):
                        multi_cta_without_warp |= (
                            cluster_rank is not None and _cluster_size(event) > 1
                        )
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
    if multi_cta_without_warp:
        return f"{label} has no live warp in the matching multi-CTA kernel launch"
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
            "delay_mode": get_mode(config.strategy).delay_mode,
            "delay_enable_prob": config.enable_prob,
            "delay_warpgroup_id": warpgroup,
            "delay_patterns": _delay_patterns(config),
        }
    return {"delay_enable_prob": 1.0, **_target_fields(config, target)}


def _write_barrier_config(path: str, barrier: NamedBarrier, delay_ns: int) -> None:
    """A delay config that turns on `barrier`'s sites, and names no other
    point: replay mode leaves every point it does not list undelayed."""
    kernels: Dict[str, Dict[str, Any]] = {}
    timestamp = datetime.datetime.now().isoformat(timespec="milliseconds")
    for site in barrier.sites:
        kernel = kernels.setdefault(
            f"{site.kernel_name}_{site.kernel_checksum}",
            {
                "kernel_name": site.kernel_name,
                "kernel_checksum": site.kernel_checksum,
                "timestamp": timestamp,
                "instrumentation_points": {},
            },
        )
        kernel["instrumentation_points"][str(site.pc)] = {
            "pc": site.pc,
            "sass": site.sass,
            "delay": delay_ns,
            "on": True,
        }
    with open(path, "w") as output:
        json.dump({"version": "1.0", "delay_ns": delay_ns, "kernels": kernels}, output)


def _attempt_config_name(
    delay_ns: int, warpgroup: Optional[int], target: Optional[DelayTarget], attempt: int
) -> str:
    if target is None:
        return f"delay-{delay_ns}-wg-{warpgroup}-attempt-{attempt}.json"
    if target.barrier is not None:
        barrier = target.barrier
        if barrier.barrier_id is not None:
            key = f"{barrier.barrier_id:#x}"
        else:
            site = barrier.sites[0]
            key = f"{site.kernel_checksum[:16]}-pc-{site.pc:#x}"
        return f"delay-{delay_ns}-barrier-{key}-attempt-{attempt}.json"
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
    replays = _replays_a_config(target)
    logs_before = main_logs(instrumentation_output_dir)
    proc: Optional[subprocess.CompletedProcess[str]] = None
    timeout_error = ""
    timeout_output: tuple[str | bytes | None, ...] = ()
    try:
        if replays:
            assert target is not None and target.barrier is not None
            _write_barrier_config(config_path, target.barrier, delay_ns)
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
                # CUTracer cannot dump and replay in one run; a replayed
                # config already is the attempt's record.
                delay_dump_path=None if replays else config_path,
                delay_load_path=config_path if replays else None,
                # A fresh seed per attempt, even when the caller's environment
                # pins one: every process of the attempt shares it.
                delay_seed=secrets.randbits(64),
                kernel_timeout_s=config.kernel_timeout_s,
                no_data_timeout_s=_no_data_timeout_s(config),
                exit_quarantine_ns=config.exit_quarantine_ns,
                cwd=config.cwd,
                timeout=config.timeout,
                base_env=config.base_env,
                use_bundled_cuda_tools=config.use_bundled_cuda_tools,
                kernel_events=(
                    "nostack"
                    if _records_launches(config, warpgroup=warpgroup, target=target)
                    else None
                ),
                **_attempt_targeting(config, warpgroup=warpgroup, target=target),
                **_launch_options(config),
            ),
            runner=runner or run_process_group,
        )
    except subprocess.TimeoutExpired as exc:
        timeout_error = f"timed out: {exc}"
        timeout_output = (exc.stdout, exc.stderr)
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
    fallback = (
        _blocking_fallbacks(instrumentation_output_dir, before=logs_before)
        if get_mode(config.strategy).nonblocking_launches
        else []
    )
    hangs = 1 if watchdog else 0
    # A hang ended by CUTracer's watchdog is the symptom itself; the oracle's
    # exit code (or its absence after an outer timeout) then carries nothing.
    hung = bool(watchdog) and config.hang_is_bug
    late_writes = _late_writes(config, instrumentation_output_dir, logs_before)
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
        late_writes=late_writes,
    )
    fatal = exit_quarantine_failures(
        instrumentation_output_dir,
        before=logs_before,
        output=timeout_output if proc is None else (proc.stdout, proc.stderr),
    )
    if fatal:
        rejected = _preserve_rejected_config(config_path)
        return _AttemptResult(
            0,
            0,
            1,
            None,
            log + "\n--- native tool failure ---\n" + fatal[0],
            rejected_config_path=rejected,
            hangs=hangs,
        )
    if fallback:
        # The mode's delays hold stores back only while kernels overlap.
        _unlink(config_path)
        return _AttemptResult(
            0,
            0,
            1,
            None,
            log + f"\n--- error ---\nlaunches did not overlap: {fallback[0]}",
            hangs=hangs,
        )
    if returncode is None and not hung and not late_writes:
        _unlink(config_path)
        return _AttemptResult(0, 0, 0, None, log, timed_out=1, hangs=hangs)

    is_interesting = hung or bool(late_writes) or returncode == 0
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
                symptom=(
                    SYMPTOM_HANG
                    if hung
                    else SYMPTOM_FAILURE
                    if returncode == 0
                    else SYMPTOM_LATE_WRITE
                ),
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
        if _records_launches(config, warpgroup=warpgroup, target=target)
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
    """One control run: the oracle's exit code (None when the run timed out),
    whether CUTracer's watchdog ended a hung kernel in it, whether its exit
    quarantine saw a late write, whether CUTracer could not overlap its
    launches although the mode asked for it, and its PDL launches (recorded
    with `_BaselineResult.geometries`)."""

    returncode: Optional[int]
    hung: bool
    late_write: bool = False
    infra_error: bool = False
    blocking: bool = False
    pdl: _PdlEvidence = field(default_factory=_PdlEvidence)


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
    # Checksums of the launched kernels CUTracer instrumented, with or without
    # delay points (recorded with `geometries`): instrumented launches but no
    # geometry mean the delay patterns match nothing in those kernels.
    instrumented_kernels: set[str] = field(default_factory=set)
    # The named-barrier delay points of those launches (recorded with
    # `geometries`), from which named-barrier targets are discovered.
    barriers: List[BarrierSite] = field(default_factory=list)
    # The PDL launches of the control runs that completed under the oracle
    # contract: a run that stopped early may not have reached them.
    pdl: _PdlEvidence = field(default_factory=_PdlEvidence)


# How `_control_outcome` counts a control run.
_CONTROL_REPRODUCED = "reproduced"
_CONTROL_COMPLETED = "completed"
_CONTROL_TIMED_OUT = "timed_out"
_CONTROL_INFRA_ERROR = "infra_error"


def _control_outcome(config: StressConfig, run: _BaselineRun) -> str:
    """How `config`'s oracle contract and hang rule count one control run."""
    if run.blocking:
        # Its launches did not overlap, unlike those of the mode's attempts.
        return _CONTROL_INFRA_ERROR
    if run.infra_error:
        # CUTracer's exit quarantine failed in the run.
        return _CONTROL_INFRA_ERROR
    if (run.hung and config.hang_is_bug) or run.late_write:
        # The target hangs, or writes a finished CTA, without injected delay.
        return _CONTROL_REPRODUCED
    if run.returncode is None:
        return _CONTROL_TIMED_OUT
    if run.returncode == 0:
        return _CONTROL_REPRODUCED
    if (
        not config.not_interesting_exit_codes
        or run.returncode in config.not_interesting_exit_codes
    ):
        return _CONTROL_COMPLETED
    # Same rule as the search arm: an exit code the oracle contract does not
    # cover means the control run did not measure anything.
    return _CONTROL_INFRA_ERROR


def _baseline_from_runs(
    config: StressConfig,
    runs: Sequence[_BaselineRun],
    *,
    logs: List[str],
    geometries: List[_LaunchGeometry],
    requested: Optional[int] = None,
    instrumented_kernels: Optional[set[str]] = None,
    barriers: Sequence[BarrierSite] = (),
) -> _BaselineResult:
    """Count control runs under `config`'s oracle contract and hang rule."""
    outcomes = [_control_outcome(config, run) for run in runs]
    pdl = _PdlEvidence()
    for run, outcome in zip(runs, outcomes):
        if outcome in (_CONTROL_REPRODUCED, _CONTROL_COMPLETED):
            pdl = pdl.merge(run.pdl)
    reproduced = outcomes.count(_CONTROL_REPRODUCED)
    return _BaselineResult(
        requested=config.baseline_runs if requested is None else requested,
        completed=reproduced + outcomes.count(_CONTROL_COMPLETED),
        reproductions=reproduced,
        timed_out=outcomes.count(_CONTROL_TIMED_OUT),
        infra_errors=outcomes.count(_CONTROL_INFRA_ERROR),
        logs=logs,
        hangs=sum(1 for run in runs if run.hung),
        geometries=geometries,
        runs=list(runs),
        instrumented_kernels=set(instrumented_kernels or ()),
        barriers=list(barriers),
        pdl=pdl,
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
    ranks the instrumented launches have, and so does a mode that needs PDL
    launches or overlaps launches, to learn whether any launch uses PDL and to
    do the same host work per launch as its attempts; `capture_geometry` does
    the same for a control arm that several modes share.

    No run starts with less than `MIN_RUN_BUDGET_S` left before `deadline`;
    the runs that do not start leave the control arm incomplete.
    """
    runs: List[_BaselineRun] = []
    geometries: List[_LaunchGeometry] = []
    instrumented: set[str] = set()
    barriers: List[BarrierSite] = []
    logs: List[str] = []
    if config.baseline_runs <= 0:
        # A caller that opted out of the control arm must not pay for anything
        # here -- including reading `delay_ladder_ns[0]`.
        return _BaselineResult(0, 0, 0, 0, 0, logs)
    delay_ns = config.delay_ladder_ns[0]
    mode = get_mode(config.strategy)
    capture = (
        mode.isolates_targets or mode.needs_pdl or mode.nonblocking_launches
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
        timeout_output: tuple[str | bytes | None, ...] = ()
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
                    exit_quarantine_ns=config.exit_quarantine_ns,
                    cwd=config.cwd,
                    timeout=_run_timeout(config, remaining),
                    base_env=config.base_env,
                    use_bundled_cuda_tools=config.use_bundled_cuda_tools,
                    **_launch_options(config),
                ),
                runner=runner or run_process_group,
            )
        except subprocess.TimeoutExpired as exc:
            error = f"baseline timed out: {exc}"
            timeout_output = (exc.stdout, exc.stderr)
        fatal = exit_quarantine_failures(
            run_dir,
            before=logs_before,
            output=timeout_output if proc is None else (proc.stdout, proc.stderr),
        )
        if fatal:
            error += "\n" + fatal[0]
        watchdog = watchdog_terminations(run_dir, before=logs_before)
        late_writes = _late_writes(config, run_dir, logs_before)
        fallback = (
            _blocking_fallbacks(run_dir, before=logs_before)
            if mode.nonblocking_launches
            else []
        )
        if fallback:
            error += f": launches did not overlap: {fallback[0]}"
        pdl = _PdlEvidence()
        if dump_path is not None:
            geometries = _merge_geometries(
                geometries, _instrumented_geometries(run_dir, dump_path)
            )
            listed = _listed_kernels(dump_path)
            instrumented |= _launched_instrumented_kernels(run_dir, listed)
            pdl = _pdl_evidence(run_dir, listed)
            barriers = _merge_barrier_sites(
                barriers, _barrier_sites(run_dir, dump_path)
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
                late_writes=late_writes,
            )
        )
        runs.append(
            _BaselineRun(
                None if proc is None else proc.returncode,
                bool(watchdog),
                late_write=bool(late_writes),
                infra_error=bool(fatal),
                blocking=bool(fallback),
                pdl=pdl,
            )
        )
    return _baseline_from_runs(
        config,
        runs,
        logs=logs,
        geometries=geometries,
        instrumented_kernels=instrumented,
        barriers=barriers,
    )


@dataclass(frozen=True)
class _PlainRun:
    """The plain oracle's one run: its exit code (None when it timed out or
    did not start) and its log."""

    returncode: Optional[int]
    log: str


def _plain_outcome(config: StressConfig, run: Optional[_PlainRun]) -> Optional[str]:
    """How `config`'s oracle contract counts the plain run (a `_CONTROL_*`
    outcome); None without one. It runs without CUTracer, so no watchdog can
    end a hang in it."""
    if run is None:
        return None
    return _control_outcome(config, _BaselineRun(run.returncode, hung=False))


def _run_plain(
    config: StressConfig,
    *,
    runner: Optional[RunTarget],
    deadline: Optional[float] = None,
) -> Optional[_PlainRun]:
    """Run `config.plain_oracle_argv` once, without CUTracer; None when the
    config has no plain oracle. It does not start with less than
    `MIN_RUN_BUDGET_S` left before `deadline`, which leaves it without a
    sample."""
    if config.plain_oracle_argv is None:
        return None
    header = "=== plain oracle (no CUTracer) ==="
    remaining = _remaining_s(deadline)
    if remaining is not None and remaining < MIN_RUN_BUDGET_S:
        return _PlainRun(
            None, _run_log(header, error="not started: the time budget ran out")
        )
    env = dict(os.environ if config.base_env is None else config.base_env)
    # An injection inherited from the caller would instrument the run.
    env.pop("CUDA_INJECTION64_PATH", None)
    try:
        proc = (runner or run_process_group)(
            list(config.plain_oracle_argv),
            env=env,
            cwd=config.cwd,
            timeout=_run_timeout(config, remaining),
            capture_output=True,
            text=True,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return _PlainRun(None, _run_log(header, error=f"timed out: {exc}"))
    log = _run_log(
        header,
        returncode=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )
    return _PlainRun(proc.returncode, log)


def _attribution(
    *,
    baseline_requested: int,
    baseline_completed: int,
    baseline_reproductions: int,
    triggering: object,
    plain_outcome: Optional[str] = None,
) -> str:
    """Classify what a reproduction can be credited to."""
    if plain_outcome == _CONTROL_REPRODUCED:
        # The program fails without CUTracer at all.
        return ATTRIBUTION_NATURAL
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
    if baseline_completed < baseline_requested or plain_outcome in (
        _CONTROL_TIMED_OUT,
        _CONTROL_INFRA_ERROR,
    ):
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


def _plain_failure(outcome: Optional[str], returncode: Optional[int]) -> Optional[str]:
    """Why the plain oracle's run produced no sample; None when it did or there
    was none."""
    if outcome == _CONTROL_TIMED_OUT:
        return "the plain oracle timed out or did not start"
    if outcome == _CONTROL_INFRA_ERROR:
        return (
            f"the plain oracle exited with {returncode}, which the oracle "
            "contract does not cover"
        )
    return None


def _unattributed_reason(result: StressResult) -> str:
    if result.plain_outcome == _CONTROL_REPRODUCED:
        return "the plain oracle reproduced the bug without CUTracer"
    if result.baseline_reproductions:
        return (
            f"the control arm reproduced {result.baseline_reproductions} of "
            f"{result.baseline_completed} run(s) with injection disabled"
        )
    if result.unattributed_reproductions:
        return (
            f"{result.unattributed_reproductions} attempt(s) reported the bug "
            "while injecting no delay"
        )
    if result.attribution == ATTRIBUTION_NO_BASELINE:
        return "reproduced, but no control arm ran"
    plain_failure = _plain_failure(result.plain_outcome, result.plain_returncode)
    if plain_failure is not None and (
        result.baseline_completed >= result.baseline_requested
    ):
        return f"reproduced, but {plain_failure}"
    return (
        f"reproduced, but only {result.baseline_completed} of "
        f"{result.baseline_requested} control run(s) produced a sample"
    )


def _incomplete_reasons(result: StressResult) -> List[str]:
    reasons: List[str] = []
    plain_failure = _plain_failure(result.plain_outcome, result.plain_returncode)
    if plain_failure is not None:
        reasons.append(plain_failure)
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

    - finding: an attempt reproduced the bug, and neither the plain oracle,
      the control arm nor an attempt that injected nothing did, so the
      injected delay is what made the difference.
    - suspect: the bug reproduced, but not attributably to an injected delay:
      it also reproduced without CUTracer (the plain oracle), with injection
      disabled, or in an attempt that injected nothing, or without a complete
      control arm.
    - clean: every planned attempt ran and measured something (no infra error,
      timeout or unexplained hang), after a complete control arm and plain run
      if they were requested, and nothing reproduced. It covers only the
      attempts made.
    - incomplete: anything else.
    """
    triggering = result.triggering
    if (
        triggering is not None
        and result.attribution == ATTRIBUTION_ATTRIBUTED
        and not result.unattributed_reproductions
    ):
        return Verdict(
            VERDICT_FINDING,
            [f"reproduced ({triggering.symptom}) at delay_ns={triggering.delay_ns}"],
        )
    if (
        triggering is not None
        or result.baseline_reproductions
        or result.unattributed_reproductions
        or result.plain_outcome == _CONTROL_REPRODUCED
    ):
        return Verdict(VERDICT_SUSPECT, [_unattributed_reason(result)])
    reasons = _incomplete_reasons(result)
    if reasons:
        return Verdict(VERDICT_INCOMPLETE, reasons)
    if result.targets_total is not None:
        return Verdict(
            VERDICT_CLEAN,
            [
                f"{result.strategy} covered {len(result.targets)} of "
                f"{result.targets_total} "
                "targets (a rotating window)"
            ],
        )
    return Verdict(VERDICT_CLEAN)


def run_stress(  # noqa: C901
    config: StressConfig,
    *,
    cutracer_so: Optional[str] = None,
    runner: Optional[RunTarget] = None,
    baseline: Optional[_BaselineResult] = None,
    deadline: Optional[float] = None,
    plain: Optional[_PlainRun] = None,
) -> StressResult:
    """Run the planned attempts (see `_trial_schedule`) until the oracle reproduces.

    A "reproduction" requires an interesting oracle exit (0), a dumped delay
    config (otherwise there is nothing to replay/reduce, so the attempt is
    counted as an infra error), AND at least one enabled injection point in that
    config (otherwise the attempt injected nothing and cannot have caused the
    failure -- counted as an unattributed reproduction).

    `baseline` is a control arm that already ran with this search's
    instrumentation (see `cutracer.stress.plan`); it is counted under this
    search's oracle contract instead of running the control arm again. `plain`
    is likewise the plain oracle's run when the caller already ran it; else a
    search that does not resume runs it before its control arm. No
    attempt starts with less than `MIN_RUN_BUDGET_S` left before `deadline`
    (a `time.monotonic()` value); the result then says `budget_exhausted`.

    A mode that needs PDL launches raises ValueError when its own control arm
    shows that no launch uses PDL, or that none can be tested under the kernel
    filter; a plan holds such a mode back before it gets here.
    """
    os.makedirs(config.output_dir, exist_ok=True)
    resume = config.resume_result
    completed = 0 if resume is None else resume.completed_trials
    reproductions = 0 if resume is None else resume.reproductions
    infra_errors = 0 if resume is None else resume.infra_errors
    timed_out = 0 if resume is None else resume.timed_out
    hangs = 0 if resume is None else resume.hangs
    unattributed = 0 if resume is None else resume.unattributed_reproductions
    checks_pdl = (
        baseline is None and resume is None and get_mode(config.strategy).needs_pdl
    )
    # The plain run's log goes into this search's log only when it ran here.
    plain_logs: List[str] = []
    if baseline is not None:
        if resume is not None:
            raise ValueError("a shared control arm cannot resume a search")
        baseline = _baseline_from_runs(
            config,
            baseline.runs,
            logs=[],
            geometries=baseline.geometries,
            requested=baseline.requested,
            instrumented_kernels=baseline.instrumented_kernels,
            barriers=baseline.barriers,
        )
    elif resume is None:
        if plain is None:
            plain = _run_plain(config, runner=runner, deadline=deadline)
            plain_logs = [] if plain is None else [plain.log]
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
    if resume is None:
        plain_requested = plain is not None
        plain_outcome = _plain_outcome(config, plain)
        plain_returncode = None if plain is None else plain.returncode
    else:
        plain_requested = resume.plain_requested
        plain_outcome = resume.plain_outcome
        plain_returncode = resume.plain_returncode
    log_path: Optional[str] = None
    log_size_bytes = 0
    if resume is None:
        # Written before target discovery can raise: the control arm's logs
        # are what explain a launch that has nothing to isolate.
        log_path, log_size_bytes = _prepare_stress_log(config.output_dir, None)
        if plain_logs or baseline.logs:
            log_path, log_size_bytes = _write_stress_log(
                config.output_dir, "\n".join(plain_logs + baseline.logs), append=False
            )
    elif (
        resume.isolate_by is not None and resume.isolate_by != _isolate_by(config)
    ) or (
        resume.barrier_arrives is not None
        and resume.barrier_arrives != config.barrier_arrives
    ):
        raise ValueError(
            "resuming a stress search needs the original isolate_by and "
            "barrier_arrives; its recorded targets were chosen under them"
        )
    rejected_config_paths: List[str] = _bounded_rejected_config_paths(
        [] if resume is None else list(resume.rejected_config_paths),
        owned_root=config.output_dir,
    )
    triggering = None if resume is None else resume.triggering
    stop = False
    targets: List[DelayTarget] = []
    targets_total: Optional[int] = None if resume is None else resume.targets_total
    restored_targets = False
    if get_mode(config.strategy).isolates_targets:
        if resume is not None and resume.targets:
            # The checkpoint's recorded targets are the original schedule;
            # see `_restored_targets` for why they cannot be rebuilt from
            # explicit ranks and warps.
            targets = _restored_targets(resume.targets)
            restored_targets = True
            barrier_search = _isolate_by(config) == ISOLATE_BY_BARRIER
            if any(
                (target.barrier is not None) != barrier_search for target in targets
            ):
                raise ValueError(
                    f"resumed isolate targets are not {_isolate_by(config)} targets"
                )
            if barrier_search and not all(
                _delays_its_sites(config, target) for target in targets
            ):
                raise ValueError(
                    "resumed named-barrier targets name sites that this search's "
                    "delay patterns and barrier_arrives do not delay"
                )
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
                    "resuming an isolate search needs the targets its checkpoint "
                    "recorded"
                    + (
                        ""
                        if _isolate_by(config) == ISOLATE_BY_BARRIER
                        else ", or explicit isolate_ranks and isolate_warps"
                    )
                )
            discovered = _isolate_targets(
                config, baseline.geometries, baseline.barriers
            )
            targets = _target_window(config, discovered)
            if len(targets) < len(discovered):
                targets_total = len(discovered)
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
    if resume is not None:
        log_path, log_size_bytes = _prepare_stress_log(config.output_dir, resume)
    if config.schedule_callback is not None:
        config.schedule_callback(config.strategy, len(targets), planned_trials)
    pdl_reason = (
        _pdl_skip_reason(baseline.pdl) or _pdl_hold_reason(baseline.pdl)
        if checks_pdl
        else None
    )
    if pdl_reason is not None:
        raise ValueError(
            f"a {config.strategy} search cannot run: {pdl_reason} (the control "
            "runs are in stress.log)"
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
                plain_outcome=plain_outcome,
            ),
            plain_requested=plain_requested,
            plain_outcome=plain_outcome,
            plain_returncode=plain_returncode,
            log_size_bytes=log_size_bytes if log_path is not None else None,
            rejected_config_paths=list(rejected_config_paths),
            strategy=config.strategy,
            targets=[target.to_dict() for target in targets],
            targets_total=targets_total,
            delay_ladder_ns=list(config.delay_ladder_ns),
            attempts_per_delay=config.attempts_per_delay,
            planned_trials=planned_trials,
            budget_exhausted=budget_exhausted,
            isolate_by=_isolate_by(config),
            barrier_arrives=config.barrier_arrives,
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
            plain_outcome=plain_outcome,
        ),
        plain_requested=plain_requested,
        plain_outcome=plain_outcome,
        plain_returncode=plain_returncode,
        log_size_bytes=log_size_bytes if log_path is not None else None,
        rejected_config_paths=list(rejected_config_paths),
        strategy=config.strategy,
        targets=[target.to_dict() for target in targets],
        targets_total=targets_total,
        delay_ladder_ns=list(config.delay_ladder_ns),
        attempts_per_delay=config.attempts_per_delay,
        planned_trials=planned_trials,
        budget_exhausted=budget_exhausted,
        isolate_by=_isolate_by(config),
        barrier_arrives=config.barrier_arrives,
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
