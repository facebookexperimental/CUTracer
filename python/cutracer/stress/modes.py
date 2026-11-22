# Copyright (c) Meta Platforms, Inc. and affiliates.

"""The delay modes of ``cutracer stress`` and their defaults, in one table."""

from __future__ import annotations

from dataclasses import dataclass

# What one attempt of a mode that isolates targets stalls, with a fixed delay.
# One CTA-local warp of one cluster rank, before the mode's delay points.
ISOLATE_BY_WARP = "warp"
# Every warp, before the points of one named barrier only: every site of one
# immediate barrier id, or one site of a barrier whose id is in a register.
ISOLATE_BY_BARRIER = "barrier"
ISOLATE_KINDS = (ISOLATE_BY_WARP, ISOLATE_BY_BARRIER)


@dataclass(frozen=True)
class StressMode:
    """How one delay mode places its delays, and the defaults it searches with."""

    name: str
    summary: str
    delay_ladder_ns: tuple[int, ...]
    attempts_per_delay: int
    # CUTracer's per-kernel timeout in every run (0 = off: the 15 s no-data
    # watchdog then ends any kernel that runs that long).
    kernel_timeout_s: int
    # Whether a hang that CUTracer's watchdog ends counts as a reproduction.
    hang_is_bug: bool
    # SASS substrings to delay at; None = CUTracer's built-in patterns. A mode
    # that isolates targets discovers them at these points in its control arm.
    delay_patterns: str | None
    # One of ISOLATE_KINDS: each attempt stalls one target, and the targets
    # come from the control arm. None: every warp (or one warpgroup) is
    # delayed at the mode's delay points.
    isolate_by: str | None
    # Probability that each delay point is on in an attempt (ignored by
    # modes that isolate targets, which turn every point of a target on).
    enable_prob: float = 1.0
    # CUTracer's exit quarantine in every run (ns, 0 = off): a CTA whose shared
    # memory changes after its last thread exited counts as a reproduction.
    exit_quarantine_ns: int = 0
    # Whether `--strategy all` (the default plan) runs the mode.
    in_default_plan: bool = True
    # For the "Mode defaults" of `cutracer stress --help`: which points an
    # attempt turns on, when "every point of the target" or the enable
    # probability does not say it, and anything else the row should say.
    points: str | None = None
    notes: str = ""

    @property
    def isolates_targets(self) -> bool:
        return self.isolate_by is not None


# The defaults of lds, random and isolate come from three known races
# reproduced on GB200 (2026-11-06): two softmax running-max races, where a warp
# reads the running max from shared memory after a faster warp's next update,
# and a DistMoE consumer that never releases its ring slot. Instrumentation
# alone slows these kernels 40 to 200 times, so useful delays are tens of
# microseconds and up.
#
# Delaying only shared-memory loads separates one reader from the writers:
# both running-max races reproduce at 100 us (6 of 6 attempts).
LDS = StressMode(
    name="lds",
    summary="random per-thread delays before shared-memory loads only, every "
    "point on: a late reader of a value other warps keep updating",
    delay_ladder_ns=(100_000, 10_000),
    attempts_per_delay=2,
    kernel_timeout_s=0,
    hang_is_bug=False,
    delay_patterns="LDS",
    isolate_by=None,
)

# CUTracer's built-in point set (memory accesses and synchronization) delays
# almost every warp at almost every step when every point is on, which keeps
# their relative order: 0 of 18 attempts on the two running-max races. With a
# tenth of the points on, it still reaches points other than shared loads
# (3 of 6 on one of them).
RANDOM = StressMode(
    name="random",
    summary="random per-thread delays at a random tenth of CUTracer's built-in "
    "points (memory accesses and synchronization), all warps at once or one "
    "warpgroup at a time",
    delay_ladder_ns=(10_000, 100_000),
    attempts_per_delay=2,
    kernel_timeout_s=0,
    hang_is_bug=False,
    delay_patterns=None,
    isolate_by=None,
    enable_prob=0.1,
)

# A single 1 ms rung lets a producer lap an unreleasing consumer by many phases,
# every hang is a finding, and the kernel timeout ends it. Mbarrier waits are
# delayed because a consumer that never releases its slot misbehaves when it
# reaches its wait late (the parity it waits for may then name a later phase),
# while arriving late at a load only makes it read a newer value.
ISOLATE = StressMode(
    name="isolate",
    summary="stall one (cluster rank, CTA-local warp) per attempt with a fixed "
    "delay before each mbarrier wait while every other warp runs at full speed; "
    "finds a warp the protocol does not wait for, such as a consumer that never "
    "releases its ring slot",
    delay_ladder_ns=(1_000_000,),
    attempts_per_delay=1,
    kernel_timeout_s=60,
    hang_is_bug=True,
    delay_patterns="SYNCS.PHASECHK",
    isolate_by=ISOLATE_BY_WARP,
)


# A named-barrier token imbalance (an FP8-KV decode attention kernel): one warp
# arrives at a barrier twice before another reaches its first sync, so that
# sync completes on the wrong arrivals and a later one never completes. It
# reproduces only with a fixed delay on every warp before one barrier's syncs
# (GB200: 30 of 30 hangs before the fix at 1, 10 and 100 us alike, 0 of 30
# after it). Delaying the barrier's arrive too slows the arriving warp just as
# much and removes the hang (0 of 20), so arrives are not delayed; `BAR.ARV` is
# still matched, to put the barriers some warps only arrive at first. The
# delays also slow a correct launch by the delay times the delayed syncs it
# runs in sequence, which one 10 us rung keeps far from the kernel timeout
# (about 6,000,000 of them reach 60 s).
BAR = StressMode(
    name="bar",
    summary="a fixed delay on every warp before the syncs of one named barrier "
    "per attempt (every site of an immediate barrier id, or one site whose "
    "barrier id is in a register); finds warps that arrive a phase ahead of a "
    "late sync",
    delay_ladder_ns=(10_000,),
    attempts_per_delay=1,
    kernel_timeout_s=60,
    hang_is_bug=True,
    delay_patterns="BAR.SYNC,BAR.ARV,BAR.RED",
    isolate_by=ISOLATE_BY_BARRIER,
    points="every sync and reduction of the target (arrives only order the targets)",
)


# TODO: Add modes as new races need them. Each mode above delays what the race
# it was calibrated on needed. A race between instructions none of them delays
# (for example distributed shared memory) needs a new mode, with its ladder and
# attempts calibrated on a race it reproduces; widening an existing mode's
# patterns would change what that mode was calibrated for. The essential points
# of a confirmed finding (`--confirm`) name the instructions a race needed
# delayed.
def get_modes() -> dict[str, StressMode]:
    """Every mode by name, in the order a multi-mode search runs them (the
    default plan runs those `in_default_plan`)."""
    return {mode.name: mode for mode in (LDS, RANDOM, ISOLATE, BAR)}


def get_mode(name: str) -> StressMode:
    modes = get_modes()
    if name not in modes:
        raise ValueError(f"unknown stress mode {name!r}; known: {', '.join(modes)}")
    return modes[name]
