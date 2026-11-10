# Copyright (c) Meta Platforms, Inc. and affiliates.

"""The delay modes of ``cutracer stress`` and their defaults, in one table."""

from __future__ import annotations

from dataclasses import dataclass


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
    # SASS substrings to delay at; None = CUTracer's built-in patterns.
    delay_patterns: str | None
    # True: each attempt stalls one (cluster rank, CTA-local warp) target with a
    # fixed delay, and the targets come from the control arm's launches.
    # False: every warp (or one warpgroup) gets random per-thread delays.
    isolates_targets: bool
    # Probability that each delay point is on in an attempt (ignored by
    # isolate, which turns every point on).
    enable_prob: float = 1.0


# The defaults below come from three known races reproduced on GB200
# (2026-11-06): two softmax running-max races, where a warp reads the running
# max from shared memory after a faster warp's next update (fixed by
# D129633882 and D129803385), and a DistMoE consumer that never releases its
# ring slot (T290953007). Instrumentation alone slows these kernels 40 to 200
# times, so useful delays are tens of microseconds and up.
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
    isolates_targets=False,
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
    isolates_targets=False,
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
    isolates_targets=True,
)


# TODO: Add modes as new races need them. The patterns above cover what the
# three races needed: shared-memory loads, CUTracer's built-in points and
# mbarrier waits. A race between instructions none of them delays (for example
# TMEM accesses, named barriers or distributed shared memory) needs a new mode,
# with its ladder and attempts calibrated on a race it reproduces; widening an
# existing mode's patterns would change what that mode was calibrated for. The
# essential points of a confirmed finding (`--confirm`) name the instructions a
# race needed delayed.
def get_modes() -> dict[str, StressMode]:
    """Every mode by name, in the order a multi-mode search runs them."""
    return {mode.name: mode for mode in (LDS, RANDOM, ISOLATE)}


def get_mode(name: str) -> StressMode:
    modes = get_modes()
    if name not in modes:
        raise ValueError(f"unknown stress mode {name!r}; known: {', '.join(modes)}")
    return modes[name]
