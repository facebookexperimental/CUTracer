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


RANDOM = StressMode(
    name="random",
    summary="random per-thread delays at every candidate point, all warps at "
    "once or one warpgroup at a time",
    delay_ladder_ns=(1000, 5000, 10000, 50000, 100000),
    attempts_per_delay=3,
    kernel_timeout_s=0,
    hang_is_bug=False,
    delay_patterns=None,
    isolates_targets=False,
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


def get_modes() -> dict[str, StressMode]:
    """Every mode by name, in the order a multi-mode search runs them."""
    return {mode.name: mode for mode in (RANDOM, ISOLATE)}


def get_mode(name: str) -> StressMode:
    modes = get_modes()
    if name not in modes:
        raise ValueError(f"unknown stress mode {name!r}; known: {', '.join(modes)}")
    return modes[name]
