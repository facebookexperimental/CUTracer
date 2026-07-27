# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
CUTracer Reduce Module.

Provides tools for reducing delay injection configurations to find minimal
sets that trigger data races.
"""

from cutracer.reduce.config_mutator import DelayConfigMutator
from cutracer.reduce.reduce import (
    ConfigDoesNotTriggerError,
    reduce_bisect,
    reduce_delay_points,
    ReplayConfig,
    ReplayOutcome,
    ReplayResult,
    run_replay,
)

__all__ = [
    "ConfigDoesNotTriggerError",
    "DelayConfigMutator",
    "ReplayConfig",
    "ReplayOutcome",
    "ReplayResult",
    "reduce_bisect",
    "reduce_delay_points",
    "run_replay",
]
