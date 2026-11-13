# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Recognize runs that CUTracer's own watchdog ended.

CUTracer ends a hung instrumented kernel itself: the kernel timeout
(`CUTRACER_KERNEL_TIMEOUT_S`), the no-data watchdog
(`CUTRACER_NO_DATA_TIMEOUT_S`) and the online deadlock check
(`-a deadlock_detection`) write a line to the process's main log and send the
process SIGTERM. The target's exit code then says little (the process died
of a signal, or a parent reported a failed worker), so a caller that needs to
know whether a run hung reads the main logs instead.
"""

from __future__ import annotations

import os

WATCHDOG_MARKERS: tuple[str, ...] = (
    "Kernel timeout: running for",
    "No-data hang detected",
    "Deadlock sustained for ",
)
_MAIN_LOG_PREFIX = "cutracer_main_"
_MAIN_LOG_SUFFIX = ".log"


def main_logs(directory: str) -> set[str]:
    """Names of the CUTracer main logs (one per process) in `directory`."""
    try:
        names = os.listdir(directory)
    except OSError:
        return set()
    return {
        name
        for name in names
        if name.startswith(_MAIN_LOG_PREFIX) and name.endswith(_MAIN_LOG_SUFFIX)
    }


def watchdog_terminations(directory: str, *, before: set[str]) -> list[str]:
    """Watchdog lines in the main logs of `directory` that are not in `before`.

    Take `before` with `main_logs` just before starting the target, so only the
    processes of that run count.
    """
    lines: list[str] = []
    for name in sorted(main_logs(directory) - before):
        try:
            with open(os.path.join(directory, name), errors="replace") as log:
                lines.extend(
                    line.rstrip("\n")
                    for line in log
                    if any(marker in line for marker in WATCHDOG_MARKERS)
                )
        except OSError:
            continue
    return lines
