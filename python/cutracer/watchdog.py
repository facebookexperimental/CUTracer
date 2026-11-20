# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Recognize CUTracer watchdog and exit-quarantine evidence in fresh main logs."""

from __future__ import annotations

import os

WATCHDOG_MARKERS: tuple[str, ...] = (
    "Kernel timeout: running for",
    "No-data hang detected",
    "Deadlock sustained for ",
)
_MAIN_LOG_PREFIX = "cutracer_main_"
_MAIN_LOG_SUFFIX = ".log"
EXIT_QUARANTINE_MARKER = "Exit quarantine: shared memory of CTA"
EXIT_QUARANTINE_FATAL = "FATAL: exit quarantine:"


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


def _main_log_lines(
    directory: str, *, before: set[str], markers: tuple[str, ...]
) -> list[str]:
    lines: list[str] = []
    for name in sorted(main_logs(directory) - before):
        try:
            with open(os.path.join(directory, name), errors="replace") as log:
                lines.extend(
                    line.rstrip("\n")
                    for line in log
                    if any(marker in line for marker in markers)
                )
        except OSError:
            continue
    return lines


def watchdog_terminations(directory: str, *, before: set[str]) -> list[str]:
    """Watchdog lines from processes created after the `main_logs` snapshot."""
    return _main_log_lines(directory, before=before, markers=WATCHDOG_MARKERS)


def exit_quarantine_failures(
    directory: str,
    *,
    before: set[str],
    output: tuple[str | bytes | None, ...] = (),
) -> list[str]:
    """Fatal tool diagnostics, including output emitted before log initialization."""
    lines = _main_log_lines(directory, before=before, markers=(EXIT_QUARANTINE_FATAL,))
    for chunk in output:
        text = (
            chunk.decode(errors="replace") if isinstance(chunk, bytes) else chunk or ""
        )
        lines.extend(
            line for line in text.splitlines() if EXIT_QUARANTINE_FATAL in line
        )
    return list(dict.fromkeys(lines))


def exit_quarantine_changes(directory: str, *, before: set[str]) -> list[str]:
    """Changed-word reports from processes created after the main-log snapshot."""
    return _main_log_lines(directory, before=before, markers=(EXIT_QUARANTINE_MARKER,))
