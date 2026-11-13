# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for cutracer.watchdog."""

import os
import tempfile
import unittest

from cutracer.watchdog import main_logs, watchdog_terminations

KERNEL_TIMEOUT = (
    "Kernel timeout: running for 61 seconds (limit: 60 seconds); sending SIGTERM. "
    "kernel=my_kernel"
)
NO_DATA = (
    "No-data hang detected: kernel launched 15 seconds ago but never produced "
    "any trace data (limit: 15 seconds); sending SIGTERM."
)
DEADLOCK = (
    "Deadlock sustained for 3 checks at least 1000 ms apart over 2000 ms (minimum "
    "stall 0 ms): launch_id=7, no active warp progressing (looping=12, barrier=4); "
    "sending SIGTERM."
)
POSSIBLE_HANG = (
    "Possible kernel hang: launch_id=7 — state(looping=12, barrier=4, "
    "progressing=0) for 1 seconds."
)


def _write(directory: str, name: str, *lines: str) -> None:
    with open(os.path.join(directory, name), "w") as log:
        log.write("".join(line + "\n" for line in lines))


class WatchdogTest(unittest.TestCase):
    def test_main_logs_lists_only_cutracer_main_logs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _write(directory, "cutracer_main_20261104_000000_1.log", "started")
            _write(directory, "cutracer_kernel_events_20261104_000000_1.ndjson")
            _write(directory, "run.log")
            self.assertEqual(
                main_logs(directory), {"cutracer_main_20261104_000000_1.log"}
            )

    def test_missing_directory_has_no_logs(self) -> None:
        self.assertEqual(main_logs("/nonexistent/cutracer"), set())

    def test_reads_watchdog_lines_of_new_logs_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _write(directory, "cutracer_main_old.log", KERNEL_TIMEOUT)
            before = main_logs(directory)
            _write(directory, "cutracer_main_rank0.log", "started", KERNEL_TIMEOUT)
            _write(directory, "cutracer_main_rank1.log", NO_DATA)
            _write(directory, "cutracer_main_rank2.log", "started", "finished")

            lines = watchdog_terminations(directory, before=before)

        self.assertEqual(lines, [KERNEL_TIMEOUT, NO_DATA])

    def test_clean_run_has_no_watchdog_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _write(directory, "cutracer_main_rank0.log", "started", "finished")
            self.assertEqual(watchdog_terminations(directory, before=set()), [])

    def test_deadlock_check_termination_counts_but_a_warning_does_not(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _write(directory, "cutracer_main_rank0.log", POSSIBLE_HANG, DEADLOCK)
            self.assertEqual(watchdog_terminations(directory, before=set()), [DEADLOCK])
