# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
E2E test: CUTracer random delay injection for data race detection.

Uses the cutracer runner API (same as ``cutracer trace`` CLI) to run the
buggy Hopper GEMM kernels with random delay injection and verify that
data races are reliably detected.

Equivalent CLI usage::

    buck2 run fbcode//triton/tools/CUTracer:cutracer -- trace \\
        -i random_delay -a random_delay --delay-ns 10000 \\
        -k matmul_kernel_bug1_late_barrier_a \\
        -- python hopper-gemm-ws_data_race_test.py --bug 1 --iters 10
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import pkg_resources
from cutracer.runner import _build_cutracer_env, resolve_cutracer_so


class TestRandomDelayDataRace(unittest.TestCase):
    """E2E: random delay injection exposes data races in Hopper GEMM."""

    @classmethod
    def setUpClass(cls):
        # Resolve cutracer.so via the cutracer runner API (same as CLI)
        cls.cutracer_so = resolve_cutracer_so()

        # Extract the data race test script from bundled resources
        script_content = pkg_resources.resource_string(
            "datarace_test_resources",
            "hopper_gemm_ws_data_race_test.py",
        ).decode()
        tmp = tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False, prefix="datarace_"
        )
        tmp.write(script_content)
        tmp.flush()
        tmp.close()
        cls.script_path = tmp.name

        # Temp directory for CUTracer trace output (avoid polluting working dir)
        cls.trace_dir = tempfile.mkdtemp(prefix="cutracer_traces_")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "script_path") and os.path.exists(cls.script_path):
            os.unlink(cls.script_path)
        if hasattr(cls, "trace_dir") and os.path.exists(cls.trace_dir):
            shutil.rmtree(cls.trace_dir, ignore_errors=True)

    def _run_with_random_delay(self, bug_num, kernel_filter, num_iters=10):
        """Run data race test with CUTracer random delay via runner API.

        This is equivalent to::

            cutracer trace -i random_delay -a random_delay \\
                --delay-ns 10000 -k <kernel_filter> \\
                -- python <script> --bug <N> --iters <M>
        """
        # Build CUTracer env using the runner API (no manual env vars)
        env = _build_cutracer_env(
            cutracer_so=self.cutracer_so,
            instrument="random_delay",
            analysis="random_delay",
            kernel_filters=kernel_filter,
            delay_ns=10000,
            instr_categories=None,
            trace_format=None,
            output_dir=self.trace_dir,
            verbose=None,
            zstd_level=None,
        )

        # Run the data race test as a subprocess (same as cutracer trace does)
        result = subprocess.run(
            [
                sys.executable,
                self.script_path,
                "--bug",
                str(bug_num),
                "--iters",
                str(num_iters),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        return result

    def _run_without_cutracer(self, bug_num, num_iters=10, spin_count=None):
        """Run data race test WITHOUT CUTracer (control case)."""
        cmd = [
            sys.executable,
            self.script_path,
            "--bug",
            str(bug_num),
            "--iters",
            str(num_iters),
        ]
        if spin_count is not None:
            cmd.extend(["--spin-count", str(spin_count)])
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        return result

    @staticmethod
    def _parse_failure_count(output):
        """Parse 'Result: X/Y failures' from test output."""
        match = re.search(r"Result:\s*(\d+)/(\d+)\s*failures", output)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None, None

    def test_bug1_late_barrier_a_detected(self):
        """Random delay exposes bug1: late barrier_wait for A.

        CUTracer instruments each kernel once per process with a fixed random
        delay configuration.  All iterations within a single subprocess share
        the same delay pattern, so multiple iterations are redundant.  We use
        1 iteration per attempt and retry with independent subprocess
        invocations (each getting a fresh random config) to handle the rare
        case where the config doesn't trigger the race.
        """
        max_attempts = 10
        detected = 0
        last_result = None
        for attempt in range(max_attempts):
            result = self._run_with_random_delay(
                bug_num=1,
                kernel_filter="matmul_kernel_bug1_late_barrier_a",
                num_iters=1,
            )
            failures, total = self._parse_failure_count(result.stdout)
            self.assertIsNotNone(
                failures,
                f"Could not parse failure count from output (attempt {attempt + 1}).\n"
                f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}",
            )
            last_result = result
            if failures > 0:
                detected += 1
        self.assertGreaterEqual(
            detected,
            max_attempts // 2,
            f"Expected >= 50% race detection rate across {max_attempts} independent "
            f"random delay configurations, but only {detected}/{max_attempts} detected. "
            f"Random delay should expose the late-barrier-A data race.\n"
            f"STDOUT: {last_result.stdout}\nSTDERR: {last_result.stderr}",
        )

    def test_bug2_missing_barrier_b_detected(self):
        """Random delay exposes bug2: missing barrier_wait for B.

        CUTracer instruments each kernel once per process with a fixed random
        delay configuration (which instructions get delays is decided at
        instrumentation time).  All iterations within a single subprocess
        share the same delay pattern, so multiple iterations are redundant.
        We use 1 iteration per attempt and retry with independent subprocess
        invocations (each getting a fresh random config) to make the test
        robust against unlucky configs.
        """
        max_attempts = 10
        detected = 0
        last_result = None
        for attempt in range(max_attempts):
            result = self._run_with_random_delay(
                bug_num=2,
                kernel_filter="matmul_kernel_bug2_missing_barrier_b",
                num_iters=1,
            )
            failures, total = self._parse_failure_count(result.stdout)
            self.assertIsNotNone(
                failures,
                f"Could not parse failure count from output (attempt {attempt + 1}).\n"
                f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}",
            )
            last_result = result
            if failures > 0:
                detected += 1
        self.assertGreaterEqual(
            detected,
            max_attempts // 2,
            f"Expected >= 50% race detection rate across {max_attempts} independent "
            f"random delay configurations, but only {detected}/{max_attempts} detected. "
            f"Random delay should expose the missing-barrier-B data race.\n"
            f"STDOUT: {last_result.stdout}\nSTDERR: {last_result.stderr}",
        )

    def test_bug1_passes_without_random_delay(self):
        """Control: bug1 should mostly pass WITHOUT random delay (high spin count)."""
        result = self._run_without_cutracer(bug_num=1, spin_count=5000)
        failures, total = self._parse_failure_count(result.stdout)
        self.assertIsNotNone(
            failures,
            f"Could not parse failure count from output.\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}",
        )
        self.assertLess(
            failures,
            total // 2,
            f"Expected < 50% failure rate WITHOUT random delay, "
            f"got {failures}/{total}. "
            f"Bug1 should mostly pass due to timing luck without CUTracer.",
        )

    def test_bug2_passes_without_random_delay(self):
        """Control: bug2 should mostly pass WITHOUT random delay (high spin count)."""
        result = self._run_without_cutracer(bug_num=2, spin_count=5000)
        failures, total = self._parse_failure_count(result.stdout)
        self.assertIsNotNone(
            failures,
            f"Could not parse failure count from output.\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}",
        )
        self.assertLess(
            failures,
            total // 2,
            f"Expected < 50% failure rate WITHOUT random delay, "
            f"got {failures}/{total}. "
            f"Bug2 should mostly pass due to timing luck without CUTracer.",
        )
