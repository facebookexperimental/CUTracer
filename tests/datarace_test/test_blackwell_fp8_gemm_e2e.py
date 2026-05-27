# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Local-only E2E test wrapper for the Blackwell FP8 GEMM driver.

Runs ``blackwell-fp8-gemm_test.py`` (bundled as a resource) as a subprocess
and asserts that all iterations pass within FP8 tolerance. The driver script
launches the upstream TLX warp-specialized Blackwell GEMM kernel with FP8
(e4m3) inputs, which emits ``UTCQMMA`` SASS — the non-HMMA alias whose
descriptor URh+1 collection was fixed in D105661162.

This target is tagged ``local_only`` because Blackwell (sm100) is not yet
available on Remote Execution platforms, so it must run on a local devGPU/
devserver with a Blackwell card.
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import pkg_resources
import torch

_BLACKWELL_MAJOR = 10  # sm100 = Blackwell


class TestBlackwellFp8GemmE2E(unittest.TestCase):
    """E2E: verify the Blackwell FP8 GEMM driver runs cleanly on sm100."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        major, _ = torch.cuda.get_device_capability()
        if major != _BLACKWELL_MAJOR:
            raise unittest.SkipTest(
                f"Requires Blackwell GPU (sm{_BLACKWELL_MAJOR}0), got sm{major}0"
            )

        script_content = pkg_resources.resource_string(
            "datarace_test_resources",
            "blackwell_fp8_gemm_test.py",
        ).decode()
        tmp = tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False, prefix="blackwell_fp8_gemm_"
        )
        tmp.write(script_content)
        tmp.flush()
        tmp.close()
        cls.script_path = tmp.name
        cls.tmp_dir = tempfile.mkdtemp(prefix="blackwell_fp8_gemm_run_")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "script_path") and os.path.exists(cls.script_path):
            os.unlink(cls.script_path)
        if hasattr(cls, "tmp_dir") and os.path.exists(cls.tmp_dir):
            shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    @staticmethod
    def _parse_results(output):
        """Parse 'Results: X passed, Y failed out of N runs' from script output."""
        match = re.search(
            r"Results:\s*(\d+)\s*passed,\s*(\d+)\s*failed\s*out of\s*(\d+)\s*runs",
            output,
        )
        if match:
            return (
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
            )
        return None, None, None

    def test_blackwell_fp8_gemm_runs_clean(self):
        """Driver should run all iterations within FP8 tolerance."""
        result = subprocess.run(
            [sys.executable, self.script_path],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=self.tmp_dir,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"Script exited non-zero ({result.returncode}).\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )
        passed, failed, total = self._parse_results(result.stdout)
        self.assertIsNotNone(
            failed,
            f"Could not parse results line from output.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )
        self.assertEqual(
            failed,
            0,
            f"Expected 0 failed runs, got {failed}/{total}.\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )
        self.assertGreater(passed, 0, "Expected at least one passed run")
