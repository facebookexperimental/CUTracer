# Copyright (c) Meta Platforms, Inc. and affiliates.

"""E2E: cluster + warp delay targeting on a two-CTA cluster pipeline.

`lagging_consumer` runs a slot ring in which the follower warp of cluster rank
1 reads slots it never releases. A delay on that warp of that CTA alone lets
the producer lap it, and the kernel hangs; the same delay anywhere else only
slows the pipeline down. Needs a GPU with thread block clusters (sm_90+).
"""

import os
import shutil
import subprocess
import tempfile
import unittest

from cutracer.runner import _build_cutracer_env, resolve_cutracer_so

FOLLOWER_WARP_MASK = "0x4"  # CTA-local warp 2
DELAY_NS = 1_000_000
KERNEL_TIMEOUT_S = 20
NO_CLUSTERS_EXIT_CODE = 77


class ClusterWarpDelayE2ETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # An inherited CUDA_INJECTION64_PATH makes resolve_cutracer_so() fail.
        os.environ.pop("CUDA_INJECTION64_PATH", None)
        cls.cutracer_so = resolve_cutracer_so()
        cls.binary = os.environ["LAGGING_CONSUMER"]

    def setUp(self) -> None:
        self.output_dir = tempfile.mkdtemp(prefix="cutracer_cluster_warp_")
        self.addCleanup(shutil.rmtree, self.output_dir, ignore_errors=True)

    def _run(
        self,
        *binary_args: str,
        delay_mode: str,
        delay_cluster_cta_id: int | None = None,
        delay_warp_mask: str | None = FOLLOWER_WARP_MASK,
    ) -> subprocess.CompletedProcess[str]:
        """Run the binary with a fixed delay before every mbarrier wait.

        Delays the follower warp only, unless `delay_warp_mask` is None (every
        warp of the selected CTA).
        """
        env = _build_cutracer_env(
            cutracer_so=self.cutracer_so,
            instrument="random_delay",
            analysis="random_delay",
            kernel_filters="lagging_consumer_kernel",
            instr_categories=None,
            trace_format=None,
            output_dir=self.output_dir,
            verbose=None,
            zstd_level=None,
            delay_ns=DELAY_NS,
            delay_enable_prob=1.0,
            delay_mode=delay_mode,
            delay_cluster_cta_id=delay_cluster_cta_id,
            delay_warp_mask=delay_warp_mask,
            delay_patterns="SYNCS.PHASECHK",
            kernel_timeout_s=KERNEL_TIMEOUT_S,
            no_data_timeout_s=0,
        )
        result = subprocess.run(
            [self.binary, *binary_args],
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode == NO_CLUSTERS_EXIT_CODE:
            self.skipTest(result.stdout.strip())
        return result

    def _assert_completed(self, result: subprocess.CompletedProcess[str]) -> None:
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output[-4000:])
        self.assertNotIn("Kernel timeout", output)

    def test_rank1_follower_delay_hangs(self) -> None:
        result = self._run(delay_mode="cluster_fixed", delay_cluster_cta_id=1)
        output = result.stdout + result.stderr
        self.assertNotEqual(result.returncode, 0, output[-4000:])
        self.assertIn("Kernel timeout: running for", output)
        self.assertIn("kernel=lagging_consumer_kernel", output)

    def test_rank0_follower_delay_completes(self) -> None:
        self._assert_completed(
            self._run(delay_mode="cluster_fixed", delay_cluster_cta_id=0)
        )

    def test_follower_delay_in_every_cta_completes(self) -> None:
        self._assert_completed(self._run(delay_mode="fixed"))

    def test_rank1_every_warp_delay_completes(self) -> None:
        # Slowing all of rank 1 also slows its releaser, which the producer
        # waits for, so rank 1's follower is never lapped.
        self._assert_completed(
            self._run(
                delay_mode="cluster_fixed",
                delay_cluster_cta_id=1,
                delay_warp_mask=None,
            )
        )

    def test_rank1_follower_delay_after_fix_completes(self) -> None:
        self._assert_completed(
            self._run("--fixed", delay_mode="cluster_fixed", delay_cluster_cta_id=1)
        )
