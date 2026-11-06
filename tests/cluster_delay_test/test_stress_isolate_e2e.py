# Copyright (c) Meta Platforms, Inc. and affiliates.

"""E2E: `cutracer stress --strategy isolate` finds the lagging consumer.

`lagging_consumer` hangs only when the follower warp (warp 2) of cluster rank 1
is delayed. An isolate search must report exactly that target as a hang, after
a clean control run; with the fix (`--fixed`) every target must pass. `cutracer
reduce`, given the same targeting, must keep the hang with fewer delay points.
Needs a GPU with thread block clusters (sm_90+).
"""

import os
import subprocess
import tempfile
import unittest

from cutracer.reduce.reduce import reduce_delay_points, ReduceConfig, ReplayConfig
from cutracer.runner import resolve_cutracer_so
from cutracer.stress.stress import (
    ATTRIBUTION_ATTRIBUTED,
    ATTRIBUTION_NOT_REPRODUCED,
    run_stress,
    STRATEGY_ISOLATE,
    StressConfig,
    StressResult,
    SYMPTOM_HANG,
)

NO_CLUSTERS_EXIT_CODE = 77


class StressIsolateE2ETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # An inherited CUDA_INJECTION64_PATH makes resolve_cutracer_so() fail.
        os.environ.pop("CUDA_INJECTION64_PATH", None)
        cls.cutracer_so = resolve_cutracer_so()
        cls.binary = os.environ["LAGGING_CONSUMER"]
        plain = subprocess.run(
            [cls.binary], capture_output=True, text=True, timeout=120
        )
        if plain.returncode == NO_CLUSTERS_EXIT_CODE:
            raise unittest.SkipTest(plain.stdout.strip())

    def _search(self, output_dir: str, *binary_args: str) -> StressResult:
        config = StressConfig(
            oracle_argv=[self.binary, "--oracle", *binary_args],
            delay_ladder_ns=[1_000_000],
            strategy=STRATEGY_ISOLATE,
            attempts_per_delay=1,
            baseline_runs=1,
            kernel_timeout_s=20,
            hang_is_bug=True,
            not_interesting_exit_codes=[1],
            kernel_filters="lagging_consumer_kernel",
            output_dir=output_dir,
            timeout=300,
        )
        return run_stress(config, cutracer_so=self.cutracer_so)

    def test_finds_the_follower_of_cluster_rank_1(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            result = self._search(output_dir)

        self.assertTrue(result.reproduced, result.to_dict())
        self.assertEqual(result.attribution, ATTRIBUTION_ATTRIBUTED)
        # 3 warps x 2 cluster ranks; rank 1's warps 0 and 1 pass first.
        self.assertEqual(len(result.targets), 6)
        self.assertEqual(result.completed_trials, 3)
        assert result.triggering is not None
        self.assertEqual(result.triggering.symptom, SYMPTOM_HANG)
        self.assertEqual(
            result.triggering.target,
            {"cluster_rank": 1, "warp": 2, "warp_mask": "0x4"},
        )

    def test_fixed_kernel_passes_every_target(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            result = self._search(output_dir, "--fixed")

        self.assertFalse(result.reproduced, result.to_dict())
        self.assertEqual(result.attribution, ATTRIBUTION_NOT_REPRODUCED)
        self.assertEqual(result.completed_trials, 6)
        self.assertEqual(result.infra_errors, 0)
        self.assertEqual(result.hangs, 0)

    def test_reduce_keeps_the_hang_with_the_recorded_targeting(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            found = self._search(output_dir)
            assert found.triggering is not None and found.triggering.target
            target = found.triggering.target
            reduced = reduce_delay_points(
                ReduceConfig(
                    config_path=found.triggering.config_path,
                    output_path=os.path.join(output_dir, "minimal.json"),
                    replay=ReplayConfig(
                        oracle_argv=[self.binary, "--oracle"],
                        cutracer_so=self.cutracer_so,
                        kernel_filters="lagging_consumer_kernel",
                        delay_mode="cluster_fixed",
                        delay_cluster_cta_id=target["cluster_rank"],
                        delay_warp_mask=target["warp_mask"],
                        delay_patterns="SYNCS.PHASECHK",
                        kernel_timeout_s=10,
                        hang_is_bug=True,
                        output_dir=output_dir,
                        timeout=120,
                    ),
                )
            )

        recorded_points = found.triggering.enabled_point_count
        assert recorded_points is not None
        self.assertTrue(reduced.success)
        self.assertLess(len(reduced.essential_points), recorded_points)
        for point in reduced.essential_points:
            self.assertIn("SYNCS.PHASECHK", point.sass)
