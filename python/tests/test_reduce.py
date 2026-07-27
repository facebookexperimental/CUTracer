# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Tests for CUTracer reduce module.

Tests cover:
- DelayPoint and DelayConfigMutator (config_mutator.py)
- Reduction algorithm (reduce.py)
- Report generation (report.py)
"""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cutracer.reduce.config_mutator import DelayConfigMutator, DelayPoint
from cutracer.reduce.reduce import (
    ConfigDoesNotTriggerError,
    reduce_delay_points,
    ReduceConfig,
    ReduceResult,
    ReplayConfig,
    ReplayExecutionError,
    ReplayOutcome,
    run_replay,
)
from cutracer.reduce.report import format_report_text, generate_report

# Sample delay config for testing (matches DELAY_CONFIG_SCHEMA)
SAMPLE_DELAY_CONFIG = {
    "version": "1.0",
    "delay_ns": 100,
    "kernels": {
        "kernel_0_a1b2c3d4e5f60001": {
            "kernel_name": "matmul_kernel",
            "kernel_checksum": "a1b2c3d4e5f60001",
            "timestamp": "2026-02-03T21:15:21.567",
            "instrumentation_points": {
                "0x100": {
                    "pc": 256,
                    "sass": "LDG.E.64 R8, [R2.64]",
                    "delay": 100,
                    "on": True,
                },
                "0x108": {
                    "pc": 264,
                    "sass": "STG.E.64 [R4.64], R8",
                    "delay": 100,
                    "on": True,
                },
                "0x110": {
                    "pc": 272,
                    "sass": "BAR.SYNC 0",
                    "delay": 100,
                    "on": False,
                },
            },
        },
        "kernel_1_a1b2c3d4e5f60002": {
            "kernel_name": "softmax_kernel",
            "kernel_checksum": "a1b2c3d4e5f60002",
            "timestamp": "2026-02-03T21:15:22.123",
            "instrumentation_points": {
                "0x200": {
                    "pc": 512,
                    "sass": "FADD R0, R1, R2",
                    "delay": 50,
                    "on": True,
                },
            },
        },
    },
}


class DelayConfigMutatorTest(unittest.TestCase):
    """Tests for DelayPoint and DelayConfigMutator classes."""

    def setUp(self):
        """Create a temporary config file for testing."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_path = self.temp_dir / "test_config.json"
        with open(self.config_path, "w") as f:
            json.dump(SAMPLE_DELAY_CONFIG, f)

    def tearDown(self):
        """Remove the temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_delay_point_creation_and_repr(self):
        """Test creating DelayPoints and string representation."""
        point_on = DelayPoint(
            kernel_key="kernel_0",
            kernel_name="matmul_kernel",
            pc_offset="0x100",
            sass="LDG.E.64 R8, [R2.64]",
            delay_ns=100,
            enabled=True,
        )
        point_off = DelayPoint(
            kernel_key="k0",
            kernel_name="test",
            pc_offset="0x00",
            sass="STG [R1], R0",
            delay_ns=100,
            enabled=False,
        )

        self.assertEqual(point_on.kernel_key, "kernel_0")
        self.assertEqual(point_on.kernel_name, "matmul_kernel")
        self.assertTrue(point_on.enabled)
        self.assertFalse(point_off.enabled)
        self.assertIn("[ON]", repr(point_on))
        self.assertIn("[OFF]", repr(point_off))

    def test_load_config_and_enabled_points(self):
        """Test loading a delay config and getting enabled points."""
        mutator = DelayConfigMutator(str(self.config_path))

        # Should have 4 delay points total (3 from kernel_0, 1 from kernel_1)
        self.assertEqual(len(mutator), 4)
        self.assertEqual(len(mutator.delay_points), 4)

        # 3 points are enabled (2 from kernel_0 + 1 from kernel_1)
        enabled = mutator.enabled_points
        self.assertEqual(len(enabled), 3)
        for point in enabled:
            self.assertTrue(point.enabled)

        # Test repr
        self.assertIn("DelayConfigMutator", repr(mutator))
        self.assertIn("3/4", repr(mutator))

    def test_set_point_enabled_and_all_enabled(self):
        """Test enabling/disabling delay points individually and all at once."""
        mutator = DelayConfigMutator(str(self.config_path))
        point = mutator.enabled_points[0]
        self.assertTrue(point.enabled)

        # Disable single point
        mutator.set_point_enabled(point, False)
        self.assertFalse(point.enabled)
        self.assertFalse(
            mutator.config["kernels"][point.kernel_key]["instrumentation_points"][
                point.pc_offset
            ]["on"]
        )

        # Disable all
        mutator.set_all_enabled(False)
        self.assertEqual(len(mutator.enabled_points), 0)

        # Enable all
        mutator.set_all_enabled(True)
        self.assertEqual(len(mutator.enabled_points), len(mutator.delay_points))

    def test_save_config_and_clone(self):
        """Test saving config and cloning mutator."""
        mutator = DelayConfigMutator(str(self.config_path))

        # Modify and save to new file
        mutator.set_all_enabled(False)
        output_path = self.temp_dir / "output_config.json"
        saved_path = mutator.save(str(output_path))

        # Verify saved file
        self.assertEqual(saved_path, str(output_path))
        with open(saved_path) as f:
            saved_config = json.load(f)
        for kernel_config in saved_config["kernels"].values():
            for point in kernel_config["instrumentation_points"].values():
                self.assertFalse(point["on"])

        # Test save to temp file
        mutator2 = DelayConfigMutator(str(self.config_path))
        temp_saved = mutator2.save()
        self.assertTrue(os.path.exists(temp_saved))
        os.unlink(temp_saved)

        # Test clone isolation
        mutator3 = DelayConfigMutator(str(self.config_path))
        cloned = mutator3.clone()
        mutator3.set_all_enabled(False)
        self.assertEqual(len(cloned.enabled_points), 3)  # Clone unaffected

    def test_schema_validation(self):
        """Test schema validation for config files."""
        # Valid config passes validation
        mutator = DelayConfigMutator(str(self.config_path), validate=True)
        self.assertEqual(len(mutator.delay_points), 4)

        # Invalid config fails validation
        invalid_config = {"delay_ns": 100, "kernels": {}}  # Missing 'version'
        invalid_path = self.temp_dir / "invalid_config.json"
        with open(invalid_path, "w") as f:
            json.dump(invalid_config, f)

        with self.assertRaises(ValueError) as context:
            DelayConfigMutator(str(invalid_path), validate=True)
        self.assertIn("Invalid delay config", str(context.exception))

        # Validation can be skipped
        simple_config = {
            "kernels": {
                "kernel_0": {
                    "kernel_name": "test",
                    "instrumentation_points": {
                        "0x100": {"sass": "NOP", "delay": 100, "on": True},
                    },
                }
            }
        }
        simple_path = self.temp_dir / "simple_config.json"
        with open(simple_path, "w") as f:
            json.dump(simple_config, f)

        mutator = DelayConfigMutator(str(simple_path), validate=False)
        self.assertEqual(len(mutator.delay_points), 1)


class ReplayTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_path = self.temp_dir / "replay.json"
        self.config_path.write_text(json.dumps(SAMPLE_DELAY_CONFIG))
        self.cutracer_so = self.temp_dir / "cutracer.so"
        self.cutracer_so.touch()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_replay_config_rejects_non_positive_timeout(self):
        for timeout in (0, -1):
            with self.subTest(timeout=timeout):
                with self.assertRaisesRegex(ValueError, "timeout must be positive"):
                    ReplayConfig(oracle_argv=["oracle"], timeout=timeout)

    def test_replay_uses_current_runtime_and_structured_argv(self):
        calls = []

        def runner(argv, **kwargs):
            calls.append((argv, kwargs))
            return subprocess.CompletedProcess(argv, 0, "oracle output", "")

        result = run_replay(
            str(self.config_path),
            ReplayConfig(
                oracle_argv=["python", "oracle.py", "--case", "one two"],
                cutracer_so=str(self.cutracer_so),
                base_env={
                    "PATH": "/bin",
                    "CUDA_INJECTION64_PATH": "/feature-installed/cutracer.so",
                },
            ),
            runner=runner,
        )

        self.assertEqual(result.outcome, ReplayOutcome.INTERESTING)
        argv, kwargs = calls[0]
        self.assertEqual(argv, ["python", "oracle.py", "--case", "one two"])
        self.assertFalse(kwargs["shell"])
        self.assertEqual(kwargs["env"]["CUDA_INJECTION64_PATH"], str(self.cutracer_so))
        self.assertEqual(kwargs["env"]["CUTRACER_DELAY_NS"], "100")
        self.assertEqual(
            kwargs["env"]["CUTRACER_DELAY_LOAD_PATH"], str(self.config_path)
        )

    def test_replay_classifies_oracle_and_infra_exit_codes(self):
        def result_for(returncode):
            def runner(argv, **kwargs):
                return subprocess.CompletedProcess(argv, returncode, "", "failure")

            return run_replay(
                str(self.config_path),
                ReplayConfig(
                    oracle_argv=["./oracle"],
                    cutracer_so=str(self.cutracer_so),
                ),
                runner=runner,
            )

        self.assertEqual(result_for(1).outcome, ReplayOutcome.NOT_INTERESTING)
        self.assertEqual(result_for(2).outcome, ReplayOutcome.INFRA_ERROR)

        def runner(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 1, "", "failure")

        empty_allowlist = run_replay(
            str(self.config_path),
            ReplayConfig(
                oracle_argv=["./oracle"],
                cutracer_so=str(self.cutracer_so),
                not_interesting_exit_codes=(),
            ),
            runner=runner,
        )
        self.assertEqual(empty_allowlist.outcome, ReplayOutcome.INFRA_ERROR)

    def test_reducer_aborts_on_replay_infra_error(self):
        def runner(argv, **kwargs):
            return subprocess.CompletedProcess(argv, 2, "", "launch failed")

        config = ReduceConfig(
            config_path=str(self.config_path),
            validate_schema=False,
            replay=ReplayConfig(
                oracle_argv=["./oracle"],
                cutracer_so=str(self.cutracer_so),
            ),
            replay_runner=runner,
        )
        with self.assertRaises(ReplayExecutionError):
            reduce_delay_points(config)


class ReduceDelayPointsTest(unittest.TestCase):
    """Tests for reduce_delay_points function and ReduceResult."""

    def setUp(self):
        """Create temporary config for testing."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config_path = self.temp_dir / "test_config.json"

        config = {
            "kernels": {
                "kernel_0": {
                    "kernel_name": "test_kernel",
                    "instrumentation_points": {
                        "0x100": {"sass": "LDG R0, [R1]", "delay": 100, "on": True},
                        "0x108": {"sass": "STG [R2], R3", "delay": 100, "on": True},
                        "0x110": {"sass": "BAR.SYNC 0", "delay": 100, "on": True},
                    },
                }
            }
        }
        with open(self.config_path, "w") as f:
            json.dump(config, f)

    def tearDown(self):
        """Remove the temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_reduce_result_properties(self):
        """Test ReduceResult success property and summary."""
        point = DelayPoint("k0", "matmul", "0x100", "BAR.SYNC 0", 100, True)

        result_with_points = ReduceResult(
            total_points=10, essential_points=[point], iterations=10
        )
        self.assertTrue(result_with_points.success)
        summary = result_with_points.summary()
        self.assertIn("Reduction complete", summary)
        self.assertIn("Total points tested: 10", summary)
        self.assertIn("Essential points found: 1", summary)

        result_no_points = ReduceResult(
            total_points=5, essential_points=[], iterations=5
        )
        self.assertFalse(result_no_points.success)

    @patch("cutracer.reduce.reduce.run_test")
    def test_reduce_finds_essential_points(self, mock_run_test):
        """Test reduction finds essential points."""

        def test_behavior(script, config_path, verbose=False):
            with open(config_path) as f:
                cfg = json.load(f)
            points = cfg["kernels"]["kernel_0"]["instrumentation_points"]
            return points["0x108"]["on"]  # Race only when STG is enabled

        mock_run_test.side_effect = test_behavior

        reduce_config = ReduceConfig(
            config_path=str(self.config_path),
            test_script="./test.sh",
            validate_schema=False,
        )
        result = reduce_delay_points(reduce_config)

        self.assertTrue(result.success)
        self.assertEqual(len(result.essential_points), 1)
        self.assertEqual(result.essential_points[0].pc_offset, "0x108")

    @patch("cutracer.reduce.reduce.run_test")
    def test_reduce_edge_cases(self, mock_run_test):
        """Test reduction edge cases: no essential points, initial no race."""
        # Test: No essential points (race always occurs)
        mock_run_test.return_value = True
        reduce_config = ReduceConfig(
            config_path=str(self.config_path),
            test_script="./test.sh",
            validate_schema=False,
        )
        result = reduce_delay_points(reduce_config)
        self.assertFalse(result.success)
        self.assertEqual(len(result.essential_points), 0)

        # Test: Initial config doesn't trigger race
        mock_run_test.return_value = False
        with self.assertRaises(ConfigDoesNotTriggerError) as context:
            reduce_delay_points(reduce_config)
        self.assertIn(
            "Initial config does not trigger the race", str(context.exception)
        )

        # Test: Empty config (no enabled points)
        empty_config = {
            "kernels": {
                "kernel_0": {
                    "kernel_name": "test",
                    "instrumentation_points": {
                        "0x100": {"sass": "LDG R0, [R1]", "delay": 100, "on": False},
                    },
                }
            }
        }
        empty_config_path = self.temp_dir / "empty_config.json"
        with open(empty_config_path, "w") as f:
            json.dump(empty_config, f)

        empty_reduce_config = ReduceConfig(
            config_path=str(empty_config_path),
            test_script="./test.sh",
            validate_schema=False,
        )
        with self.assertRaises(ValueError) as context:
            reduce_delay_points(empty_reduce_config)
        self.assertIn("No enabled delay points", str(context.exception))

    @patch("cutracer.reduce.reduce.run_test")
    def test_reduce_with_callback_and_output(self, mock_run_test):
        """Test reduction with progress callback and output file."""
        mock_run_test.return_value = True
        callback_calls = []

        def progress_callback(iteration, total, point, is_essential):
            callback_calls.append((iteration, total, point.pc_offset, is_essential))

        output_path = str(self.temp_dir / "minimal.json")
        reduce_config = ReduceConfig(
            config_path=str(self.config_path),
            test_script="./test.sh",
            output_path=output_path,
            validate_schema=False,
        )
        result = reduce_delay_points(reduce_config, progress_callback=progress_callback)

        # Verify callbacks
        self.assertEqual(len(callback_calls), 3)
        for iteration, total, pc, _is_essential in callback_calls:
            self.assertGreater(iteration, 0)
            self.assertEqual(total, 3)
            self.assertTrue(pc.startswith("0x"))

        # Verify output saved
        self.assertEqual(result.minimal_config_path, output_path)
        self.assertTrue(os.path.exists(output_path))


class ReportTest(unittest.TestCase):
    """Tests for report generation."""

    def test_generate_and_format_report(self):
        """Test generating and formatting a report."""
        point = DelayPoint("k0", "matmul", "0x100", "BAR.SYNC 0", 100, True)
        result = ReduceResult(
            total_points=10,
            essential_points=[point],
            iterations=10,
            minimal_config_path="/tmp/minimal.json",
        )

        report = generate_report(
            result=result,
            config_path="/tmp/config.json",
            test_script="./test.sh",
            source_path="/tmp/source.py",
        )

        # Verify report structure
        self.assertIn("metadata", report)
        self.assertIn("summary", report)
        self.assertIn("essential_delay_points", report)
        self.assertEqual(report["summary"]["total_points"], 10)
        self.assertEqual(report["summary"]["essential_points"], 1)
        self.assertEqual(len(report["essential_delay_points"]), 1)
        self.assertEqual(report["essential_delay_points"][0]["sass"], "BAR.SYNC 0")

        # Verify text format
        text = format_report_text(report)
        self.assertIn("CUTRACER REDUCE REPORT", text)
        self.assertIn("SUMMARY", text)
        self.assertIn("ESSENTIAL DELAY POINTS", text)
        self.assertIn("BAR.SYNC 0", text)
        self.assertIn("Total points tested: 10", text)


if __name__ == "__main__":
    unittest.main()
