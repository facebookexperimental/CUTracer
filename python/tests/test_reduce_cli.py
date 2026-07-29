# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for the reduce CLI command."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from click.testing import CliRunner
from cutracer.cli import main


class ReduceCliTest(unittest.TestCase):
    """Tests for reduce command CLI interface."""

    def setUp(self):
        self.runner = CliRunner()
        self.isolated_filesystem = self.runner.isolated_filesystem()
        self.isolated_filesystem.__enter__()
        self.temp_dir = tempfile.mkdtemp()
        self.cutracer_so = Path(self.temp_dir) / "cutracer.so"
        self.cutracer_so.touch()
        self.runtime_args = ["--cutracer-so", str(self.cutracer_so)]

        # Create a test delay config file matching DELAY_CONFIG_SCHEMA
        self.config_file = Path(self.temp_dir) / "delay_config.json"
        self.config_data = {
            "version": "1.0",
            "delay_ns": 10000,
            "kernels": {
                "matmul_kernel_1234567890abcdef": {
                    "kernel_name": "matmul_kernel",
                    "kernel_checksum": "1234567890abcdef",
                    "timestamp": "2026-01-01T00:00:00.000",
                    "instrumentation_points": {
                        "256": {
                            "pc": 256,
                            "sass": "STG.E [R2], R4;",
                            "delay": 10000,
                            "on": True,
                        },
                        "512": {
                            "pc": 512,
                            "sass": "LDG.E R5, [R6];",
                            "delay": 10000,
                            "on": True,
                        },
                        "768": {
                            "pc": 768,
                            "sass": "BAR.SYNC 0x0;",
                            "delay": 10000,
                            "on": True,
                        },
                    },
                },
                "softmax_kernel_fedcba0987654321": {
                    "kernel_name": "softmax_kernel",
                    "kernel_checksum": "fedcba0987654321",
                    "timestamp": "2026-01-01T00:00:01.000",
                    "instrumentation_points": {
                        "1024": {
                            "pc": 1024,
                            "sass": "STG.E [R10], R12;",
                            "delay": 10000,
                            "on": True,
                        },
                    },
                },
            },
        }
        with open(self.config_file, "w") as f:
            json.dump(self.config_data, f, indent=2)

        # Create a test script that always returns "race detected" (exit 0)
        self.test_script_race = Path(self.temp_dir) / "test_race.sh"
        self.test_script_race.write_text("#!/bin/bash\nexit 0\n")
        self.test_script_race.chmod(0o755)

        # Create a test script that always returns "no race" (exit 1)
        self.test_script_no_race = Path(self.temp_dir) / "test_no_race.sh"
        self.test_script_no_race.write_text("#!/bin/bash\nexit 1\n")
        self.test_script_no_race.chmod(0o755)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        self.isolated_filesystem.__exit__(None, None, None)

    def test_reduce_help(self):
        """Test reduce --help shows usage information."""
        result = self.runner.invoke(
            main,
            ["reduce", "--help"],
        )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Reduce delay injection config", result.output)
        self.assertIn("--config", result.output)
        self.assertIn("--test", result.output)

    def test_reduce_missing_config(self):
        """Test reduce fails with missing config file."""
        result = self.runner.invoke(
            main,
            [
                "reduce",
                "--config",
                "/nonexistent/config.json",
                "--test",
                str(self.test_script_race),
            ],
        )

        self.assertNotEqual(result.exit_code, 0)

    def test_reduce_requires_config_option(self):
        """Test reduce never discovers or synthesizes a config implicitly."""
        result = self.runner.invoke(
            main,
            [
                "reduce",
                "--test",
                str(self.test_script_race),
            ],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Missing option '--config'", result.output)

    def test_reduce_rejects_non_positive_timeout(self):
        for timeout in (0, -1):
            with self.subTest(timeout=timeout):
                result = self.runner.invoke(
                    main,
                    [
                        "reduce",
                        "--config",
                        str(self.config_file),
                        "--test",
                        str(self.test_script_race),
                        "--timeout",
                        str(timeout),
                    ],
                )

                self.assertNotEqual(result.exit_code, 0)
                self.assertIn("not in the range", result.output)

    def test_reduce_missing_test_script(self):
        """Test reduce fails with missing test script."""
        result = self.runner.invoke(
            main,
            [
                "reduce",
                *self.runtime_args,
                "--config",
                str(self.config_file),
                "--test",
                "/nonexistent/test.sh",
            ],
        )

        # Should fail when trying to run the nonexistent test script
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("No such file or directory", result.output)

    def test_reduce_basic_with_race(self):
        """Test basic reduce command with race always detected."""
        output_file = Path(self.temp_dir) / "report.json"
        minimal_config = Path(self.temp_dir) / "minimal.json"

        result = self.runner.invoke(
            main,
            [
                "reduce",
                *self.runtime_args,
                "--config",
                str(self.config_file),
                "--test",
                str(self.test_script_race),
                "--output",
                str(output_file),
                "--minimal-config",
                str(minimal_config),
                "-v",
            ],
        )

        self.assertEqual(result.exit_code, 0, f"Failed with output: {result.output}")
        self.assertIn("CUTRACER REDUCE", result.output)
        self.assertIn("REDUCTION COMPLETE", result.output)
        self.assertTrue(output_file.exists())
        self.assertTrue(minimal_config.exists())

    def test_reduce_initial_no_race_error(self):
        """Test reduce fails if initial config doesn't trigger race."""
        result = self.runner.invoke(
            main,
            [
                "reduce",
                *self.runtime_args,
                "--config",
                str(self.config_file),
                "--test",
                str(self.test_script_no_race),
            ],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Initial config does not trigger the race", result.output)

    def test_reduce_verbose_output(self):
        """Test reduce with verbose output."""
        result = self.runner.invoke(
            main,
            [
                "reduce",
                *self.runtime_args,
                "--config",
                str(self.config_file),
                "--test",
                str(self.test_script_race),
                "-v",
            ],
        )

        self.assertEqual(result.exit_code, 0, f"Failed with output: {result.output}")

    def test_reduce_report_json_format(self):
        """Test that reduce report is valid JSON."""
        output_file = Path(self.temp_dir) / "report.json"

        result = self.runner.invoke(
            main,
            [
                "reduce",
                *self.runtime_args,
                "--config",
                str(self.config_file),
                "--test",
                str(self.test_script_race),
                "--output",
                str(output_file),
            ],
        )

        self.assertEqual(result.exit_code, 0, f"Failed with output: {result.output}")
        self.assertTrue(output_file.exists())

        # Verify it's valid JSON
        with open(output_file) as f:
            report = json.load(f)
        self.assertIn("metadata", report)

    def test_reduce_stats_output(self):
        """Test reduce shows statistics at the end."""
        result = self.runner.invoke(
            main,
            [
                "reduce",
                *self.runtime_args,
                "--config",
                str(self.config_file),
                "--test",
                str(self.test_script_race),
            ],
        )

        self.assertEqual(result.exit_code, 0, f"Failed with output: {result.output}")
        self.assertIn("Stats:", result.output)
        self.assertIn("Total points:", result.output)
        self.assertIn("Minimal set:", result.output)
        self.assertIn("Iterations:", result.output)


class ReduceCliEmptyConfigTest(unittest.TestCase):
    """Tests for reduce command with edge case configs."""

    def setUp(self):
        self.runner = CliRunner()
        self.isolated_filesystem = self.runner.isolated_filesystem()
        self.isolated_filesystem.__enter__()
        self.temp_dir = tempfile.mkdtemp()

        # Create test script
        self.test_script = Path(self.temp_dir) / "test.sh"
        self.test_script.write_text("#!/bin/bash\nexit 0\n")
        self.test_script.chmod(0o755)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        self.isolated_filesystem.__exit__(None, None, None)

    def test_reduce_empty_config(self):
        """Test reduce with config that has no delay points."""
        config_file = Path(self.temp_dir) / "empty_config.json"
        config_data = {
            "version": "1.0",
            "delay_ns": 10000,
            "kernels": {},
        }
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        result = self.runner.invoke(
            main,
            [
                "reduce",
                "--config",
                str(config_file),
                "--test",
                str(self.test_script),
            ],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Error", result.output)

    def test_reduce_all_disabled_points(self):
        """Test reduce with config where all points are disabled."""
        config_file = Path(self.temp_dir) / "disabled_config.json"
        config_data = {
            "version": "1.0",
            "delay_ns": 10000,
            "kernels": {
                "test_kernel_a1b2c3d4e5f67890": {
                    "kernel_name": "test_kernel",
                    "kernel_checksum": "a1b2c3d4e5f67890",
                    "timestamp": "2026-01-01T00:00:00.000",
                    "instrumentation_points": {
                        "256": {
                            "pc": 256,
                            "sass": "STG.E [R2], R4;",
                            "delay": 10000,
                            "on": False,
                        },
                    },
                },
            },
        }
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        result = self.runner.invoke(
            main,
            [
                "reduce",
                "--config",
                str(config_file),
                "--test",
                str(self.test_script),
            ],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Error", result.output)


if __name__ == "__main__":
    unittest.main()
