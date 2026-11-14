# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Tests for cutracer CLI."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from click.testing import CliRunner
from cutracer.cli import main
from cutracer.validation.cli import _detect_format, _format_size
from tests.test_base import EXAMPLE_INPUTS_DIR


class DetectFormatTest(unittest.TestCase):
    """Tests for _detect_format function."""

    def test_detect_ndjson(self):
        self.assertEqual(_detect_format(Path("trace.ndjson")), "json")

    def test_detect_ndjson_zst(self):
        self.assertEqual(_detect_format(Path("trace.ndjson.zst")), "json")

    def test_detect_log(self):
        self.assertEqual(_detect_format(Path("trace.log")), "text")

    def test_detect_unknown(self):
        self.assertEqual(_detect_format(Path("trace.txt")), "unknown")
        self.assertEqual(_detect_format(Path("trace.bin")), "unknown")


class FormatSizeTest(unittest.TestCase):
    """Tests for _format_size function."""

    def test_format_bytes(self):
        self.assertEqual(_format_size(500), "500 B")

    def test_format_kilobytes(self):
        self.assertEqual(_format_size(2048), "2.0 KB")

    def test_format_megabytes(self):
        self.assertEqual(_format_size(1048576), "1.0 MB")
        self.assertEqual(_format_size(2621440), "2.5 MB")


class ValidateCommandTest(unittest.TestCase):
    """Tests for validate subcommand."""

    def setUp(self):
        self.test_dir = EXAMPLE_INPUTS_DIR
        self.runner = CliRunner()

    def test_validate_valid_json(self):
        """Test validating a valid NDJSON file."""
        result = self.runner.invoke(
            main, ["validate", str(self.test_dir / "reg_trace_sample.ndjson")]
        )
        self.assertEqual(result.exit_code, 0)

    def test_validate_valid_json_zst(self):
        """Test validating a valid Zstd-compressed NDJSON file."""
        result = self.runner.invoke(
            main, ["validate", str(self.test_dir / "reg_trace_sample.ndjson.zst")]
        )
        self.assertEqual(result.exit_code, 0)

    def test_validate_valid_text(self):
        """Test validating a valid text log file."""
        result = self.runner.invoke(
            main, ["validate", str(self.test_dir / "reg_trace_sample.log")]
        )
        self.assertEqual(result.exit_code, 0)

    def test_validate_invalid_syntax(self):
        """Test validating a file with invalid JSON syntax."""
        result = self.runner.invoke(
            main, ["validate", str(self.test_dir / "invalid_syntax.ndjson")]
        )
        self.assertEqual(result.exit_code, 1)

    def test_validate_invalid_schema(self):
        """Test validating a file with schema errors."""
        result = self.runner.invoke(
            main, ["validate", str(self.test_dir / "invalid_schema.ndjson")]
        )
        self.assertEqual(result.exit_code, 1)

    def test_validate_quiet_mode(self):
        """Test quiet mode returns only exit code."""
        result = self.runner.invoke(
            main,
            ["validate", "--quiet", str(self.test_dir / "reg_trace_sample.ndjson")],
        )
        self.assertEqual(result.exit_code, 0)
        # Quiet mode should produce no output
        self.assertEqual(result.output.strip(), "")

    def test_validate_json_output(self):
        """Test JSON output format."""
        result = self.runner.invoke(
            main,
            ["validate", "--json", str(self.test_dir / "reg_trace_sample.ndjson")],
        )
        self.assertEqual(result.exit_code, 0)
        # Should contain JSON output
        self.assertIn('"valid"', result.output)

    def test_validate_file_not_found(self):
        """Test error handling for non-existent file."""
        result = self.runner.invoke(main, ["validate", "/nonexistent/file.ndjson"])
        self.assertEqual(result.exit_code, 2)

    def test_validate_unknown_format(self):
        """Test error handling for unknown format."""
        # Create a temporary file with unknown extension
        unknown_file = self.test_dir / "reg_trace_sample.unknown"
        # Copy content from existing file for the test
        if not unknown_file.exists():
            import shutil

            shutil.copy(self.test_dir / "reg_trace_sample.ndjson", unknown_file)

        try:
            result = self.runner.invoke(main, ["validate", str(unknown_file)])
            self.assertEqual(result.exit_code, 2)
            self.assertIn("Cannot auto-detect format", result.output)
        finally:
            # Cleanup
            if unknown_file.exists():
                unknown_file.unlink()

    def test_validate_explicit_format_json(self):
        """Test explicit --format json option."""
        result = self.runner.invoke(
            main,
            [
                "validate",
                "--format",
                "json",
                str(self.test_dir / "reg_trace_sample.ndjson"),
            ],
        )
        self.assertEqual(result.exit_code, 0)

    def test_validate_explicit_format_text(self):
        """Test explicit --format text option."""
        result = self.runner.invoke(
            main,
            [
                "validate",
                "--format",
                "text",
                str(self.test_dir / "reg_trace_sample.log"),
            ],
        )
        self.assertEqual(result.exit_code, 0)


class MainEntryPointTest(unittest.TestCase):
    """Tests for main entry point."""

    def setUp(self):
        self.runner = CliRunner()

    def test_version_flag(self):
        """Test --version flag."""
        result = self.runner.invoke(main, ["--version"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("cutracer", result.output)

    def test_help_flag(self):
        """Test --help flag."""
        result = self.runner.invoke(main, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("validate", result.output)

    def test_help_shows_trace_subcommand(self):
        """Test that trace subcommand appears in help output."""
        result = self.runner.invoke(main, ["--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("trace", result.output)

    def test_no_command(self):
        """Test error when no command is provided."""
        result = self.runner.invoke(main, [])
        # Click group with required subcommand returns exit code 2 when no command provided
        self.assertEqual(result.exit_code, 2)


class TraceCommandTest(unittest.TestCase):
    """Tests for trace subcommand."""

    def setUp(self):
        self.runner = CliRunner()

    def test_trace_help(self):
        """Test trace --help shows expected options."""
        result = self.runner.invoke(main, ["trace", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--instrument", result.output)
        self.assertIn("--cutracer-so", result.output)
        self.assertIn("--kernel-filters", result.output)

    def test_trace_no_command_shows_error(self):
        """Test trace without a command shows usage error."""
        result = self.runner.invoke(main, ["trace"])
        self.assertNotEqual(result.exit_code, 0)

    def test_trace_rejects_nonblocking_launches_with_a_serial_analysis(self):
        with tempfile.NamedTemporaryFile(suffix=".so") as library:
            result = self.runner.invoke(
                main,
                [
                    "trace",
                    "--cutracer-so",
                    library.name,
                    "--nonblocking-launches",
                    "-a",
                    "deadlock_detection",
                    "--",
                    "true",
                ],
            )
        # A clean CLI error, not a traceback.
        self.assertIsInstance(result.exception, SystemExit, result.output)
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("deadlock_detection", result.output)


class ModuleEntryPointTest(unittest.TestCase):
    """Tests for python -m cutracer entry point."""

    def test_module_help(self):
        """Test python -m cutracer --help works."""
        result = subprocess.run(
            [sys.executable, "-m", "cutracer", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("cutracer", result.stdout)
        self.assertIn("validate", result.stdout)


if __name__ == "__main__":
    unittest.main()
