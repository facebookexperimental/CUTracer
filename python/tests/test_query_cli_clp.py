# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Unit tests for query CLI command.
"""

import unittest

from click.testing import CliRunner
from cutracer.cli import main
from tests.test_base import BaseValidationTest, REG_TRACE_NDJSON

try:
    import yscope_clp_core
except ModuleNotFoundError:
    yscope_clp_core = None


@unittest.skipIf(
    yscope_clp_core is None, "needs yscope_clp_core: pip install 'cutracer[clp]'"
)
class TestQueryCommand(BaseValidationTest):
    """Tests for query CLI command."""

    def setUp(self):
        super().setUp()
        self.runner = CliRunner()
        # create clp archive from ndjson
        self.clp_archive_path = self.temp_dir.joinpath("example_input.clp")
        with yscope_clp_core.open_archive(self.clp_archive_path, "w") as clp_archive:
            clp_archive.add(REG_TRACE_NDJSON)

    def test_analyze_default_head(self):
        """Test analyze with default head (10 records)."""
        result = self.runner.invoke(
            main, ["query", str(str(self.clp_archive_path.absolute()))]
        )
        self.assertEqual(result.exit_code, 0)
        # Should have header + 10 data rows = 11 lines
        lines = [line for line in result.stdout.strip().split("\n") if line]
        self.assertEqual(len(lines), 11)

    def test_analyze_custom_head(self):
        """Test analyze with custom head value."""
        result = self.runner.invoke(
            main, ["query", str(str(self.clp_archive_path.absolute())), "--head", "5"]
        )
        self.assertEqual(result.exit_code, 0)
        lines = [line for line in result.stdout.strip().split("\n") if line]
        self.assertEqual(len(lines), 6)  # header + 5 data rows

    def test_analyze_tail(self):
        """Test analyze with tail option."""
        result = self.runner.invoke(
            main, ["query", str(str(self.clp_archive_path.absolute())), "--tail", "3"]
        )
        self.assertEqual(result.exit_code, 0)
        lines = [line for line in result.stdout.strip().split("\n") if line]
        self.assertEqual(len(lines), 4)  # header + 3 data rows

    def test_analyze_filter(self):
        """Test analyze with filter expression."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(str(self.clp_archive_path.absolute())),
                "--filter",
                "warp=0",
                "--head",
                "5",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        # All displayed records should have warp=0
        # Check that output contains "0" in the warp column
        self.assertIn("0", result.stdout)

    def test_analyze_filter_invalid(self):
        """Test analyze with invalid filter expression."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(str(self.clp_archive_path.absolute())),
                "--filter",
                "invalid",
            ],
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Invalid filter expression", str(result.exception))

    def test_analyze_short_options(self):
        """Test analyze with short option names."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(str(self.clp_archive_path.absolute())),
                "-n",
                "3",
                "-f",
                "warp=0",
            ],
        )
        self.assertEqual(result.exit_code, 0)

    def test_analyze_format_json(self):
        """Test analyze with JSON output format."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(str(self.clp_archive_path.absolute())),
                "--format",
                "json",
                "--head",
                "3",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        # JSON output should be parseable
        import json

        data = json.loads(result.stdout)
        self.assertEqual(len(data), 3)

    def test_analyze_format_csv(self):
        """Test analyze with CSV output format."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(str(self.clp_archive_path.absolute())),
                "--format",
                "csv",
                "--head",
                "3",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        lines = result.stdout.strip().split("\n")
        self.assertEqual(len(lines), 4)  # header + 3 data rows
        # CSV header should contain field names
        self.assertIn("warp", lines[0])

    def test_analyze_format_table_explicit(self):
        """Test analyze with explicit table format."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(str(self.clp_archive_path.absolute())),
                "--format",
                "table",
                "--head",
                "3",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        lines = [line for line in result.stdout.strip().split("\n") if line]
        self.assertEqual(len(lines), 4)  # header + 3 data rows

    def test_analyze_fields_custom(self):
        """Test analyze with custom fields."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(str(self.clp_archive_path.absolute())),
                "--fields",
                "warp,sass",
                "--head",
                "3",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        # Output should contain WARP and SASS headers
        self.assertIn("WARP", result.stdout)
        self.assertIn("SASS", result.stdout)
        # But not PC (since we only asked for warp,sass)
        self.assertNotIn("PC", result.stdout.split("\n")[0])

    def test_analyze_no_header(self):
        """Test analyze with --no-header flag."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(str(self.clp_archive_path.absolute())),
                "--no-header",
                "--head",
                "3",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        lines = [line for line in result.stdout.strip().split("\n") if line]
        self.assertEqual(len(lines), 3)  # 3 data rows, no header

    def test_analyze_format_csv_no_header(self):
        """Test analyze with CSV format and no header."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(str(self.clp_archive_path.absolute())),
                "--format",
                "csv",
                "--no-header",
                "--head",
                "3",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        lines = result.stdout.strip().split("\n")
        self.assertEqual(len(lines), 3)  # 3 data rows, no header

    def test_analyze_group_by_basic(self):
        """Test analyze with --group-by option."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(str(self.clp_archive_path.absolute())),
                "--group-by",
                "warp",
                "--head",
                "3",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        # Should have group headers
        self.assertIn("=== Group:", result.stdout)
        self.assertIn("records) ===", result.stdout)

    def test_analyze_group_by_with_tail(self):
        """Test analyze with --group-by and --tail."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(str(self.clp_archive_path.absolute())),
                "--group-by",
                "warp",
                "--tail",
                "2",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("=== Group:", result.stdout)

    def test_analyze_group_by_with_filter(self):
        """Test analyze with --group-by and --filter."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(str(self.clp_archive_path.absolute())),
                "--group-by",
                "warp",
                "--filter",
                "warp=0",
                "--head",
                "5",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        # Should only have warp=0 group
        self.assertIn("=== Group: 0", result.stdout)

    def test_analyze_group_by_json_format(self):
        """Test analyze with --group-by and JSON format."""
        import json

        result = self.runner.invoke(
            main,
            [
                "query",
                str(str(self.clp_archive_path.absolute())),
                "--group-by",
                "warp",
                "--format",
                "json",
                "--head",
                "2",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        # JSON output should be valid and parseable
        data = json.loads(result.stdout)
        # Since --group-by warp is used, output should have groups and warp_summary
        self.assertIn("groups", data)
        self.assertIsInstance(data["groups"], dict)
        # Each group should be a list of records
        for records in data["groups"].values():
            self.assertIsInstance(records, list)

    def test_analyze_group_by_csv_format(self):
        """Test analyze with --group-by and CSV format."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(str(self.clp_archive_path.absolute())),
                "--group-by",
                "warp",
                "--format",
                "csv",
                "--head",
                "2",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("=== Group:", result.stdout)

    def test_analyze_group_by_short_option(self):
        """Test analyze with -g short option for --group-by."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(str(self.clp_archive_path.absolute())),
                "-g",
                "warp",
                "-n",
                "2",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("=== Group:", result.stdout)

    def test_analyze_count_requires_group_by(self):
        """Test that --count requires --group-by."""
        result = self.runner.invoke(
            main, ["query", str(str(self.clp_archive_path.absolute())), "--count"]
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--count requires --group-by", str(result.stderr))

    def test_analyze_top_requires_count(self):
        """Test that --top requires --count."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(str(self.clp_archive_path.absolute())),
                "--group-by",
                "warp",
                "--top",
                "5",
            ],
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--top requires --count", result.output)

    def test_analyze_count_basic(self):
        """Test analyze with --group-by and --count."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(str(self.clp_archive_path.absolute())),
                "--group-by",
                "warp",
                "--count",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("WARP", result.stdout)
        self.assertIn("COUNT", result.stdout)

    def test_analyze_count_top(self):
        """Test analyze with --count --top."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(str(self.clp_archive_path.absolute())),
                "--group-by",
                "warp",
                "--count",
                "--top",
                "3",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        lines = [line for line in result.stdout.strip().split("\n") if line]
        self.assertEqual(len(lines), 4)  # header + 3 data rows

    def test_analyze_count_json_format(self):
        """Test analyze --count with JSON format."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(str(self.clp_archive_path.absolute())),
                "--group-by",
                "warp",
                "--count",
                "--format",
                "json",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        import json

        data = json.loads(result.stdout)
        self.assertIsInstance(data, dict)

    def test_analyze_count_csv_format(self):
        """Test analyze --count with CSV format."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(str(self.clp_archive_path.absolute())),
                "--group-by",
                "warp",
                "--count",
                "--format",
                "csv",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("warp,count", result.stdout)

    def test_analyze_count_no_header(self):
        """Test analyze --count with --no-header."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(str(self.clp_archive_path.absolute())),
                "--group-by",
                "warp",
                "--count",
                "--no-header",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        # Should not contain header row
        self.assertNotIn("WARP", result.stdout)
        self.assertNotIn("COUNT", result.stdout)

    def test_analyze_group_by_warp_shows_summary(self):
        """Test analyze --group-by warp shows warp summary."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(str(self.clp_archive_path.absolute())),
                "--group-by",
                "warp",
                "--tail",
                "3",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        # Should have warp summary section
        self.assertIn("Warp Summary", result.stdout)
        self.assertIn("Total warps observed:", result.stdout)
        self.assertIn("Completed (EXIT):", result.stdout)
        self.assertIn("In-progress:", result.stdout)
        self.assertIn("Missing (never seen):", result.stdout)

    def test_analyze_group_by_warp_json_has_summary(self):
        """Test analyze --group-by warp with JSON format includes warp_summary."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(str(self.clp_archive_path.absolute())),
                "--group-by",
                "warp",
                "--format",
                "json",
                "--head",
                "2",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        import json

        data = json.loads(result.stdout)
        # JSON should have groups and warp_summary keys
        self.assertIn("groups", data)
        self.assertIn("warp_summary", data)
        self.assertIn("total_observed", data["warp_summary"])
        self.assertIn("completed", data["warp_summary"])
        self.assertIn("in_progress", data["warp_summary"])
        self.assertIn("missing", data["warp_summary"])

    def test_analyze_group_by_non_warp_no_summary(self):
        """Test analyze --group-by with non-warp field does not show warp summary."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(str(self.clp_archive_path.absolute())),
                "--group-by",
                "sass",
                "--head",
                "2",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        # Should NOT have warp summary section
        self.assertNotIn("Warp Summary", result.stdout)

    def test_analyze_group_by_warp_csv_no_summary(self):
        """Test analyze --group-by warp with CSV format does not include summary."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(str(self.clp_archive_path.absolute())),
                "--group-by",
                "warp",
                "--format",
                "csv",
                "--head",
                "2",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        # CSV format should NOT have warp summary section
        self.assertNotIn("Warp Summary", result.stdout)

    def test_analyze_all_records(self):
        """Test analyze with --all-lines flag to show all records."""
        # First count total records
        result_all = self.runner.invoke(
            main,
            [
                "query",
                str(str(self.clp_archive_path.absolute())),
                "--all-lines",
                "--no-header",
            ],
        )
        self.assertEqual(result_all.exit_code, 0)
        all_lines = [line for line in result_all.output.strip().split("\n") if line]

        # Compare with head 10
        result_head = self.runner.invoke(
            main,
            [
                "query",
                str(str(self.clp_archive_path.absolute())),
                "--head",
                "10",
                "--no-header",
            ],
        )
        head_lines = [line for line in result_head.output.strip().split("\n") if line]

        # All should have >= 10 records (or same if file has exactly 10)
        self.assertGreaterEqual(len(all_lines), len(head_lines))

    def test_analyze_all_short_option(self):
        """Test analyze with -a short option for --all-lines."""
        result = self.runner.invoke(
            main,
            ["query", str(str(self.clp_archive_path.absolute())), "-a", "--no-header"],
        )
        self.assertEqual(result.exit_code, 0)

    def test_analyze_format_ndjson(self):
        """Test analyze with NDJSON output format."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(str(self.clp_archive_path.absolute())),
                "--format",
                "ndjson",
                "--head",
                "3",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        lines = result.stdout.strip().split("\n")
        self.assertEqual(len(lines), 3)
        # Each line should be valid JSON
        import json

        for line in lines:
            data = json.loads(line)
            self.assertIsInstance(data, dict)

    def test_analyze_filter_hex_pc(self):
        """Test analyze with hex filter for PC value."""
        # First get the first record's PC
        result = self.runner.invoke(
            main,
            [
                "query",
                str(str(self.clp_archive_path.absolute())),
                "--format",
                "json",
                "--head",
                "1",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        import json

        records = json.loads(result.stdout)
        if records and "pc" in records[0]:
            pc_value = records[0]["pc"]
            # pc is now a hex string like "0x0" or "0x10"
            # Filter using the hex string directly
            result_hex = self.runner.invoke(
                main,
                [
                    "query",
                    str(str(self.clp_archive_path.absolute())),
                    "--filter",
                    f"pc={pc_value}",
                    "--head",
                    "5",
                ],
            )
            self.assertEqual(result_hex.exit_code, 0)
            # All filtered records should have this PC
            self.assertIn(pc_value, result_hex.output)

    def test_analyze_fields_all(self):
        """Test analyze with --fields '*' to show all fields."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(str(self.clp_archive_path.absolute())),
                "--fields",
                "*",
                "--head",
                "1",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        # Should show more than just default fields (warp, pc, sass)
        header = result.stdout.strip().split("\n")[0]
        # Check that non-default fields are included (like type, time, etc.)
        # At minimum should have WARP, PC, SASS
        self.assertIn("WARP", header)
        self.assertIn("PC", header)

    def test_analyze_fields_all_keyword(self):
        """Test analyze with --fields 'all' to show all fields."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(str(self.clp_archive_path.absolute())),
                "--fields",
                "all",
                "--head",
                "1",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        header = result.stdout.strip().split("\n")[0]
        self.assertIn("WARP", header)
        self.assertIn("PC", header)

    def test_analyze_output_file(self):
        """Test analyze with --output to write to file."""
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.txt")
            result = self.runner.invoke(
                main,
                [
                    "query",
                    str(str(self.clp_archive_path.absolute())),
                    "--head",
                    "3",
                    "--output",
                    output_path,
                ],
            )
            self.assertEqual(result.exit_code, 0)
            # Check file was created
            self.assertTrue(os.path.exists(output_path))
            # Check file content
            with open(output_path) as f:
                content = f.read()
            lines = [line for line in content.strip().split("\n") if line]
            self.assertEqual(len(lines), 4)  # header + 3 rows

    def test_analyze_output_ndjson_file(self):
        """Test analyze with --output and --format ndjson."""
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.ndjson")
            result = self.runner.invoke(
                main,
                [
                    "query",
                    str(str(self.clp_archive_path.absolute())),
                    "--format",
                    "ndjson",
                    "--head",
                    "3",
                    "--output",
                    output_path,
                ],
            )
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(os.path.exists(output_path))
            # Check file content is valid NDJSON
            with open(output_path) as f:
                lines = f.read().strip().split("\n")
            self.assertEqual(len(lines), 3)
            for line in lines:
                data = json.loads(line)
                self.assertIsInstance(data, dict)

    def test_analyze_group_by_all_records(self):
        """Test analyze with --group-by and --all-lines."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(str(self.clp_archive_path.absolute())),
                "--group-by",
                "warp",
                "--all-lines",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("=== Group:", result.stdout)

    def test_analyze_count_ndjson_format(self):
        """Test analyze --count with NDJSON format."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(str(self.clp_archive_path.absolute())),
                "--group-by",
                "warp",
                "--count",
                "--format",
                "ndjson",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        import json

        lines = result.stdout.strip().split("\n")
        for line in lines:
            data = json.loads(line)
            self.assertIn("warp", data)
            self.assertIn("count", data)

    def test_query_output_compress(self):
        """Test query with --output and --compress produces valid Zstd file."""
        import json
        import os
        import tempfile

        import zstandard as zstd

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "output.ndjson.zst")
            result = self.runner.invoke(
                main,
                [
                    "query",
                    str(str(self.clp_archive_path.absolute())),
                    "--format",
                    "ndjson",
                    "--head",
                    "3",
                    "--output",
                    output_path,
                    "--compress",
                ],
            )
            self.assertEqual(result.exit_code, 0)
            self.assertTrue(os.path.exists(output_path))

            # Verify it's valid zstd compressed data
            dctx = zstd.ZstdDecompressor()
            with open(output_path, "rb") as f:
                content = dctx.decompress(f.read()).decode("utf-8")
            lines = [line for line in content.strip().split("\n") if line]
            self.assertEqual(len(lines), 3)
            for line in lines:
                data = json.loads(line)
                self.assertIsInstance(data, dict)

    def test_query_compress_requires_output(self):
        """Test that --compress without --output raises an error."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(self.clp_archive_path.absolute()),
                "--head",
                "3",
                "--compress",
            ],
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--compress requires --output", result.output)


if __name__ == "__main__":
    unittest.main()
