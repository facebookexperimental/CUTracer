# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Unit tests for query CLI command.
"""

import unittest

from click.testing import CliRunner
from cutracer.cli import main
from tests.test_base import (
    BaseValidationTest,
    count_records_of_type,
    KERNEL_EVENTS_NDJSON,
    REG_TRACE_NDJSON,
    REG_TRACE_NDJSON_ZST,
)


class TestQueryCommand(BaseValidationTest):
    """Tests for query CLI command."""

    def setUp(self):
        super().setUp()
        self.runner = CliRunner()

    def test_query_help(self):
        """Test query --help shows usage information."""
        result = self.runner.invoke(main, ["query", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Query and view trace data", result.output)
        self.assertIn("--head", result.output)
        self.assertIn("--tail", result.output)
        self.assertIn("--filter", result.output)

    def test_analyze_default_head(self):
        """Test analyze with default head (10 records)."""
        result = self.runner.invoke(main, ["query", str(REG_TRACE_NDJSON)])
        self.assertEqual(result.exit_code, 0)
        # Should have header + 10 data rows = 11 lines
        lines = [line for line in result.output.strip().split("\n") if line]
        self.assertEqual(len(lines), 11)

    def test_analyze_custom_head(self):
        """Test analyze with custom head value."""
        result = self.runner.invoke(
            main, ["query", str(REG_TRACE_NDJSON), "--head", "5"]
        )
        self.assertEqual(result.exit_code, 0)
        lines = [line for line in result.output.strip().split("\n") if line]
        self.assertEqual(len(lines), 6)  # header + 5 data rows

    def test_analyze_tail(self):
        """Test analyze with tail option."""
        result = self.runner.invoke(
            main, ["query", str(REG_TRACE_NDJSON), "--tail", "3"]
        )
        self.assertEqual(result.exit_code, 0)
        lines = [line for line in result.output.strip().split("\n") if line]
        self.assertEqual(len(lines), 4)  # header + 3 data rows

    def test_analyze_zst_file(self):
        """Test analyze with Zstd-compressed file."""
        result = self.runner.invoke(
            main, ["query", str(REG_TRACE_NDJSON_ZST), "--head", "5"]
        )
        self.assertEqual(result.exit_code, 0)
        lines = [line for line in result.output.strip().split("\n") if line]
        self.assertEqual(len(lines), 6)

    def test_analyze_filter(self):
        """Test analyze with filter expression."""
        result = self.runner.invoke(
            main,
            ["query", str(REG_TRACE_NDJSON), "--filter", "warp=0", "--head", "5"],
        )
        self.assertEqual(result.exit_code, 0)
        # All displayed records should have warp=0
        # Check that output contains "0" in the warp column
        self.assertIn("0", result.output)

    def test_analyze_filter_invalid(self):
        """Test analyze with invalid filter expression."""
        result = self.runner.invoke(
            main, ["query", str(REG_TRACE_NDJSON), "--filter", "invalid"]
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Invalid filter expression", result.output)

    def test_analyze_nonexistent_file(self):
        """Test analyze with non-existent file."""
        result = self.runner.invoke(main, ["query", "/nonexistent/file.ndjson"])
        self.assertNotEqual(result.exit_code, 0)

    def test_analyze_empty_file(self):
        """Test analyze with empty file."""
        empty_file = self.create_temp_file("empty.ndjson", "")
        result = self.runner.invoke(main, ["query", str(empty_file)])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("No records found", result.output)

    def test_analyze_short_options(self):
        """Test analyze with short option names."""
        result = self.runner.invoke(
            main, ["query", str(REG_TRACE_NDJSON), "-n", "3", "-f", "warp=0"]
        )
        self.assertEqual(result.exit_code, 0)

    def test_analyze_format_json(self):
        """Test analyze with JSON output format."""
        result = self.runner.invoke(
            main, ["query", str(REG_TRACE_NDJSON), "--format", "json", "--head", "3"]
        )
        self.assertEqual(result.exit_code, 0)
        # JSON output should be parseable
        import json

        data = json.loads(result.output)
        self.assertEqual(len(data), 3)

    def test_analyze_format_csv(self):
        """Test analyze with CSV output format."""
        result = self.runner.invoke(
            main, ["query", str(REG_TRACE_NDJSON), "--format", "csv", "--head", "3"]
        )
        self.assertEqual(result.exit_code, 0)
        lines = result.output.strip().split("\n")
        self.assertEqual(len(lines), 4)  # header + 3 data rows
        # CSV header should contain field names
        self.assertIn("warp", lines[0])

    def test_analyze_format_table_explicit(self):
        """Test analyze with explicit table format."""
        result = self.runner.invoke(
            main,
            ["query", str(REG_TRACE_NDJSON), "--format", "table", "--head", "3"],
        )
        self.assertEqual(result.exit_code, 0)
        lines = [line for line in result.output.strip().split("\n") if line]
        self.assertEqual(len(lines), 4)  # header + 3 data rows

    def test_analyze_fields_custom(self):
        """Test analyze with custom fields."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(REG_TRACE_NDJSON),
                "--fields",
                "warp,sass",
                "--head",
                "3",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        # Output should contain WARP and SASS headers
        self.assertIn("WARP", result.output)
        self.assertIn("SASS", result.output)
        # But not PC (since we only asked for warp,sass)
        self.assertNotIn("PC", result.output.split("\n")[0])

    def test_analyze_no_header(self):
        """Test analyze with --no-header flag."""
        result = self.runner.invoke(
            main, ["query", str(REG_TRACE_NDJSON), "--no-header", "--head", "3"]
        )
        self.assertEqual(result.exit_code, 0)
        lines = [line for line in result.output.strip().split("\n") if line]
        self.assertEqual(len(lines), 3)  # 3 data rows, no header

    def test_analyze_format_csv_no_header(self):
        """Test analyze with CSV format and no header."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(REG_TRACE_NDJSON),
                "--format",
                "csv",
                "--no-header",
                "--head",
                "3",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        lines = result.output.strip().split("\n")
        self.assertEqual(len(lines), 3)  # 3 data rows, no header

    def test_analyze_group_by_basic(self):
        """Test analyze with --group-by option."""
        result = self.runner.invoke(
            main,
            ["query", str(REG_TRACE_NDJSON), "--group-by", "warp", "--head", "3"],
        )
        self.assertEqual(result.exit_code, 0)
        # Should have group headers
        self.assertIn("=== Group:", result.output)
        self.assertIn("records) ===", result.output)

    def test_group_by_list_field_count(self):
        """Test --group-by on a list-valued field (cta) with --count."""
        result = self.runner.invoke(
            main,
            ["query", str(REG_TRACE_NDJSON), "--group-by", "cta", "--count"],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("[0, 0, 0]", result.output)

    def test_group_by_list_field_head(self):
        """Test --group-by on a list-valued field (cta) with --head."""
        result = self.runner.invoke(
            main,
            ["query", str(REG_TRACE_NDJSON), "--group-by", "cta", "--head", "1"],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("=== Group: [0, 0, 0]", result.output)

    def test_analyze_group_by_with_tail(self):
        """Test analyze with --group-by and --tail."""
        result = self.runner.invoke(
            main,
            ["query", str(REG_TRACE_NDJSON), "--group-by", "warp", "--tail", "2"],
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("=== Group:", result.output)

    def test_analyze_group_by_with_filter(self):
        """Test analyze with --group-by and --filter."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(REG_TRACE_NDJSON),
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
        self.assertIn("=== Group: 0", result.output)

    def test_analyze_group_by_json_format(self):
        """Test analyze with --group-by and JSON format."""
        import json

        result = self.runner.invoke(
            main,
            [
                "query",
                str(REG_TRACE_NDJSON),
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
        data = json.loads(result.output)
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
                str(REG_TRACE_NDJSON),
                "--group-by",
                "warp",
                "--format",
                "csv",
                "--head",
                "2",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("=== Group:", result.output)

    def test_analyze_group_by_short_option(self):
        """Test analyze with -g short option for --group-by."""
        result = self.runner.invoke(
            main,
            ["query", str(REG_TRACE_NDJSON), "-g", "warp", "-n", "2"],
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("=== Group:", result.output)

    def test_analyze_count_requires_group_by(self):
        """Test that --count requires --group-by."""
        result = self.runner.invoke(main, ["query", str(REG_TRACE_NDJSON), "--count"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--count requires --group-by", result.output)

    def test_analyze_top_requires_count(self):
        """Test that --top requires --count."""
        result = self.runner.invoke(
            main,
            ["query", str(REG_TRACE_NDJSON), "--group-by", "warp", "--top", "5"],
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--top requires --count", result.output)

    def test_analyze_count_basic(self):
        """Test analyze with --group-by and --count."""
        result = self.runner.invoke(
            main,
            ["query", str(REG_TRACE_NDJSON), "--group-by", "warp", "--count"],
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("WARP", result.output)
        self.assertIn("COUNT", result.output)

    def test_analyze_count_top(self):
        """Test analyze with --count --top."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(REG_TRACE_NDJSON),
                "--group-by",
                "warp",
                "--count",
                "--top",
                "3",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        lines = [line for line in result.output.strip().split("\n") if line]
        self.assertEqual(len(lines), 4)  # header + 3 data rows

    def test_analyze_count_json_format(self):
        """Test analyze --count with JSON format."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(REG_TRACE_NDJSON),
                "--group-by",
                "warp",
                "--count",
                "--format",
                "json",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        import json

        data = json.loads(result.output)
        self.assertIsInstance(data, dict)

    def test_analyze_count_csv_format(self):
        """Test analyze --count with CSV format."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(REG_TRACE_NDJSON),
                "--group-by",
                "warp",
                "--count",
                "--format",
                "csv",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("warp,count", result.output)

    def test_analyze_count_no_header(self):
        """Test analyze --count with --no-header."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(REG_TRACE_NDJSON),
                "--group-by",
                "warp",
                "--count",
                "--no-header",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        # Should not contain header row
        self.assertNotIn("WARP", result.output)
        self.assertNotIn("COUNT", result.output)

    def test_analyze_group_by_warp_shows_summary(self):
        """Test analyze --group-by warp shows warp summary."""
        result = self.runner.invoke(
            main,
            ["query", str(REG_TRACE_NDJSON), "--group-by", "warp", "--tail", "3"],
        )
        self.assertEqual(result.exit_code, 0)
        # Should have warp summary section
        self.assertIn("Warp Summary", result.output)
        self.assertIn("Total warps observed:", result.output)
        self.assertIn("Completed (EXIT):", result.output)
        self.assertIn("In-progress:", result.output)
        self.assertIn("Missing (never seen):", result.output)

    def test_analyze_group_by_warp_json_has_summary(self):
        """Test analyze --group-by warp with JSON format includes warp_summary."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(REG_TRACE_NDJSON),
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

        data = json.loads(result.output)
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
            ["query", str(REG_TRACE_NDJSON), "--group-by", "sass", "--head", "2"],
        )
        self.assertEqual(result.exit_code, 0)
        # Should NOT have warp summary section
        self.assertNotIn("Warp Summary", result.output)

    def test_analyze_group_by_warp_csv_no_summary(self):
        """Test analyze --group-by warp with CSV format does not include summary."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(REG_TRACE_NDJSON),
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
        self.assertNotIn("Warp Summary", result.output)

    def test_analyze_all_records(self):
        """Test analyze with --all-lines flag to show all records."""
        # First count total records
        result_all = self.runner.invoke(
            main, ["query", str(REG_TRACE_NDJSON), "--all-lines", "--no-header"]
        )
        self.assertEqual(result_all.exit_code, 0)
        all_lines = [line for line in result_all.output.strip().split("\n") if line]

        # Compare with head 10
        result_head = self.runner.invoke(
            main, ["query", str(REG_TRACE_NDJSON), "--head", "10", "--no-header"]
        )
        head_lines = [line for line in result_head.output.strip().split("\n") if line]

        # All should have >= 10 records (or same if file has exactly 10)
        self.assertGreaterEqual(len(all_lines), len(head_lines))

    def test_analyze_all_short_option(self):
        """Test analyze with -a short option for --all-lines."""
        result = self.runner.invoke(
            main, ["query", str(REG_TRACE_NDJSON), "-a", "--no-header"]
        )
        self.assertEqual(result.exit_code, 0)

    def test_analyze_format_ndjson(self):
        """Test analyze with NDJSON output format."""
        result = self.runner.invoke(
            main, ["query", str(REG_TRACE_NDJSON), "--format", "ndjson", "--head", "3"]
        )
        self.assertEqual(result.exit_code, 0)
        lines = result.output.strip().split("\n")
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
            ["query", str(REG_TRACE_NDJSON), "--format", "json", "--head", "1"],
        )
        self.assertEqual(result.exit_code, 0)
        import json

        records = json.loads(result.output)
        if records and "pc" in records[0]:
            pc_value = records[0]["pc"]
            # pc is now a hex string like "0x0" or "0x10"
            # Filter using the hex string directly
            result_hex = self.runner.invoke(
                main,
                [
                    "query",
                    str(REG_TRACE_NDJSON),
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
            ["query", str(REG_TRACE_NDJSON), "--fields", "*", "--head", "1"],
        )
        self.assertEqual(result.exit_code, 0)
        # Should show more than just default fields (warp, pc, sass)
        header = result.output.strip().split("\n")[0]
        # Check that non-default fields are included (like type, time, etc.)
        # At minimum should have WARP, PC, SASS
        self.assertIn("WARP", header)
        self.assertIn("PC", header)

    def test_analyze_fields_all_keyword(self):
        """Test analyze with --fields 'all' to show all fields."""
        result = self.runner.invoke(
            main,
            ["query", str(REG_TRACE_NDJSON), "--fields", "all", "--head", "1"],
        )
        self.assertEqual(result.exit_code, 0)
        header = result.output.strip().split("\n")[0]
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
                    str(REG_TRACE_NDJSON),
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
                    str(REG_TRACE_NDJSON),
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
            ["query", str(REG_TRACE_NDJSON), "--group-by", "warp", "--all-lines"],
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("=== Group:", result.output)

    def test_analyze_count_ndjson_format(self):
        """Test analyze --count with NDJSON format."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(REG_TRACE_NDJSON),
                "--group-by",
                "warp",
                "--count",
                "--format",
                "ndjson",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        import json

        lines = result.output.strip().split("\n")
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
                    str(REG_TRACE_NDJSON),
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
                str(REG_TRACE_NDJSON),
                "--head",
                "3",
                "--compress",
            ],
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--compress requires --output", result.output)

    def test_query_ndjson_all_fields_heterogeneous(self):
        """Test --fields all --format ndjson preserves all fields from heterogeneous records.

        This tests the fix for the bug where fields like 'uregs' that only
        appear in some records were missing from the NDJSON output.
        """
        import json
        import os
        import tempfile

        # Create test file with heterogeneous records (different field sets)
        test_records = [
            {"warp": 0, "pc": "0x0", "sass": "NOP ;", "regs": [1, 2]},
            {"warp": 0, "pc": "0x20", "sass": "ULDC ;", "regs": [], "uregs": [10, 20]},
            {"warp": 0, "pc": "0x30", "sass": "LDG ;", "addrs": [100], "values": [200]},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "hetero.ndjson")
            output_path = os.path.join(tmpdir, "output.ndjson")

            # Write test input
            with open(input_path, "w") as f:
                for record in test_records:
                    f.write(json.dumps(record) + "\n")

            # Run query with --fields all --format ndjson
            result = self.runner.invoke(
                main,
                [
                    "query",
                    input_path,
                    "--fields",
                    "all",
                    "--format",
                    "ndjson",
                    "--all-lines",
                    "--output",
                    output_path,
                ],
            )
            self.assertEqual(result.exit_code, 0)

            # Read back and verify all records preserved
            with open(output_path) as f:
                output_lines = f.read().strip().split("\n")

            self.assertEqual(len(output_lines), 3)

            # Parse each line
            out_records = [json.loads(line) for line in output_lines]

            # First record should NOT have 'uregs' (it's not in the original)
            self.assertNotIn("uregs", out_records[0])
            self.assertIn("regs", out_records[0])

            # Second record MUST have 'uregs' (this was the bug)
            self.assertIn("uregs", out_records[1])
            self.assertEqual(out_records[1]["uregs"], [10, 20])

            # Third record should have 'addrs' and 'values'
            self.assertIn("addrs", out_records[2])
            self.assertIn("values", out_records[2])

    def test_query_ndjson_all_fields_preserves_original_fields(self):
        """Test --fields all with NDJSON preserves each record's original fields exactly.

        Records should only contain the fields they originally had - no extra fields
        should be added even if other records have them.
        """
        import json
        import os
        import tempfile

        test_records = [
            {"warp": 0, "pc": "0x0", "regs": [1]},
            {"warp": 1, "pc": "0x10", "uregs": [2]},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.ndjson")
            output_path = os.path.join(tmpdir, "output.ndjson")

            with open(input_path, "w") as f:
                for record in test_records:
                    f.write(json.dumps(record) + "\n")

            result = self.runner.invoke(
                main,
                [
                    "query",
                    input_path,
                    "--fields",
                    "all",
                    "--format",
                    "ndjson",
                    "--all-lines",
                    "--output",
                    output_path,
                ],
            )
            self.assertEqual(result.exit_code, 0)

            with open(output_path) as f:
                output_lines = f.read().strip().split("\n")

            out_records = [json.loads(line) for line in output_lines]

            # First record should NOT have 'uregs' added
            self.assertIn("regs", out_records[0])
            self.assertNotIn("uregs", out_records[0])

            # Second record should NOT have 'regs' added
            self.assertIn("uregs", out_records[1])
            self.assertNotIn("regs", out_records[1])


class TestQueryKernelEvents(BaseValidationTest):
    """Tests for query command with kernel events files."""

    def setUp(self):
        super().setUp()
        self.runner = CliRunner()

    def test_kernel_events_default(self):
        """Test querying kernel events file with default options."""
        result = self.runner.invoke(main, ["query", str(KERNEL_EVENTS_NDJSON)])
        self.assertEqual(result.exit_code, 0, result.output)
        # Should display kernel_launch records, not callstack_def
        self.assertNotIn("callstack_def", result.output)
        self.assertIn("kernel_launch", result.output)
        # Should have caller field from callstack resolution
        self.assertIn("CALLER", result.output)

    def test_kernel_events_no_callstack_def_in_output(self):
        """Test that callstack_def records are excluded from output."""
        result = self.runner.invoke(
            main,
            ["query", str(KERNEL_EVENTS_NDJSON), "--all-lines", "--format", "json"],
        )
        self.assertEqual(result.exit_code, 0)
        import json

        records = json.loads(result.output)
        for r in records:
            self.assertNotEqual(r.get("type"), "callstack_def")
        # Output count must equal the number of kernel_launch records in
        # the fixture — fixture changes should not silently break this test.
        expected = count_records_of_type(KERNEL_EVENTS_NDJSON, "kernel_launch")
        self.assertEqual(len(records), expected)

    def test_kernel_events_filter_stream_id(self):
        """Test filtering kernel events by stream_id."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(KERNEL_EVENTS_NDJSON),
                "--filter",
                "stream_id=0",
                "--all-lines",
                "--format",
                "json",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        import json

        records = json.loads(result.output)
        for r in records:
            self.assertEqual(r["stream_id"], 0)

    def test_kernel_events_group_by_kernel_name_count(self):
        """Test grouping kernel events by kernel_name with count."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(KERNEL_EVENTS_NDJSON),
                "--group-by",
                "kernel_checksum",
                "--count",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("KERNEL_CHECKSUM", result.output)
        self.assertIn("COUNT", result.output)

    def test_kernel_events_caller_field_visible(self):
        """Test that caller field is visible in output."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(KERNEL_EVENTS_NDJSON),
                "--fields",
                "kernel_launch_id,caller",
                "--head",
                "3",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("CALLER", result.output)
        self.assertIn("test_matmul", result.output)

    def test_kernel_events_fields_filter(self):
        """Test custom fields selection for kernel events."""
        result = self.runner.invoke(
            main,
            [
                "query",
                str(KERNEL_EVENTS_NDJSON),
                "--fields",
                "kernel_launch_id,grid,block,stream_id",
                "--head",
                "3",
            ],
        )
        self.assertEqual(result.exit_code, 0)
        self.assertIn("KERNEL_LAUNCH_ID", result.output)
        self.assertIn("GRID", result.output)
        self.assertIn("STREAM_ID", result.output)


if __name__ == "__main__":
    unittest.main()
