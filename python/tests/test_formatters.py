# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Unit tests for formatters module.
"""

import json
import unittest

from cutracer.query.formatters import (
    DEFAULT_FIELDS,
    format_records_csv,
    format_records_json,
    format_records_table,
    format_value,
    get_display_fields,
)


class TestFormatValue(unittest.TestCase):
    """Tests for format_value function."""

    def test_format_value_none(self):
        """Test None returns empty string."""
        self.assertEqual(format_value(None), "")

    def test_format_value_bool_true(self):
        """Test True returns lowercase 'true'."""
        self.assertEqual(format_value(True), "true")

    def test_format_value_bool_false(self):
        """Test False returns lowercase 'false'."""
        self.assertEqual(format_value(False), "false")

    def test_format_value_int(self):
        """Test int returns string representation."""
        self.assertEqual(format_value(42), "42")

    def test_format_value_str(self):
        """Test string returns as-is."""
        self.assertEqual(format_value("hello"), "hello")

    def test_format_value_list(self):
        """Test list returns bracketed comma-separated values."""
        self.assertEqual(format_value([1, 2, 3]), "[1,2,3]")

    def test_format_value_empty_list(self):
        """Test empty list returns empty brackets."""
        self.assertEqual(format_value([]), "[]")

    def test_format_value_dict(self):
        """Test dict returns JSON string."""
        result = format_value({"a": 1})
        self.assertEqual(result, '{"a":1}')


class TestGetDisplayFields(unittest.TestCase):
    """Tests for get_display_fields function."""

    def test_get_display_fields_default(self):
        """Test default returns all fields when none requested."""
        records = [{"warp": 0, "pc": "0x100", "sass": "NOP ;", "extra": "data"}]
        fields = get_display_fields(records)
        # Default behavior now returns all fields from the record
        self.assertEqual(fields, ["warp", "pc", "sass", "extra"])

    def test_get_display_fields_requested(self):
        """Test user-specified fields are used."""
        records = [{"warp": 0, "pc": "0x100", "sass": "NOP ;"}]
        fields = get_display_fields(records, "warp,sass")
        self.assertEqual(fields, ["warp", "sass"])

    def test_get_display_fields_with_spaces(self):
        """Test fields with spaces are trimmed."""
        records = [{"warp": 0, "pc": "0x100"}]
        fields = get_display_fields(records, " warp , pc ")
        self.assertEqual(fields, ["warp", "pc"])

    def test_get_display_fields_empty_records(self):
        """Test empty records returns DEFAULT_FIELDS."""
        fields = get_display_fields([])
        self.assertEqual(fields, DEFAULT_FIELDS)

    def test_get_display_fields_missing_default(self):
        """Test default returns all record fields."""
        records = [{"warp": 0, "custom": "value"}]
        fields = get_display_fields(records)
        # Default behavior now returns all fields from the record
        self.assertEqual(fields, ["warp", "custom"])

    def test_get_display_fields_all_union(self):
        """Test --fields all returns union of fields from all records.

        This tests the fix for the bug where fields like 'uregs' that only
        appear in some records were missing from the output.
        """
        records = [
            {"warp": 0, "pc": "0x0", "regs": []},
            {"warp": 0, "pc": "0x20", "regs": [], "uregs": [1, 2]},
            {"warp": 0, "pc": "0x30", "addrs": [100], "values": [200]},
        ]
        fields = get_display_fields(records, "all")

        # Should include fields from all records
        self.assertIn("uregs", fields)
        self.assertIn("addrs", fields)
        self.assertIn("values", fields)

        # First record's fields should come first (preserving order)
        self.assertEqual(fields[:3], ["warp", "pc", "regs"])

    def test_get_display_fields_star_union(self):
        """Test --fields '*' also returns union of fields."""
        records = [
            {"warp": 0, "pc": "0x0"},
            {"warp": 0, "pc": "0x20", "uregs": [1, 2]},
        ]
        fields = get_display_fields(records, "*")

        self.assertIn("uregs", fields)
        self.assertEqual(fields[:2], ["warp", "pc"])

    def test_get_display_fields_all_with_spaces(self):
        """Test --fields ' all ' with whitespace is handled."""
        records = [
            {"warp": 0, "pc": "0x0"},
            {"warp": 0, "pc": "0x20", "uregs": [1, 2]},
        ]
        fields = get_display_fields(records, "  all  ")

        self.assertIn("uregs", fields)

    def test_get_display_fields_all_single_record(self):
        """Test --fields all with single record returns that record's fields."""
        records = [{"warp": 0, "pc": "0x0", "sass": "NOP ;"}]
        fields = get_display_fields(records, "all")

        self.assertEqual(fields, ["warp", "pc", "sass"])


class TestFormatRecordsTable(unittest.TestCase):
    """Tests for format_records_table function."""

    def test_format_table_empty_records(self):
        """Test empty records returns message."""
        result = format_records_table([], ["warp", "pc"])
        self.assertEqual(result, "No records found.")

    def test_format_table_with_header(self):
        """Test table with header row."""
        records = [{"warp": 0, "pc": "0x100"}]
        result = format_records_table(records, ["warp", "pc"], show_header=True)
        lines = result.split("\n")
        self.assertEqual(len(lines), 2)
        self.assertIn("WARP", lines[0])
        self.assertIn("PC", lines[0])
        self.assertIn("0", lines[1])
        self.assertIn("0x100", lines[1])

    def test_format_table_without_header(self):
        """Test table without header row."""
        records = [{"warp": 0, "pc": "0x100"}]
        result = format_records_table(records, ["warp", "pc"], show_header=False)
        lines = result.split("\n")
        self.assertEqual(len(lines), 1)
        self.assertNotIn("WARP", result)

    def test_format_table_column_alignment(self):
        """Test columns are aligned by width."""
        records = [
            {"warp": 0, "sass": "NOP ;"},
            {"warp": 123, "sass": "EXIT ;"},
        ]
        result = format_records_table(records, ["warp", "sass"], show_header=True)
        lines = result.split("\n")
        # All lines should have consistent column positions
        self.assertEqual(len(lines), 3)  # header + 2 data rows

    def test_format_table_missing_field(self):
        """Test missing field shows empty string."""
        records = [{"warp": 0}]  # no 'pc' field
        result = format_records_table(records, ["warp", "pc"], show_header=False)
        self.assertIn("0", result)


class TestFormatRecordsJson(unittest.TestCase):
    """Tests for format_records_json function."""

    def test_format_json_empty_records(self):
        """Test empty records returns empty array."""
        result = format_records_json([], ["warp"])
        self.assertEqual(result, "[]")

    def test_format_json_single_record(self):
        """Test single record formatting."""
        records = [{"warp": 0, "pc": "0x100", "extra": "ignored"}]
        result = format_records_json(records, ["warp", "pc"])
        parsed = json.loads(result)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["warp"], 0)
        self.assertEqual(parsed[0]["pc"], "0x100")
        self.assertNotIn("extra", parsed[0])

    def test_format_json_multiple_records(self):
        """Test multiple records formatting."""
        records = [{"warp": 0}, {"warp": 1}]
        result = format_records_json(records, ["warp"])
        parsed = json.loads(result)
        self.assertEqual(len(parsed), 2)

    def test_format_json_filters_fields(self):
        """Test only requested fields are included."""
        records = [{"warp": 0, "pc": "0x100", "sass": "NOP ;"}]
        result = format_records_json(records, ["warp"])
        parsed = json.loads(result)
        self.assertEqual(list(parsed[0].keys()), ["warp"])


class TestFormatRecordsCsv(unittest.TestCase):
    """Tests for format_records_csv function."""

    def test_format_csv_empty_records(self):
        """Test empty records returns empty string."""
        result = format_records_csv([], ["warp"])
        self.assertEqual(result, "")

    def test_format_csv_with_header(self):
        """Test CSV with header row."""
        records = [{"warp": 0, "pc": "0x100"}]
        result = format_records_csv(records, ["warp", "pc"], show_header=True)
        lines = result.split("\n")
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0], "warp,pc")
        self.assertEqual(lines[1], "0,0x100")

    def test_format_csv_without_header(self):
        """Test CSV without header row."""
        records = [{"warp": 0, "pc": "0x100"}]
        result = format_records_csv(records, ["warp", "pc"], show_header=False)
        lines = result.split("\n")
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0], "0,0x100")

    def test_format_csv_escapes_comma(self):
        """Test CSV properly escapes values with commas."""
        records = [{"sass": "MOV R0, R1 ;"}]
        result = format_records_csv(records, ["sass"], show_header=False)
        # CSV should quote the value containing comma
        self.assertIn('"', result)

    def test_format_csv_multiple_records(self):
        """Test CSV with multiple records."""
        records = [{"warp": 0}, {"warp": 1}, {"warp": 2}]
        result = format_records_csv(records, ["warp"], show_header=True)
        lines = result.split("\n")
        self.assertEqual(len(lines), 4)  # 1 header + 3 data


if __name__ == "__main__":
    unittest.main()
