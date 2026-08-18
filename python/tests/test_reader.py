# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Unit tests for TraceReader class and related functions.
"""

import json
import types
import unittest

from cutracer.query import parse_filter_expr, select_records, TraceReader
from tests.test_base import (
    BaseValidationTest,
    count_records_of_type,
    KERNEL_EVENTS_NDJSON,
    REG_TRACE_NDJSON,
    REG_TRACE_NDJSON_RECORD_COUNT,
    REG_TRACE_NDJSON_ZST,
    REG_TRACE_NDJSON_ZST_RECORD_COUNT,
)


class TestParseFilterExpr(unittest.TestCase):
    """Tests for parse_filter_expr function."""

    def test_parse_filter_int_value(self):
        """Test parsing filter with integer value."""
        pred = parse_filter_expr("warp=24")
        self.assertTrue(pred({"warp": 24}))
        self.assertFalse(pred({"warp": 25}))
        # String "24" matches int 24 for backward compatibility
        self.assertTrue(pred({"warp": "24"}))

    def test_parse_filter_string_value(self):
        """Test parsing filter with string value."""
        pred = parse_filter_expr("type=mem_trace")
        self.assertTrue(pred({"type": "mem_trace"}))
        self.assertFalse(pred({"type": "reg_trace"}))

    def test_parse_filter_missing_field(self):
        """Test filter returns False for missing field."""
        pred = parse_filter_expr("warp=0")
        self.assertFalse(pred({"type": "reg_trace"}))

    def test_parse_filter_with_spaces(self):
        """Test filter with spaces around = is handled."""
        pred = parse_filter_expr(" warp = 24 ")
        self.assertTrue(pred({"warp": 24}))

    def test_parse_filter_invalid_no_equals(self):
        """Test that filter without = raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            parse_filter_expr("warp")
        self.assertIn("Invalid filter expression", str(ctx.exception))

    def test_parse_filter_invalid_empty_field(self):
        """Test that filter with empty field raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            parse_filter_expr("=24")
        self.assertIn("empty", str(ctx.exception))

    def test_parse_filter_semicolon_separated_and(self):
        """Test semicolon-separated multi-condition filter (AND logic)."""
        pred = parse_filter_expr("warp=0;pc=100")
        self.assertTrue(pred({"warp": 0, "pc": 100}))
        self.assertFalse(pred({"warp": 0, "pc": 200}))
        self.assertFalse(pred({"warp": 1, "pc": 100}))

    def test_parse_filter_semicolon_separated_hex(self):
        """Test semicolon-separated filter with hex values."""
        pred = parse_filter_expr("pc=0x64;warp=1")
        self.assertTrue(pred({"pc": 100, "warp": 1}))
        self.assertFalse(pred({"pc": 100, "warp": 2}))
        self.assertFalse(pred({"pc": 200, "warp": 1}))

    def test_parse_filter_semicolon_separated_three_conditions(self):
        """Test three semicolon-separated conditions."""
        pred = parse_filter_expr("warp=0;pc=100;sass=MOV")
        self.assertTrue(pred({"warp": 0, "pc": 100, "sass": "MOV"}))
        self.assertFalse(pred({"warp": 0, "pc": 100, "sass": "ADD"}))

    def test_parse_filter_semicolon_separated_with_spaces(self):
        """Test semicolon-separated filter with spaces around semicolons."""
        pred = parse_filter_expr(" warp=0 ; pc=100 ")
        self.assertTrue(pred({"warp": 0, "pc": 100}))
        self.assertFalse(pred({"warp": 1, "pc": 100}))

    def test_parse_filter_semicolon_separated_invalid_part(self):
        """Test semicolon-separated filter with an invalid part raises ValueError."""
        with self.assertRaises(ValueError):
            parse_filter_expr("warp=0;invalid")

    def test_parse_filter_empty_expression(self):
        """Test empty filter expression raises ValueError."""
        with self.assertRaises(ValueError):
            parse_filter_expr("")

    def test_parse_filter_list_value(self):
        """Test parsing filter with JSON list value (e.g., cta=[0,0,0])."""
        pred = parse_filter_expr("cta=[0,0,0]")
        self.assertTrue(pred({"cta": [0, 0, 0]}))
        self.assertFalse(pred({"cta": [1, 0, 0]}))
        self.assertFalse(pred({"cta": [0, 0]}))

    def test_parse_filter_list_value_with_spaces(self):
        """Test parsing filter with JSON list value containing spaces."""
        pred = parse_filter_expr("cta=[0, 0, 0]")
        self.assertTrue(pred({"cta": [0, 0, 0]}))

    def test_parse_filter_list_missing_field(self):
        """Test list filter returns False for missing field."""
        pred = parse_filter_expr("cta=[0,0,0]")
        self.assertFalse(pred({"warp": 0}))

    def test_parse_filter_semicolon_with_list(self):
        """Test semicolon-separated filter combining list and int values."""
        pred = parse_filter_expr("cta=[0,0,0];warp=1")
        self.assertTrue(pred({"cta": [0, 0, 0], "warp": 1}))
        self.assertFalse(pred({"cta": [0, 0, 0], "warp": 2}))
        self.assertFalse(pred({"cta": [1, 0, 0], "warp": 1}))

    def test_parse_filter_single_still_works(self):
        """Test that single condition still works after refactor."""
        pred = parse_filter_expr("warp=24")
        self.assertTrue(pred({"warp": 24}))
        self.assertFalse(pred({"warp": 25}))


class TestSelectRecords(unittest.TestCase):
    """Tests for select_records function."""

    def test_select_head_default(self):
        """Test default head selection (10 records)."""
        records = iter([{"i": i} for i in range(20)])
        result = select_records(records)
        self.assertEqual(len(result), 10)
        self.assertEqual(result[0]["i"], 0)
        self.assertEqual(result[9]["i"], 9)

    def test_select_head_custom(self):
        """Test custom head selection."""
        records = iter([{"i": i} for i in range(20)])
        result = select_records(records, head=5)
        self.assertEqual(len(result), 5)
        self.assertEqual(result[4]["i"], 4)

    def test_select_tail(self):
        """Test tail selection."""
        records = iter([{"i": i} for i in range(20)])
        result = select_records(records, tail=5)
        self.assertEqual(len(result), 5)
        self.assertEqual(result[0]["i"], 15)
        self.assertEqual(result[4]["i"], 19)

    def test_select_tail_overrides_head(self):
        """Test that tail overrides head when both specified."""
        records = iter([{"i": i} for i in range(20)])
        result = select_records(records, head=3, tail=5)
        self.assertEqual(len(result), 5)
        self.assertEqual(result[0]["i"], 15)

    def test_select_head_zero(self):
        """Test head=0 returns empty list."""
        records = iter([{"i": i} for i in range(10)])
        result = select_records(records, head=0)
        self.assertEqual(result, [])

    def test_select_tail_zero(self):
        """Test tail=0 returns empty list."""
        records = iter([{"i": i} for i in range(10)])
        result = select_records(records, tail=0)
        self.assertEqual(result, [])

    def test_select_head_more_than_available(self):
        """Test head larger than available records."""
        records = iter([{"i": i} for i in range(5)])
        result = select_records(records, head=20)
        self.assertEqual(len(result), 5)

    def test_select_tail_more_than_available(self):
        """Test tail larger than available records."""
        records = iter([{"i": i} for i in range(5)])
        result = select_records(records, tail=20)
        self.assertEqual(len(result), 5)

    def test_select_empty_iterator(self):
        """Test selection from empty iterator."""
        records = iter([])
        result = select_records(records, head=10)
        self.assertEqual(result, [])


class TestTraceReaderInit(BaseValidationTest):
    """Tests for TraceReader initialization."""

    def test_init_with_ndjson_file(self):
        """Test initialization with a plain NDJSON file."""
        reader = TraceReader(REG_TRACE_NDJSON)
        self.assertEqual(reader.file_path, REG_TRACE_NDJSON)
        self.assertEqual(reader.compression, "none")

    def test_init_with_zst_file(self):
        """Test initialization with a Zstd-compressed NDJSON file."""
        reader = TraceReader(REG_TRACE_NDJSON_ZST)
        self.assertEqual(reader.file_path, REG_TRACE_NDJSON_ZST)
        self.assertEqual(reader.compression, "zstd")

    def test_init_with_string_path(self):
        """Test initialization with a string path."""
        reader = TraceReader(str(REG_TRACE_NDJSON))
        self.assertEqual(reader.file_path, REG_TRACE_NDJSON)

    def test_init_with_nonexistent_file(self):
        """Test that FileNotFoundError is raised for non-existent file."""
        with self.assertRaises(FileNotFoundError):
            TraceReader("/nonexistent/path/file.ndjson")


class TestTraceReaderIterRecords(BaseValidationTest):
    """Tests for TraceReader.iter_records() method."""

    def test_iter_records_ndjson(self):
        """Test iterating over records in a plain NDJSON file."""
        reader = TraceReader(REG_TRACE_NDJSON)
        records = list(reader.iter_records())

        self.assertEqual(len(records), REG_TRACE_NDJSON_RECORD_COUNT)

        # Verify first record structure
        first_record = records[0]
        self.assertIn("type", first_record)
        self.assertIn("warp", first_record)
        self.assertIn("sass", first_record)
        self.assertEqual(first_record["type"], "reg_trace")

    def test_iter_records_zst(self):
        """Test iterating over records in a Zstd-compressed NDJSON file."""
        reader = TraceReader(REG_TRACE_NDJSON_ZST)
        records = list(reader.iter_records())

        self.assertEqual(len(records), REG_TRACE_NDJSON_ZST_RECORD_COUNT)

    def test_iter_records_zst_has_valid_structure(self):
        """Test that Zstd compressed file records have valid structure."""
        reader = TraceReader(REG_TRACE_NDJSON_ZST)
        records = list(reader.iter_records())

        self.assertGreater(len(records), 0)

        # Verify record structure
        first_record = records[0]
        self.assertIn("type", first_record)
        self.assertIn("warp", first_record)
        self.assertIn("sass", first_record)

    def test_iter_records_is_generator(self):
        """Test that iter_records returns a generator (lazy evaluation)."""
        reader = TraceReader(REG_TRACE_NDJSON)
        result = reader.iter_records()

        # Should be a generator, not a list
        self.assertIsInstance(result, types.GeneratorType)

    def test_iter_records_can_iterate_multiple_times(self):
        """Test that iter_records can be called multiple times."""
        reader = TraceReader(REG_TRACE_NDJSON)

        # First iteration
        records1 = list(reader.iter_records())
        # Second iteration
        records2 = list(reader.iter_records())

        self.assertEqual(len(records1), len(records2))
        self.assertEqual(records1[0], records2[0])


class TestTraceReaderEdgeCases(BaseValidationTest):
    """Edge case tests for TraceReader."""

    def test_empty_file(self):
        """Test reading an empty file."""
        filepath = self.create_temp_file("empty.ndjson", "")

        reader = TraceReader(filepath)
        records = list(reader.iter_records())

        self.assertEqual(len(records), 0)

    def test_single_record_file(self):
        """Test reading a file with a single record."""
        content = '{"type": "reg_trace", "warp": 0, "sass": "NOP ;"}\n'
        filepath = self.create_temp_file("single.ndjson", content)

        reader = TraceReader(filepath)
        records = list(reader.iter_records())

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["type"], "reg_trace")

    def test_empty_lines_skipped(self):
        """Test that empty lines in the file are skipped."""
        content = (
            '{"type": "reg_trace", "warp": 0}\n\n{"type": "reg_trace", "warp": 1}\n'
        )
        filepath = self.create_temp_file("test.ndjson", content)

        reader = TraceReader(filepath)
        records = list(reader.iter_records())

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["warp"], 0)
        self.assertEqual(records[1]["warp"], 1)

    def test_invalid_json_raises_error(self):
        """Test that invalid JSON raises JSONDecodeError."""
        content = '{"type": "reg_trace"}\n{invalid json}\n'
        filepath = self.create_temp_file("invalid.ndjson", content)

        reader = TraceReader(filepath)
        iterator = reader.iter_records()

        # First record should work
        first = next(iterator)
        self.assertEqual(first["type"], "reg_trace")

        # Second record should raise
        with self.assertRaises(json.JSONDecodeError):
            next(iterator)


class TestIntegration(BaseValidationTest):
    """Integration tests combining TraceReader with filter and select."""

    def test_filter_and_select(self):
        """Test combining TraceReader with filter and select_records."""
        reader = TraceReader(REG_TRACE_NDJSON)
        pred = parse_filter_expr("warp=0")

        # Filter and select first 5
        filtered = (r for r in reader.iter_records() if pred(r))
        result = select_records(filtered, head=5)

        self.assertLessEqual(len(result), 5)
        for record in result:
            self.assertEqual(record["warp"], 0)


class TestTraceReaderKernelEvents(BaseValidationTest):
    """Tests for TraceReader with kernel events files."""

    def test_callstack_def_records_are_filtered(self):
        """Test that callstack_def records are not yielded."""
        reader = TraceReader(KERNEL_EVENTS_NDJSON)
        records = list(reader.iter_records())

        types = {r.get("type") for r in records}
        self.assertNotIn("callstack_def", types)
        self.assertIn("kernel_launch", types)

    def test_kernel_launch_count(self):
        """Test that correct number of kernel_launch records are yielded."""
        reader = TraceReader(KERNEL_EVENTS_NDJSON)
        records = list(reader.iter_records())

        # callstack_def records must be filtered out — count yielded
        # records against the actual number of kernel_launch lines in
        # the fixture (decoupled from any specific fixture size).
        expected = count_records_of_type(KERNEL_EVENTS_NDJSON, "kernel_launch")
        self.assertEqual(len(records), expected)

    def test_caller_field_injected(self):
        """Test that caller field is injected from callstack resolution."""
        reader = TraceReader(KERNEL_EVENTS_NDJSON)
        records = list(reader.iter_records())

        # All kernel_launch records have callstack_id, so all should have caller
        for record in records:
            self.assertIn("caller", record)
            self.assertIsInstance(record["caller"], str)
            self.assertGreater(len(record["caller"]), 0)

    def test_caller_matches_callstack(self):
        """Test that injected caller matches the correct callstack."""
        reader = TraceReader(KERNEL_EVENTS_NDJSON)
        records = list(reader.iter_records())

        # First record has callstack_id "aaa111" -> last frame is test_matmul_basic
        first = records[0]
        self.assertIn("test_matmul_basic", first["caller"])

        # Third record (launch_id=2) has callstack_id "bbb222" -> test_matmul_large
        third = records[2]
        self.assertIn("test_matmul_large", third["caller"])

    def test_kernel_launch_fields_preserved(self):
        """Test that all kernel_launch fields are preserved."""
        reader = TraceReader(KERNEL_EVENTS_NDJSON)
        records = list(reader.iter_records())

        first = records[0]
        self.assertEqual(first["type"], "kernel_launch")
        self.assertEqual(first["kernel_launch_id"], 0)
        self.assertIn("kernel_name", first)
        self.assertIn("grid", first)
        self.assertIn("block", first)
        self.assertIn("stream_id", first)
        self.assertIn("kernel_checksum", first)

    def test_inline_callstack_not_overwritten(self):
        """Test that records with existing caller field are not overwritten."""
        content = (
            '{"callstack_id":"x1","frames":["a.py:1 in f"],"type":"callstack_def"}\n'
            '{"type":"kernel_launch","callstack_id":"x1","caller":"already_set","kernel_launch_id":0}\n'
        )
        filepath = self.create_temp_file("inline_caller.ndjson", content)
        reader = TraceReader(filepath)
        records = list(reader.iter_records())

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["caller"], "already_set")

    def test_caller_not_injected_into_non_kernel_launch_records(self):
        """Test that caller is only injected into kernel_launch records.

        Even if another record type happens to carry a callstack_id, we
        must not synthesize a 'caller' field on it — that would silently
        change its schema.
        """
        content = (
            '{"callstack_id":"x1","frames":["a.py:1 in f"],"type":"callstack_def"}\n'
            '{"type":"some_other_event","callstack_id":"x1","payload":42}\n'
            '{"type":"kernel_launch","callstack_id":"x1","kernel_launch_id":0}\n'
        )
        filepath = self.create_temp_file("non_launch_callstack.ndjson", content)
        reader = TraceReader(filepath)
        records = list(reader.iter_records())

        self.assertEqual(len(records), 2)
        non_launch = next(r for r in records if r["type"] == "some_other_event")
        launch = next(r for r in records if r["type"] == "kernel_launch")
        self.assertNotIn("caller", non_launch)
        self.assertIn("caller", launch)


if __name__ == "__main__":
    unittest.main()
