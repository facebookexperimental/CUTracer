# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Unit tests for JSON validator module.
"""

from __future__ import annotations

import json
import unittest
from typing import Any

import zstandard as zstd
from cutracer.validation.json_validator import (
    JsonValidationError,
    validate_json_schema,
    validate_json_syntax,
    validate_json_trace,
)
from tests.test_base import (
    BaseValidationTest,
    INVALID_SCHEMA_NDJSON,
    INVALID_SYNTAX_NDJSON,
    REG_TRACE_NDJSON,
    REG_TRACE_NDJSON_RECORD_COUNT,
    REG_TRACE_NDJSON_ZST,
    REG_TRACE_NDJSON_ZST_RECORD_COUNT,
)


def _capture_records(count: int) -> list[dict[str, Any]]:
    with REG_TRACE_NDJSON.open() as source:
        event = json.loads(source.readline())
    header = {
        "type": "kernel_metadata",
        "mangled_name": "kernel",
        "unmangled_name": "kernel",
        "kernel_checksum": "1",
        "func_addr": "0x0",
        "nregs": 1,
        "shmem_static": 0,
        "shmem_dynamic": 0,
        "grid": [1, 1, 1],
        "block": [32, 1, 1],
    }
    footer = {
        "type": "capture_completion",
        "version": 1,
        "grid_launch_id": event["grid_launch_id"],
        "status": "complete",
        "kernel_completed": True,
        "channel_drained": True,
        "trace_records_written": count,
        "dropped_records": 0,
        "errors": [],
    }
    return [header, *({**event, "trace_index": i} for i in range(count)), footer]


class JsonValidatorTest(BaseValidationTest):
    """Core tests for JSON validator functions."""

    def test_metadata_and_completion_are_not_counted_as_instructions(self) -> None:
        for count, compressed in ((0, False), (2, False), (0, True), (2, True)):
            with self.subTest(count=count, compressed=compressed):
                content = (
                    "\n".join(json.dumps(row) for row in _capture_records(count)) + "\n"
                )
                suffix = ".ndjson.zst" if compressed else ".ndjson"
                path = self.temp_dir / f"completion{suffix}"
                if compressed:
                    path.write_bytes(zstd.ZstdCompressor().compress(content.encode()))
                else:
                    path.write_text(content)
                self.assertEqual(validate_json_syntax(path), (count + 2, []))
                result = validate_json_trace(path)
                self.assertTrue(result["valid"], result["errors"])
                self.assertEqual(result["record_count"], count)
                self.assertEqual(result["message_type"], "reg_trace" if count else None)
                self.assertEqual(
                    result["compression"], "zstd" if compressed else "none"
                )

    def test_completion_schema_is_checked_even_though_it_is_not_counted(self) -> None:
        records = _capture_records(1)
        del records[-1]["channel_drained"]
        path = self.create_temp_file(
            "invalid_footer.ndjson",
            "\n".join(json.dumps(row) for row in records) + "\n",
        )
        result = validate_json_trace(path)
        self.assertFalse(result["valid"])
        self.assertEqual(result["record_count"], 1)
        self.assertIn("channel_drained", result["errors"][0])

    def test_valid_json_with_an_unsupported_record_is_not_a_valid_trace(self) -> None:
        for record in ({"type": "unknown"}, 42):
            with self.subTest(record=record):
                rows = [*_capture_records(1), record]
                path = self.create_temp_file(
                    "unsupported.ndjson",
                    "\n".join(json.dumps(row) for row in rows) + "\n",
                )
                self.assertEqual(validate_json_syntax(path), (4, []))
                result = validate_json_trace(path)
                self.assertFalse(result["valid"])
                self.assertTrue(result["errors"])

    def test_validate_json_syntax_valid_file(self):
        """Test validation of real NDJSON trace file."""
        valid_count, errors = validate_json_syntax(REG_TRACE_NDJSON)

        self.assertEqual(valid_count, REG_TRACE_NDJSON_RECORD_COUNT)
        self.assertEqual(len(errors), 0)

    def test_validate_json_syntax_invalid_json(self):
        """Test detection of invalid JSON syntax."""
        valid_count, errors = validate_json_syntax(INVALID_SYNTAX_NDJSON)

        self.assertEqual(valid_count, 2)
        self.assertEqual(len(errors), 1)
        self.assertIn("Line 2", errors[0])

    def test_validate_json_syntax_file_not_found(self):
        """Test handling of non-existent file."""
        non_existent = self.temp_dir / "missing.ndjson"

        with self.assertRaises(FileNotFoundError):
            validate_json_syntax(non_existent)

    def test_validate_json_schema_valid(self):
        """Test schema validation with real trace data."""
        result = validate_json_schema(REG_TRACE_NDJSON, message_type="reg_trace")

        self.assertTrue(result)

    def test_validate_json_schema_invalid(self):
        """Test schema validation with invalid records."""
        with self.assertRaises(JsonValidationError) as ctx:
            validate_json_schema(INVALID_SCHEMA_NDJSON, message_type="reg_trace")

        self.assertIn("Schema validation failed", str(ctx.exception))

    def test_validate_json_trace_complete(self):
        """Test complete validation of real trace file."""
        result = validate_json_trace(REG_TRACE_NDJSON)

        self.assertTrue(result["valid"])
        self.assertEqual(result["record_count"], REG_TRACE_NDJSON_RECORD_COUNT)
        self.assertEqual(result["message_type"], "reg_trace")
        self.assertEqual(len(result["errors"]), 0)

    def test_validate_json_trace_empty_file(self):
        """Test validation of empty file."""
        empty_file = self.create_temp_file("empty.ndjson", "")

        result = validate_json_trace(empty_file)

        self.assertFalse(result["valid"])
        self.assertEqual(result["record_count"], 0)


class JsonValidatorCompressedTest(BaseValidationTest):
    """Tests for JSON validator with compressed files."""

    def test_validate_json_syntax_compressed(self):
        """Test validation of Zstd compressed NDJSON trace file."""
        if not REG_TRACE_NDJSON_ZST.exists():
            self.skipTest("Zstd test file not available")

        valid_count, errors = validate_json_syntax(REG_TRACE_NDJSON_ZST)

        self.assertEqual(valid_count, REG_TRACE_NDJSON_ZST_RECORD_COUNT)
        self.assertEqual(len(errors), 0)

    def test_validate_json_schema_compressed(self):
        """Test schema validation with compressed trace data."""
        if not REG_TRACE_NDJSON_ZST.exists():
            self.skipTest("Zstd test file not available")

        result = validate_json_schema(REG_TRACE_NDJSON_ZST, message_type="reg_trace")

        self.assertTrue(result)

    def test_validate_json_trace_compressed(self):
        """Test complete validation of compressed trace file."""
        if not REG_TRACE_NDJSON_ZST.exists():
            self.skipTest("Zstd test file not available")

        result = validate_json_trace(REG_TRACE_NDJSON_ZST)

        self.assertTrue(result["valid"])
        self.assertEqual(result["record_count"], REG_TRACE_NDJSON_ZST_RECORD_COUNT)
        self.assertEqual(result["message_type"], "reg_trace")
        self.assertEqual(result["compression"], "zstd")
        self.assertEqual(len(result["errors"]), 0)

    def test_validate_json_trace_compression_field_uncompressed(self):
        """Test that uncompressed files have compression='none'."""
        result = validate_json_trace(REG_TRACE_NDJSON)

        self.assertEqual(result["compression"], "none")


if __name__ == "__main__":
    unittest.main()
