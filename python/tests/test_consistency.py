# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Unit tests for consistency checker module.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

import zstandard as zstd
from cutracer.validation.consistency import (
    compare_record_counts,
    compare_trace_formats,
    get_trace_statistics,
)
from tests.test_base import (
    BaseValidationTest,
    REG_TRACE_LOG,
    REG_TRACE_LOG_RECORD_COUNT,
    REG_TRACE_NDJSON,
    REG_TRACE_NDJSON_RECORD_COUNT,
    REG_TRACE_NDJSON_ZST,
    REG_TRACE_NDJSON_ZST_RECORD_COUNT,
)


class ConsistencyTest(BaseValidationTest):
    """Core tests for consistency checker functions."""

    def test_compare_record_counts_exact_match(self):
        """Test that exact count match returns True."""
        text_metadata = {"record_count": 100}
        json_metadata = {"record_count": 100}

        result = compare_record_counts(text_metadata, json_metadata)

        self.assertTrue(result)

    def test_compare_record_counts_outside_tolerance(self):
        """Test that counts outside tolerance return False."""
        text_metadata = {"record_count": 100}
        json_metadata = {"record_count": 150}

        result = compare_record_counts(text_metadata, json_metadata, tolerance=0.1)

        self.assertFalse(result)

    def test_compare_record_counts_missing_field(self):
        """Test that missing record_count field raises ValueError."""
        with self.assertRaises(ValueError):
            compare_record_counts({}, {"record_count": 100})

        with self.assertRaises(ValueError):
            compare_record_counts({"record_count": 100}, {})

    def test_get_trace_statistics_real_ndjson(self):
        """Test statistics extraction from real NDJSON file."""
        stats = get_trace_statistics(REG_TRACE_NDJSON)

        self.assertEqual(stats["format"], "json")
        self.assertEqual(stats["record_count"], REG_TRACE_NDJSON_RECORD_COUNT)
        self.assertGreater(stats["file_size"], 0)
        self.assertIn("reg_trace", stats["message_types"])

    def test_get_trace_statistics_real_text(self):
        """Test statistics extraction from real text file."""
        stats = get_trace_statistics(REG_TRACE_LOG)

        self.assertEqual(stats["format"], "text")
        self.assertEqual(stats["record_count"], REG_TRACE_LOG_RECORD_COUNT)
        self.assertGreater(stats["file_size"], 0)

    def test_get_trace_statistics_file_not_found(self):
        """Test handling of non-existent file."""
        non_existent = self.temp_dir / "missing.ndjson"

        with self.assertRaises(FileNotFoundError):
            get_trace_statistics(non_existent)

    def test_compare_trace_formats_real_files(self):
        """Test comparison of real text and JSON trace files."""
        result = compare_trace_formats(REG_TRACE_LOG, REG_TRACE_NDJSON)

        self.assertIn("consistent", result)
        self.assertIn("record_count_match", result)
        self.assertIn("text_records", result)
        self.assertIn("json_records", result)
        self.assertEqual(result["text_records"], REG_TRACE_LOG_RECORD_COUNT)
        self.assertEqual(result["json_records"], REG_TRACE_NDJSON_RECORD_COUNT)


class ConsistencyCompressedTest(BaseValidationTest):
    """Tests for consistency checker with compressed files."""

    def test_get_trace_statistics_compressed(self):
        """Test statistics extraction from compressed NDJSON file."""
        if not REG_TRACE_NDJSON_ZST.exists():
            self.skipTest("Zstd test file not available")

        stats = get_trace_statistics(REG_TRACE_NDJSON_ZST)

        self.assertEqual(stats["format"], "json")
        self.assertEqual(stats["compression"], "zstd")
        self.assertEqual(stats["record_count"], REG_TRACE_NDJSON_ZST_RECORD_COUNT)
        self.assertGreater(stats["file_size"], 0)
        self.assertIn("reg_trace", stats["message_types"])

    def test_get_trace_statistics_compression_field_uncompressed(self):
        """Test that uncompressed files have compression='none'."""
        stats = get_trace_statistics(REG_TRACE_NDJSON)

        self.assertEqual(stats["compression"], "none")

    def test_compare_trace_formats_with_compressed_json(self):
        """Test comparison of text and compressed JSON trace files."""
        if not REG_TRACE_NDJSON_ZST.exists():
            self.skipTest("Zstd test file not available")

        result = compare_trace_formats(REG_TRACE_LOG, REG_TRACE_NDJSON_ZST)

        self.assertIn("consistent", result)
        self.assertIn("record_count_match", result)
        self.assertEqual(result["text_records"], REG_TRACE_LOG_RECORD_COUNT)
        self.assertEqual(result["json_records"], REG_TRACE_NDJSON_ZST_RECORD_COUNT)


class CompletionConsistencyTest(BaseValidationTest):
    """Compare instruction statistics while validating capture metadata."""

    def setUp(self) -> None:
        super().setUp()
        self.text_file = self.create_temp_file(
            "capture.log",
            "CTX 0x1 - CTA 0,0,0 - warp 0 - MOV R0, RZ;:\n    * Reg0_T0: 0x00000000\n",
        )
        self.header: dict[str, Any] = {
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
        self.event: dict[str, Any] = {
            "type": "reg_trace",
            "ctx": "0x1",
            "trace_index": 0,
            "timestamp": 0,
            "grid_launch_id": 0,
            "cta": [0, 0, 0],
            "warp": 0,
            "opcode_id": 0,
            "pc": "0x0",
            "sass": "MOV R0, RZ;",
            "regs": [[0] * 32],
        }
        self.footer: dict[str, Any] = {
            "type": "capture_completion",
            "version": 1,
            "grid_launch_id": 0,
            "kernel_completed": True,
            "channel_drained": True,
            "trace_records_written": 1,
            "dropped_records": 0,
            "errors": [],
            "status": "complete",
        }

    def _write_records(self, records: list[dict[str, Any]], compressed: bool) -> Path:
        data = ("\n".join(json.dumps(record) for record in records) + "\n").encode()
        suffix = ".ndjson.zst" if compressed else ".ndjson"
        path = self.temp_dir / f"capture{suffix}"
        path.write_bytes(zstd.ZstdCompressor().compress(data) if compressed else data)
        return path

    def test_completion_does_not_change_instruction_statistics(self) -> None:
        for compressed in (False, True):
            for with_footer in (False, True):
                with self.subTest(compressed=compressed, with_footer=with_footer):
                    records = [self.header, self.event]
                    if with_footer:
                        records.append(self.footer)
                    result = compare_trace_formats(
                        self.text_file, self._write_records(records, compressed)
                    )
                    self.assertTrue(result["consistent"], result["differences"])
                    self.assertEqual(result["text_records"], 1)
                    self.assertEqual(result["json_records"], 1)
                    self.assertEqual(result["unique_sass_count"], 1)

    def test_completion_does_not_hide_missing_instructions(self) -> None:
        self.text_file.write_text(self.text_file.read_text() * 2)
        for compressed in (False, True):
            with self.subTest(compressed=compressed):
                result = compare_trace_formats(
                    self.text_file,
                    self._write_records(
                        [self.header, self.event, self.footer], compressed
                    ),
                )
                self.assertFalse(result["consistent"])
                self.assertFalse(result["record_count_match"])
                self.assertEqual(result["text_records"], 2)
                self.assertEqual(result["json_records"], 1)

    def test_completion_does_not_hide_sass_mismatch(self) -> None:
        for compressed in (False, True):
            with self.subTest(compressed=compressed):
                result = compare_trace_formats(
                    self.text_file,
                    self._write_records(
                        [self.header, {**self.event, "sass": "NOP;"}, self.footer],
                        compressed,
                    ),
                )
                self.assertTrue(result["record_count_match"])
                self.assertFalse(result["content_match"])
                self.assertFalse(result["consistent"])
                self.assertTrue(
                    any("SASS set mismatch" in error for error in result["differences"])
                )

    def test_invalid_completion_is_still_rejected(self) -> None:
        footer = dict(self.footer)
        del footer["channel_drained"]
        for compressed in (False, True):
            with self.subTest(compressed=compressed):
                result = compare_trace_formats(
                    self.text_file,
                    self._write_records([self.header, self.event, footer], compressed),
                )
                self.assertFalse(result["consistent"])
                self.assertTrue(
                    any("channel_drained" in error for error in result["differences"])
                )


if __name__ == "__main__":
    unittest.main()
