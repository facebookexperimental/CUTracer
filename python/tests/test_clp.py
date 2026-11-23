# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Unit tests for CLP archive validator module.
"""

import unittest

from cutracer.validation.clp_validator import ClpValidationError, detect_clp_archive
from tests.test_base import (
    BaseValidationTest,
    CLP_ARCHIVE_SAMPLE,
    EXAMPLE_INPUTS_DIR,
    REG_TRACE_NDJSON,
    REG_TRACE_NDJSON_ZST,
)


class ClpDetectionTest(BaseValidationTest):
    """Tests for CLP archive detection."""

    def test_detect_clp_archive_by_extension(self):
        """Test detection of CLP archive by .clp extension."""
        if not CLP_ARCHIVE_SAMPLE.exists():
            self.skipTest("CLP sample file not available")

        result = detect_clp_archive(CLP_ARCHIVE_SAMPLE)
        self.assertTrue(result)

    def test_detect_clp_archive_not_clp_ndjson(self):
        """Test that NDJSON files are not detected as CLP."""
        result = detect_clp_archive(REG_TRACE_NDJSON)
        self.assertFalse(result)

    def test_detect_clp_archive_not_clp_zst(self):
        """Test that Zstd files are not detected as CLP."""
        if not REG_TRACE_NDJSON_ZST.exists():
            self.skipTest("Zstd test file not available")

        result = detect_clp_archive(REG_TRACE_NDJSON_ZST)
        self.assertFalse(result)

    def test_detect_clp_archive_nonexistent(self):
        """Test that FileNotFoundError is raised for non-existent file."""
        non_existent = EXAMPLE_INPUTS_DIR / "nonexistent.clp"

        with self.assertRaises(FileNotFoundError):
            detect_clp_archive(non_existent)

    def test_detect_clp_archive_by_extension_only(self):
        """Test detection works based on .clp extension for existing file."""
        fake_clp = self.create_temp_file("fake.clp", "not real clp content")

        result = detect_clp_archive(fake_clp)
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
