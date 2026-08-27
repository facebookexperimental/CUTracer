# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Unit tests for warp_summary module.
"""

import unittest

from cutracer.query.warp_summary import (
    compute_warp_summary,
    format_ranges,
    format_warp_summary_text,
    is_exit_instruction,
    is_exit_sass,
    merge_to_ranges,
    warp_completed,
    warp_summary_to_dict,
    WarpSummary,
)


class TestIsExitInstruction(unittest.TestCase):
    """Tests for is_exit_instruction function."""

    def test_simple_exit(self):
        """Test simple EXIT instruction."""
        record = {"sass": "EXIT;"}
        self.assertTrue(is_exit_instruction(record))

    def test_predicated_exit(self):
        """Test predicated EXIT instruction."""
        record = {"sass": "@P0 EXIT;"}
        self.assertTrue(is_exit_instruction(record))

    def test_exit_with_modifier(self):
        """Test EXIT with modifier."""
        record = {"sass": "EXIT.KEEPREFCOUNT;"}
        self.assertTrue(is_exit_instruction(record))

    def test_lowercase_exit(self):
        """Test lowercase exit."""
        record = {"sass": "exit;"}
        self.assertTrue(is_exit_instruction(record))

    def test_non_exit_instruction(self):
        """Test non-EXIT instruction."""
        record = {"sass": "MOV R1, R0;"}
        self.assertFalse(is_exit_instruction(record))

    def test_empty_sass(self):
        """Test empty sass field."""
        record = {"sass": ""}
        self.assertFalse(is_exit_instruction(record))

    def test_no_sass_field(self):
        """Test record without sass field."""
        record = {"warp": 0, "pc": 16}
        self.assertFalse(is_exit_instruction(record))

    def test_exit_without_semicolon(self):
        """Test EXIT without semicolon should return False."""
        record = {"sass": "EXIT"}
        self.assertFalse(is_exit_instruction(record))


class IsExitSassTest(unittest.TestCase):
    """Tests for the sass-level is_exit_sass predicate.

    is_exit_instruction is a thin wrapper over this, so the two must agree.
    """

    def test_exit_forms(self):
        """All the EXIT spellings a cubin emits."""
        for sass in ("EXIT;", "EXIT ;", "@P0 EXIT ;", "EXIT.KEEPREFCOUNT;", "exit;"):
            with self.subTest(sass=sass):
                self.assertTrue(is_exit_sass(sass))

    def test_non_exit(self):
        """Ordinary instructions, including the trailing NOP, are not EXIT."""
        for sass in ("MOV R1, R0;", "NOP;", "BAR.SYNC.DEFER_BLOCKING 0x1 ;"):
            with self.subTest(sass=sass):
                self.assertFalse(is_exit_sass(sass))

    def test_empty_and_none(self):
        """Missing sass is not an EXIT (and must not raise)."""
        self.assertFalse(is_exit_sass(""))
        self.assertFalse(is_exit_sass(None))

    def test_agrees_with_record_wrapper(self):
        """is_exit_instruction delegates to is_exit_sass."""
        for sass in ("EXIT ;", "NOP;", "", "EXIT"):
            with self.subTest(sass=sass):
                self.assertEqual(
                    is_exit_sass(sass), is_exit_instruction({"sass": sass})
                )


class WarpCompletedTest(unittest.TestCase):
    """Tests for the shared warp-completion predicate."""

    def test_no_records(self):
        """A warp with no records did not complete."""
        self.assertFalse(warp_completed([]))

    def test_exit_is_last_record(self):
        """The ordinary shape: EXIT terminates the record sequence."""
        records = [{"sass": "MOV R1, R0;"}, {"sass": "EXIT ;"}]
        self.assertTrue(warp_completed(records))

    def test_epilogue_after_exit(self):
        """Records emitted past EXIT do not un-complete the warp.

        This is the case-24 shape: 24 of 32 warps execute "EXIT ;" and then
        emit one more record, "NOP;".
        """
        records = [{"sass": "MOV R1, R0;"}, {"sass": "EXIT ;"}, {"sass": "NOP;"}]
        self.assertTrue(warp_completed(records))

    def test_never_exits(self):
        """A warp stuck on a barrier never completes."""
        records = [
            {"sass": "MOV R1, R0;"},
            {"sass": "BAR.SYNC.DEFER_BLOCKING 0x1 ;"},
        ]
        self.assertFalse(warp_completed(records))

    def test_accepts_an_iterator(self):
        """The argument is consumed once, so a bare generator works."""
        records = iter([{"sass": "NOP;"}, {"sass": "EXIT ;"}])
        self.assertTrue(warp_completed(records))


class TestMergeToRanges(unittest.TestCase):
    """Tests for merge_to_ranges function."""

    def test_empty_list(self):
        """Test with empty list."""
        result = merge_to_ranges([])
        self.assertEqual(result, [])

    def test_single_id(self):
        """Test with single ID."""
        result = merge_to_ranges([5])
        self.assertEqual(result, [(5, 5)])

    def test_consecutive_ids(self):
        """Test consecutive IDs."""
        result = merge_to_ranges([0, 1, 2, 3])
        self.assertEqual(result, [(0, 3)])

    def test_non_consecutive_ids(self):
        """Test non-consecutive IDs."""
        result = merge_to_ranges([0, 1, 2, 5, 6, 7])
        self.assertEqual(result, [(0, 2), (5, 7)])

    def test_unsorted_ids(self):
        """Test unsorted IDs are sorted."""
        result = merge_to_ranges([5, 2, 3, 0, 1])
        self.assertEqual(result, [(0, 3), (5, 5)])

    def test_multiple_gaps(self):
        """Test multiple gaps."""
        result = merge_to_ranges([0, 2, 4, 6])
        self.assertEqual(result, [(0, 0), (2, 2), (4, 4), (6, 6)])


class TestFormatRanges(unittest.TestCase):
    """Tests for format_ranges function."""

    def test_empty_ranges(self):
        """Test empty ranges."""
        result = format_ranges([])
        self.assertEqual(result, "(none)")

    def test_single_value_range(self):
        """Test single value range."""
        result = format_ranges([(5, 5)])
        self.assertEqual(result, "5")

    def test_actual_range(self):
        """Test actual range."""
        result = format_ranges([(0, 3)])
        self.assertEqual(result, "0-3")

    def test_multiple_ranges(self):
        """Test multiple ranges."""
        result = format_ranges([(0, 3), (6, 9)])
        self.assertEqual(result, "0-3, 6-9")

    def test_mixed_ranges(self):
        """Test mixed single values and ranges."""
        result = format_ranges([(0, 3), (5, 5), (8, 10)])
        self.assertEqual(result, "0-3, 5, 8-10")


class TestComputeWarpSummary(unittest.TestCase):
    """Tests for compute_warp_summary function."""

    def test_empty_groups(self):
        """Test with empty groups."""
        result = compute_warp_summary({})
        self.assertIsNone(result)

    def test_non_integer_keys(self):
        """Test with non-integer keys."""
        groups = {"a": [{"sass": "MOV R1, R0;"}], "b": [{"sass": "EXIT;"}]}
        result = compute_warp_summary(groups)
        self.assertIsNone(result)

    def test_skips_none_key_among_integers(self):
        """A None key (e.g. a kernel_metadata record with no 'warp' field)
        is skipped rather than aborting the whole summary."""
        groups = {
            None: [{"type": "kernel_metadata"}],
            0: [{"sass": "EXIT;"}],
            1: [{"sass": "MOV R1, R0;"}],
        }
        result = compute_warp_summary(groups)
        self.assertIsNotNone(result)
        # The None group is excluded from the observed-warp count.
        self.assertEqual(result.total_observed, 2)
        self.assertEqual(result.min_warp_id, 0)
        self.assertEqual(result.max_warp_id, 1)
        self.assertEqual(result.completed_warp_ids, [0])
        self.assertEqual(result.inprogress_warp_ids, [1])

    def test_all_completed(self):
        """Test all warps completed."""
        groups = {
            0: [{"sass": "MOV R1, R0;"}, {"sass": "EXIT;"}],
            1: [{"sass": "ADD R1, R2;"}, {"sass": "EXIT;"}],
        }
        result = compute_warp_summary(groups)
        self.assertIsNotNone(result)
        self.assertEqual(result.total_observed, 2)
        self.assertEqual(result.completed_warp_ids, [0, 1])
        self.assertEqual(result.inprogress_warp_ids, [])

    def test_all_inprogress(self):
        """Test all warps in progress."""
        groups = {
            0: [{"sass": "MOV R1, R0;"}],
            1: [{"sass": "ADD R1, R2;"}],
        }
        result = compute_warp_summary(groups)
        self.assertIsNotNone(result)
        self.assertEqual(result.completed_warp_ids, [])
        self.assertEqual(result.inprogress_warp_ids, [0, 1])

    def test_mixed_status(self):
        """Test mixed completed and in-progress."""
        groups = {
            0: [{"sass": "EXIT;"}],
            1: [{"sass": "MOV R1, R0;"}],
            2: [{"sass": "EXIT;"}],
        }
        result = compute_warp_summary(groups)
        self.assertIsNotNone(result)
        self.assertEqual(result.completed_warp_ids, [0, 2])
        self.assertEqual(result.inprogress_warp_ids, [1])

    def test_epilogue_past_exit_counts_as_completed(self):
        """Regression: a warp whose records run "... EXIT ; NOP ;" is completed.

        Testing only the last file-order record scored these warps in-progress.
        Measured on the case-24 capture
        kernel_caf5167c275460d4_iter0_buggy_matmul_kernel_tma_ws_blackwell
        (32 warps, 69699 instruction records), 24 warps emit exactly one "NOP;"
        after "EXIT ;", so the last-record test reported 0/32 completed on a
        trace where 24 warps had provably finished.
        """
        groups = {
            0: [{"sass": "MOV R1, R0;"}, {"sass": "EXIT ;"}, {"sass": "NOP;"}],
            1: [{"sass": "MOV R1, R0;"}, {"sass": "EXIT ;"}, {"sass": "NOP;"}],
            2: [{"sass": "MOV R1, R0;"}, {"sass": "BAR.SYNC.DEFER_BLOCKING 0x1 ;"}],
        }
        result = compute_warp_summary(groups)
        self.assertIsNotNone(result)
        self.assertEqual(result.completed_warp_ids, [0, 1])
        self.assertEqual(result.inprogress_warp_ids, [2])

    def test_exit_anywhere_in_the_sequence_completes(self):
        """The EXIT need not be the last or the second-to-last record."""
        groups = {
            0: [
                {"sass": "EXIT ;"},
                {"sass": "NOP;"},
                {"sass": "NOP;"},
                {"sass": "NOP;"},
            ],
        }
        result = compute_warp_summary(groups)
        self.assertIsNotNone(result)
        self.assertEqual(result.completed_warp_ids, [0])
        self.assertEqual(result.inprogress_warp_ids, [])

    def test_missing_warps(self):
        """Test missing warps detection."""
        groups = {
            0: [{"sass": "EXIT;"}],
            3: [{"sass": "EXIT;"}],
        }
        result = compute_warp_summary(groups)
        self.assertIsNotNone(result)
        self.assertEqual(result.missing_warp_ids, [1, 2])

    def test_warp_id_range(self):
        """Test warp ID range calculation."""
        groups = {
            5: [{"sass": "EXIT;"}],
            10: [{"sass": "EXIT;"}],
        }
        result = compute_warp_summary(groups)
        self.assertIsNotNone(result)
        self.assertEqual(result.min_warp_id, 5)
        self.assertEqual(result.max_warp_id, 10)


class TestFormatWarpSummaryText(unittest.TestCase):
    """Tests for format_warp_summary_text function."""

    def test_format_basic(self):
        """Test basic formatting."""
        summary = WarpSummary(
            total_observed=3,
            min_warp_id=0,
            max_warp_id=2,
            completed_warp_ids=[0, 2],
            inprogress_warp_ids=[1],
            missing_warp_ids=[],
        )
        result = format_warp_summary_text(summary)
        self.assertIn("Warp Summary", result)
        self.assertIn("Total warps observed:   3", result)
        self.assertIn("Completed (EXIT):", result)
        self.assertIn("In-progress:", result)
        self.assertIn("Missing (never seen):", result)

    def test_format_percentages(self):
        """Test percentage calculation."""
        summary = WarpSummary(
            total_observed=4,
            min_warp_id=0,
            max_warp_id=3,
            completed_warp_ids=[0, 1],
            inprogress_warp_ids=[2, 3],
            missing_warp_ids=[],
        )
        result = format_warp_summary_text(summary)
        self.assertIn("50.0%", result)


class TestWarpSummaryToDict(unittest.TestCase):
    """Tests for warp_summary_to_dict function."""

    def test_to_dict_basic(self):
        """Test basic dict conversion."""
        summary = WarpSummary(
            total_observed=2,
            min_warp_id=0,
            max_warp_id=1,
            completed_warp_ids=[0],
            inprogress_warp_ids=[1],
            missing_warp_ids=[],
        )
        result = warp_summary_to_dict(summary)
        self.assertEqual(result["total_observed"], 2)
        self.assertEqual(result["warp_id_range"], [0, 1])
        self.assertEqual(result["completed"]["count"], 1)
        self.assertEqual(result["in_progress"]["count"], 1)
        self.assertEqual(result["missing"]["count"], 0)

    def test_to_dict_percentages(self):
        """Test percentage calculation in dict."""
        summary = WarpSummary(
            total_observed=4,
            min_warp_id=0,
            max_warp_id=3,
            completed_warp_ids=[0, 1, 2, 3],
            inprogress_warp_ids=[],
            missing_warp_ids=[],
        )
        result = warp_summary_to_dict(summary)
        self.assertEqual(result["completed"]["percentage"], 100.0)
        self.assertEqual(result["in_progress"]["percentage"], 0.0)

    def test_to_dict_ranges(self):
        """Test ranges in dict."""
        summary = WarpSummary(
            total_observed=2,
            min_warp_id=0,
            max_warp_id=3,
            completed_warp_ids=[0, 1],
            inprogress_warp_ids=[],
            missing_warp_ids=[2, 3],
        )
        result = warp_summary_to_dict(summary)
        self.assertEqual(result["completed"]["ranges"], [(0, 1)])
        self.assertEqual(result["missing"]["ranges"], [(2, 3)])


if __name__ == "__main__":
    unittest.main()
