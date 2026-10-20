# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Unit tests for StreamingGrouper class.
"""

import unittest

from cutracer.query.grouper import StreamingGrouper


class TestStreamingGrouperInit(unittest.TestCase):
    """Tests for StreamingGrouper initialization."""

    def test_init_stores_group_field(self):
        """Test that group field is stored correctly."""
        records = iter([])
        grouper = StreamingGrouper(records, "warp")
        self.assertEqual(grouper._group_field, "warp")

    def test_init_not_consumed(self):
        """Test that grouper is not consumed initially."""
        records = iter([])
        grouper = StreamingGrouper(records, "warp")
        self.assertFalse(grouper._consumed)


class TestHeadPerGroup(unittest.TestCase):
    """Tests for head_per_group method."""

    def test_head_per_group_basic(self):
        """Test basic head per group functionality."""
        records = iter(
            [
                {"warp": 0, "pc": 1},
                {"warp": 0, "pc": 2},
                {"warp": 1, "pc": 3},
                {"warp": 0, "pc": 4},
                {"warp": 1, "pc": 5},
            ]
        )
        grouper = StreamingGrouper(records, "warp")
        groups = grouper.head_per_group(2)

        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups[0]), 2)
        self.assertEqual(len(groups[1]), 2)
        self.assertEqual(groups[0][0]["pc"], 1)
        self.assertEqual(groups[0][1]["pc"], 2)

    def test_head_per_group_limits_count(self):
        """Test that head limits records per group."""
        records = iter([{"warp": 0, "pc": i} for i in range(10)])
        grouper = StreamingGrouper(records, "warp")
        groups = grouper.head_per_group(3)

        self.assertEqual(len(groups[0]), 3)

    def test_head_per_group_zero(self):
        """Test head=0 returns empty dict."""
        records = iter([{"warp": 0}])
        grouper = StreamingGrouper(records, "warp")
        groups = grouper.head_per_group(0)

        self.assertEqual(groups, {})

    def test_head_per_group_empty_records(self):
        """Test with empty records iterator."""
        records = iter([])
        grouper = StreamingGrouper(records, "warp")
        groups = grouper.head_per_group(10)

        self.assertEqual(groups, {})


class TestTailPerGroup(unittest.TestCase):
    """Tests for tail_per_group method."""

    def test_tail_per_group_basic(self):
        """Test basic tail per group functionality."""
        records = iter(
            [
                {"warp": 0, "pc": 1},
                {"warp": 0, "pc": 2},
                {"warp": 0, "pc": 3},
                {"warp": 1, "pc": 10},
                {"warp": 1, "pc": 20},
            ]
        )
        grouper = StreamingGrouper(records, "warp")
        groups = grouper.tail_per_group(2)

        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups[0]), 2)
        # Should have last 2 records for warp 0
        self.assertEqual(groups[0][0]["pc"], 2)
        self.assertEqual(groups[0][1]["pc"], 3)

    def test_tail_per_group_discards_old(self):
        """Test that tail discards older records."""
        records = iter([{"warp": 0, "pc": i} for i in range(10)])
        grouper = StreamingGrouper(records, "warp")
        groups = grouper.tail_per_group(3)

        self.assertEqual(len(groups[0]), 3)
        # Should have pc values 7, 8, 9
        self.assertEqual(groups[0][0]["pc"], 7)
        self.assertEqual(groups[0][1]["pc"], 8)
        self.assertEqual(groups[0][2]["pc"], 9)

    def test_tail_per_group_zero(self):
        """Test tail=0 returns empty dict."""
        records = iter([{"warp": 0}])
        grouper = StreamingGrouper(records, "warp")
        groups = grouper.tail_per_group(0)

        self.assertEqual(groups, {})


class TestCountPerGroup(unittest.TestCase):
    """Tests for count_per_group method."""

    def test_count_per_group_basic(self):
        """Test basic count per group functionality."""
        records = iter(
            [
                {"warp": 0},
                {"warp": 0},
                {"warp": 0},
                {"warp": 1},
                {"warp": 1},
                {"warp": 2},
            ]
        )
        grouper = StreamingGrouper(records, "warp")
        counts = grouper.count_per_group()

        self.assertEqual(counts[0], 3)
        self.assertEqual(counts[1], 2)
        self.assertEqual(counts[2], 1)

    def test_count_per_group_empty(self):
        """Test count with empty records."""
        records = iter([])
        grouper = StreamingGrouper(records, "warp")
        counts = grouper.count_per_group()

        self.assertEqual(counts, {})


class TestConsumedBehavior(unittest.TestCase):
    """Tests for consumed state handling."""

    def test_cannot_use_after_consumed(self):
        """Test that grouper raises error after consumption."""
        records = iter([{"warp": 0}])
        grouper = StreamingGrouper(records, "warp")

        # First call consumes the iterator
        grouper.head_per_group(10)

        # Second call should raise
        with self.assertRaises(RuntimeError) as ctx:
            grouper.head_per_group(10)

        self.assertIn("already been consumed", str(ctx.exception))

    def test_different_methods_consume(self):
        """Test that any method marks as consumed."""
        records = iter([{"warp": 0}])
        grouper = StreamingGrouper(records, "warp")

        grouper.count_per_group()

        with self.assertRaises(RuntimeError):
            grouper.tail_per_group(10)


if __name__ == "__main__":
    unittest.main()


class TestListValuedGroupField(unittest.TestCase):
    """Grouping by a field whose values are lists, such as cta."""

    def _records(self):
        return iter(
            [
                {"cta": [0, 0, 0], "pc": 1},
                {"cta": [1, 0, 0], "pc": 2},
                {"cta": [0, 0, 0], "pc": 3},
            ]
        )

    def test_count_per_group(self):
        counts = StreamingGrouper(self._records(), "cta").count_per_group()
        self.assertEqual(counts, {(0, 0, 0): 2, (1, 0, 0): 1})

    def test_head_per_group(self):
        groups = StreamingGrouper(self._records(), "cta").head_per_group(1)
        self.assertEqual([r["pc"] for r in groups[(0, 0, 0)]], [1])

    def test_tail_per_group(self):
        groups = StreamingGrouper(self._records(), "cta").tail_per_group(1)
        self.assertEqual([r["pc"] for r in groups[(0, 0, 0)]], [3])

    def test_all_per_group(self):
        groups = StreamingGrouper(self._records(), "cta").all_per_group()
        self.assertEqual([r["pc"] for r in groups[(0, 0, 0)]], [1, 3])
        self.assertEqual([r["pc"] for r in groups[(1, 0, 0)]], [2])
