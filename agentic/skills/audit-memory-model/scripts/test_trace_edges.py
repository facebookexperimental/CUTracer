#!/usr/bin/env python3
# MIT License
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Unit tests for trace_edges.py on synthetic CUTracer traces.

Run: python3 test_trace_edges.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trace_edges as te

PARITY1 = 0x80000000
X, Y, Z = 0x20500, 0x20510, 0x20520  # mbarrier addresses
FULL = 0xFFFFFFFF


class Trace:
    """Build a synthetic trace. Each warp gets its own code region; pc = opcode_id * 16."""

    def __init__(self):
        self.code = {}  # opcode_id -> sass
        self.records = []
        self.next_region = 0

    def program(self, sass_lines):
        """Place straight-line code; return the list of pcs."""
        base = self.next_region
        self.next_region += 64
        pcs = []
        for i, sass in enumerate(sass_lines):
            self.code[base + i] = sass
            pcs.append((base + i) * 16)
        return pcs

    def run(
        self, warp, pc, *, mask=FULL, addrs=None, uregs=None, regs=None, cta=(0, 0, 0)
    ):
        oid = pc // 16
        if addrs is not None:
            # Lanes past the given list repeat its last address.
            lanes = [addrs[min(i, len(addrs) - 1)] for i in range(32)]
            self.records.append(
                {
                    "type": "mem_addr_trace",
                    "cta": list(cta),
                    "warp": warp,
                    "pc": hex(pc),
                    "opcode_id": oid,
                    "active_mask": hex(mask),
                    "addrs": lanes,
                }
            )
        rec = {
            "type": "reg_trace",
            "cta": list(cta),
            "warp": warp,
            "pc": hex(pc),
            "opcode_id": oid,
            "active_mask": hex(mask),
        }
        if uregs:
            rec["uregs_indices"], rec["uregs"] = list(uregs), list(uregs.values())
        if regs:
            rec["regs_indices"], rec["regs"] = (
                list(regs),
                [[v] * 32 for v in regs.values()],
            )
        self.records.append(rec)

    def analyze(self):
        for i, r in enumerate(self.records):
            r["trace_index"] = i
        meta = {
            "type": "kernel_metadata",
            "instructions": {str(k): {"sass": v} for k, v in self.code.items()},
        }
        with tempfile.NamedTemporaryFile("w", suffix=".ndjson", delete=False) as f:
            f.write(json.dumps(meta) + "\n")
            for r in self.records:
                f.write(json.dumps(r) + "\n")
            path = f.name
        try:
            return te.analyze(path)
        finally:
            os.unlink(path)


def wait_code(bar_reg="UR8", off="0x0", pred="P0", tok_reg="R6"):
    """A TRYWAIT followed by its guarded retry branch."""
    return [
        f"SYNCS.PHASECHK.TRANS64.TRYWAIT {pred}, [{bar_reg}+{off}], {tok_reg} ;",
        f"@!{pred} BRA 0x3f00 ;",
    ]


def succeed_wait(t, warp, pcs, bar, token):
    """Execute a successful wait: TRYWAIT, guarded branch skipped (no record)."""
    t.run(warp, pcs[0], addrs=[bar], regs={6: token})


def fail_wait(t, warp, pcs, bar, token):
    """Execute a failed poll: the guarded branch runs and goes to its target."""
    t.run(warp, pcs[0], addrs=[bar], regs={6: token})
    t.run(warp, pcs[1])
    t.run(warp, 0x3F00)


def row(rows, kind, a, b):
    return rows.get((kind, hex(a), hex(b)), {})


class WarTests(unittest.TestCase):
    """Generic read by warp 1, then the CLC-style async write by warp 0."""

    def build(
        self,
        *,
        fence=True,
        arrive_bar=X,
        wait_bar=X,
        wait_ok=True,
        writer_first_parity=0,
    ):
        t = Trace()
        t.code[0x3F00 // 16] = "NOP ;"
        r = t.program(
            [
                "LDS.128 R4, [UR9+0x10] ;",
                "FENCE.VIEW.ASYNC.S ;",
                "SYNCS.ARRIVE.TRANS64.A1T0 RZ, [UR9+0x0], RZ ;",
            ]
        )
        w = t.program(
            [*wait_code(), "UGETNEXTWORKID.BROADCAST [UR4], [UR5] ;", "NOP ;"]
        )
        # writer issues once (phase 0 of Y), reader waits for it (omitted), reads, releases.
        t.run(1, r[0], addrs=[0x1000], mask=1)
        if fence:
            t.run(1, r[1], mask=1)
        t.run(1, r[2], addrs=[arrive_bar], mask=1)
        (succeed_wait if wait_ok else fail_wait)(t, 0, w, wait_bar, writer_first_parity)
        if wait_ok:
            t.run(0, w[3])  # fall-through after the skipped branch
        t.run(0, w[2], mask=1, uregs={4: 0x1000, 5: Y})
        return t, r, w

    def test_positive_control(self):
        t, r, w = self.build()
        rows, *_ = t.analyze()
        got = row(rows, "generic_read->async_write", r[0], w[2])
        self.assertEqual(
            (got["occurrences"], got["fence_seen"], got["sync_linked"]), (1, 1, 1)
        )

    def test_fence_without_handshake(self):
        t, r, w = self.build(wait_bar=Z)  # writer waits on an unrelated barrier
        rows, *_ = t.analyze()
        got = row(rows, "generic_read->async_write", r[0], w[2])
        self.assertEqual(
            (got["fence_seen"], got["sync_linked"], got["no_sync_link"]), (1, 0, 1)
        )

    def test_missing_fence(self):
        t, r, w = self.build(fence=False)
        rows, *_ = t.analyze()
        got = row(rows, "generic_read->async_write", r[0], w[2])
        self.assertEqual((got["fence_seen"], got["sync_linked"]), (0, 1))

    def test_failed_poll_is_not_a_wait(self):
        t, r, w = self.build(wait_ok=False)
        rows, *_ = t.analyze()
        self.assertEqual(
            row(rows, "generic_read->async_write", r[0], w[2])["sync_linked"], 0
        )

    def test_phase_origin(self):
        """A writer whose first wait has parity 1 observed the pre-completed phase, not the
        reader's arrive (phase 0)."""
        t, r, w = self.build(writer_first_parity=PARITY1)
        rows, *_ = t.analyze()
        self.assertEqual(
            row(rows, "generic_read->async_write", r[0], w[2])["sync_linked"], 0
        )


class ClusterRankTests(unittest.TestCase):
    """Shared addresses carry the CTA's cluster rank above bit 24."""

    RANK1 = 1 << 24

    def build(self, rank_base, remote_read=False):
        t = Trace()
        t.code[0x3F00 // 16] = "NOP ;"
        cta = (1, 0, 0)
        r = t.program(
            [
                "LDS.128 R4, [UR9+0x10] ;",
                "FENCE.VIEW.ASYNC.S ;",
                "SYNCS.ARRIVE.TRANS64.A1T0 RZ, [UR9+0x0], RZ ;",
            ]
        )
        w = t.program(
            [*wait_code(), "UGETNEXTWORKID.BROADCAST [UR4], [UR5] ;", "NOP ;"]
        )
        read_at = 0x1000 + (0 if remote_read else rank_base)
        t.run(1, r[0], addrs=[read_at], mask=1, cta=cta)
        t.run(1, r[1], mask=1, cta=cta)
        t.run(1, r[2], addrs=[X + rank_base], mask=1, cta=cta)
        t.run(0, w[0], addrs=[X + rank_base], regs={6: 0}, cta=cta)
        t.run(0, w[3], cta=cta)
        t.run(
            0,
            w[2],
            mask=1,
            uregs={4: 0x1000 + rank_base, 5: Y + rank_base},
            cta=cta,
        )
        return t, r, w

    def test_rank1_addresses_match_like_rank0(self):
        for base in (0, self.RANK1):
            t, r, w = self.build(base)
            rows, *_ = t.analyze()
            got = row(rows, "generic_read->async_write", r[0], w[2])
            self.assertEqual(
                (got.get("occurrences"), got.get("fence_and_link")), (1, 1), hex(base)
            )

    def test_other_rank_access_is_remote_unresolved(self):
        t, r, w = self.build(self.RANK1, remote_read=True)
        rows, _witness, unresolved, _barriers = t.analyze()
        self.assertEqual(row(rows, "generic_read->async_write", r[0], w[2]), {})
        self.assertIn("remote_shared", str(unresolved))


class OwnRankTests(unittest.TestCase):
    """Async writes into another CTA must not outvote the CTA's own addresses."""

    def test_remote_async_writes_do_not_set_own_rank(self):
        rank1 = 1 << 24
        t = Trace()
        cta = (1, 0, 0)
        r = t.program(["LDS.128 R4, [UR9+0x10] ;"])
        w = t.program(["UTMALDG.2D [UR4], [UR6] ;"] * 4)
        t.run(1, r[0], addrs=[rank1 + 0x1000], mask=1, cta=cta)
        for p in w:
            t.run(0, p, mask=1, uregs={4: 0x1000, 6: 0x7F0000000000}, cta=cta)
        rows, _witness, unresolved, _barriers = t.analyze()
        self.assertFalse(any(hex(r[0]) in k for k in rows), rows)
        flags = {k[2] for k in unresolved if k[1].startswith("UTMALDG")}
        self.assertTrue(any("remote_shared" in f for f in flags), unresolved)


class RemoteArriveRebaseTests(unittest.TestCase):
    """A remote arrive's exclusion must survive cluster rebasing."""

    def test_remote_arrive_into_rank1_excludes_target_barrier(self):
        b = 1 << 24
        t = Trace()
        a = t.program(
            ["STS [R2], R4 ;", "SYNCS.ARRIVE.TRANS64.RED.A1T0 RZ, [R0+URZ], RZ ;"]
        )
        r = t.program(
            [
                "LDS.128 R4, [R2] ;",
                "FENCE.VIEW.ASYNC.S ;",
                "SYNCS.ARRIVE.TRANS64.A1T0 RZ, [UR9], RZ ;",
            ]
        )
        w = t.program(
            [*wait_code(), "NOP ;", "UGETNEXTWORKID.BROADCAST [UR4], [UR5] ;"]
        )
        for _ in range(2):
            t.run(0, a[0], mask=1, addrs=[0x3000], cta=(0, 0, 0))
        t.run(0, a[1], mask=1, addrs=[b + X], cta=(0, 0, 0))
        t.run(0, w[0], mask=1, addrs=[b + X], regs={6: 0}, cta=(1, 0, 0))
        t.run(0, w[2], mask=1, cta=(1, 0, 0))
        t.run(1, r[0], mask=1, addrs=[b + 0x1000], cta=(1, 0, 0))
        t.run(1, r[1], mask=1, cta=(1, 0, 0))
        t.run(1, r[2], mask=1, addrs=[b + X], cta=(1, 0, 0))
        t.run(0, w[3], mask=1, uregs={4: b + 0x1000, 5: b + Y}, cta=(1, 0, 0))
        rows, _witness, _unresolved, barriers = t.analyze()
        got = row(rows, "generic_read->async_write", r[0], w[3])
        self.assertEqual(got.get("fence_and_link", 0), 0, got)
        self.assertFalse(barriers.mbar_ok((1, 0, 0), X))


class ClassifyTests(unittest.TestCase):
    def test_descriptor_mma_forms(self):
        for op in (
            "UTCHMMA",
            "UTCIMMA",
            "UTCQMMA",
            "UTCOMMA",
            "UTCMMA",
            "UTCCP",
            "HGMMA",
            "IGMMA",
            "QGMMA",
            "BGMMA",
        ):
            self.assertEqual(te.classify(op + ".64x128x32")[0], "async_desc", op)


class ReuseWarningTests(unittest.TestCase):
    """A trace with no read-then-write candidate warns that reuse edges are missing."""

    def test_war_candidate_suppresses_warning(self):
        t, _r, _w = WarTests().build()
        rows, *_ = t.analyze()
        self.assertIsNone(te.reuse_warning(rows))

    def test_write_then_read_only_warns(self):
        t = Trace()
        t.code[0x3F00 // 16] = "NOP ;"
        w = t.program([*wait_code(), "UGETNEXTWORKID.BROADCAST [UR4], [UR5] ;"])
        r = t.program([*wait_code(), "LDS.128 R4, [UR9+0x10] ;"])
        t.run(0, w[2], mask=1, uregs={4: 0x1000, 5: Y})
        t.run(1, r[0], addrs=[Y], regs={6: 0})
        t.run(1, r[2], addrs=[0x1000], mask=1)
        rows, *_ = t.analyze()
        self.assertTrue(rows)
        self.assertEqual(te.reuse_warning(rows), te.NO_REUSE_WARNING)


class BarrierChecks(unittest.TestCase):
    def test_token_repeat_excludes_barrier(self):
        t = Trace()
        t.code[0x3F00 // 16] = "NOP ;"
        w = t.program([*wait_code(), "NOP ;"])
        for _ in range(2):
            succeed_wait(t, 0, w, X, 0)
            t.run(0, w[2])
        *_, barriers = t.analyze()
        self.assertIn(
            "phase_token_repeat", barriers.problems.get(("mbar", (0, 0, 0), X), set())
        )

    def test_reinitialized_excludes_barrier(self):
        t = Trace()
        i = t.program(["SYNCS.EXCH.64 URZ, [UR8+0x0], UR4 ;"])
        t.run(0, i[0], addrs=[X])
        t.run(0, i[0], addrs=[X])
        t.code[0x3F00 // 16] = "NOP ;"
        w = t.program([*wait_code(), "NOP ;"])
        succeed_wait(t, 1, w, X, 0)
        t.run(1, w[2])
        *_, barriers = t.analyze()
        self.assertIn(
            "reinitialized", barriers.problems.get(("mbar", (0, 0, 0), X), set())
        )

    def test_divergent_branch_leaves_wait_unresolved(self):
        t = Trace()
        t.code[0x3F00 // 16] = "NOP ;"
        w = t.program([*wait_code(), "NOP ;"])
        t.run(0, w[0], addrs=[X], regs={6: 0})
        t.run(0, w[1], mask=0x0000FFFF)  # half the lanes take the retry branch
        t.run(0, 0x3F00, mask=0x0000FFFF)
        *_, barriers = t.analyze()
        self.assertEqual(barriers.wait.get(((0, 0, 0), X), {}).get(0, []), [])


class BarSyncTests(unittest.TestCase):
    def test_reverse_receive_order_still_links(self):
        """W's BAR.SYNC record is received before R's; ordinals, not receive order, link them."""
        t = Trace()
        r = t.program(
            [
                "LDS R4, [R2] ;",
                "FENCE.VIEW.ASYNC.S ;",
                "BAR.SYNC.DEFER_BLOCKING 0x1, 0x40 ;",
            ]
        )
        w = t.program(
            ["BAR.SYNC.DEFER_BLOCKING 0x1, 0x40 ;", "UTMALDG.2D [UR24], [UR10] ;"]
        )
        t.run(1, r[0], addrs=[0x2000])
        t.run(1, r[1])
        t.run(0, w[0])  # received first
        t.run(1, r[2])
        t.run(0, w[1], uregs={24: 0x2000, 10: 0x7F0000000000})
        rows, *_ = t.analyze()
        got = row(rows, "generic_read->async_write", r[0], w[1])
        self.assertEqual((got["fence_seen"], got["sync_linked"]), (1, 1))

    def test_transitive_chain(self):
        """R releases X; V waits on X and releases Y; W waits on Y, then writes."""
        t = Trace()
        t.code[0x3F00 // 16] = "NOP ;"
        r = t.program(
            [
                "LDS R4, [R2] ;",
                "FENCE.VIEW.ASYNC.S ;",
                "SYNCS.ARRIVE.TRANS64.A1T0 RZ, [UR9+0x0], RZ ;",
            ]
        )
        v = t.program(
            [*wait_code(), "NOP ;", "SYNCS.ARRIVE.TRANS64.A1T0 RZ, [UR9+0x0], RZ ;"]
        )
        w = t.program([*wait_code(), "NOP ;", "UTMALDG.2D [UR24], [UR10] ;"])
        t.run(1, r[0], addrs=[0x3000], mask=1)
        t.run(1, r[1], mask=1)
        t.run(1, r[2], addrs=[X], mask=1)
        succeed_wait(t, 2, v, X, 0)
        t.run(2, v[2])
        t.run(2, v[3], addrs=[Y], mask=1)
        succeed_wait(t, 0, w, Y, 0)
        t.run(0, w[2])
        t.run(0, w[3], uregs={24: 0x3000, 10: 0x7F0000000000})
        rows, *_ = t.analyze()
        got = row(rows, "generic_read->async_write", r[0], w[3])
        self.assertEqual((got["sync_linked"], got["linked_via_intermediate"]), (1, 1))


class CandidateTests(unittest.TestCase):
    def test_offset_zero_and_earlier_read_retained(self):
        t = Trace()
        r = t.program(["LDS R4, [R2] ;", "LDS R5, [R2+0x4] ;"])
        w = t.program(["UTMALDG.2D [UR24], [UR10] ;"])
        t.run(1, r[0], addrs=[0x0])
        t.run(1, r[1], addrs=[0x0])
        t.run(0, w[0], uregs={24: 0x0, 10: 0x7F0000000000})
        rows, _w, unresolved, _b = t.analyze()
        self.assertEqual(
            row(rows, "generic_read->async_write", r[0], w[0])["occurrences"], 1
        )
        self.assertEqual(
            row(rows, "generic_read->async_write", r[1], w[0])["occurrences"], 1
        )
        self.assertFalse(any("operand_unresolved" in k[2] for k in unresolved))

    def test_disjoint_bytes_in_one_granule(self):
        t = Trace()
        g = t.program(["STS [R2], R4 ;"])
        a = t.program(["UTMASTG.2D [UR12], [UR10] ;"])
        t.run(1, g[0], addrs=[0x100])  # bytes 0x100..0x103
        t.run(0, a[0], uregs={12: 0x108, 10: 0x7F0000000000})  # starts at 0x108
        rows, *_ = t.analyze()
        self.assertEqual(rows, {})

    def test_interior_bulk_overlap_is_unresolved(self):
        t = Trace()
        a = t.program(["UTMALDG.2D [UR24], [UR10] ;"])
        r = t.program(["LDS R4, [R2] ;"])
        t.run(0, a[0], uregs={24: 0x400, 10: 0x7F0000000000})
        t.run(1, r[0], addrs=[0x440])  # inside the copied box, past its first granule
        rows, _w, unresolved, _b = t.analyze()
        self.assertEqual(rows, {})
        self.assertTrue(
            any(
                k[0] == hex(a[0]) and "footprint_unresolved" in k[2] for k in unresolved
            )
        )

    def test_waw_and_async_read_reuse(self):
        t = Trace()
        g = t.program(["STS [R2], R4 ;"])
        a = t.program(["UTMALDG.2D [UR24], [UR10] ;", "UTMASTG.2D [UR12], [UR10] ;"])
        g2 = t.program(["STS [R3], R5 ;"])
        t.run(1, g[0], addrs=[0x500])
        t.run(0, a[0], uregs={24: 0x500, 10: 0x7F0000000000})
        t.run(0, a[1], uregs={12: 0x500, 10: 0x7F0000000000})
        t.run(2, g2[0], addrs=[0x500])
        rows, *_ = t.analyze()
        self.assertEqual(
            row(rows, "generic_write->async_write", g[0], a[0])["occurrences"], 1
        )
        got = row(rows, "async_read->generic_write", a[1], g2[0])
        self.assertEqual((got["occurrences"], got["completion_unchecked"]), (1, 1))

    def test_same_warp_pair(self):
        t = Trace()
        p = t.program(
            ["LDS R4, [R2] ;", "FENCE.VIEW.ASYNC.S ;", "UTMALDG.2D [UR24], [UR10] ;"]
        )
        t.run(0, p[0], addrs=[0x600])
        t.run(0, p[1])
        t.run(0, p[2], uregs={24: 0x600, 10: 0x7F0000000000})
        rows, *_ = t.analyze()
        got = row(rows, "generic_read->async_write", p[0], p[2])
        self.assertEqual((got["same_warp"], got["same_warp_fence_seen"]), (1, 1))

    def test_shared_fence_only(self):
        """FENCE.VIEW.ASYNC.T (a TMEM fence) does not count as a shared-memory proxy fence."""
        t = Trace()
        p = t.program(
            ["LDS R4, [R2] ;", "FENCE.VIEW.ASYNC.T ;", "UTMALDG.2D [UR24], [UR10] ;"]
        )
        t.run(0, p[0], addrs=[0x700])
        t.run(0, p[1])
        t.run(0, p[2], uregs={24: 0x700, 10: 0x7F0000000000})
        rows, *_ = t.analyze()
        self.assertEqual(
            row(rows, "generic_read->async_write", p[0], p[2])["same_warp_fence_seen"],
            0,
        )


class RegressionTests(unittest.TestCase):
    """Failing inputs found in review."""

    def test_read_retained_across_intervening_async_write(self):
        t = Trace()
        r = t.program(["LDS R4, [R2] ;"])
        w = t.program(
            [
                "UGETNEXTWORKID.BROADCAST [UR4], [UR5] ;",
                "UGETNEXTWORKID.BROADCAST [UR4], [UR5] ;",
            ]
        )
        t.run(1, r[0], addrs=[0x1000])
        t.run(0, w[0], mask=1, uregs={4: 0x1000, 5: Y})
        t.run(0, w[1], mask=1, uregs={4: 0x1000, 5: Y})
        rows, *_ = t.analyze()
        self.assertEqual(
            row(rows, "generic_read->async_write", r[0], w[1])["occurrences"], 1
        )

    def test_cpasync_write_then_read(self):
        t = Trace()
        c = t.program(["LDGSTS.E.BYPASS.128 [R2], desc[UR4][R6.64] ;"])
        r = t.program(["LDS R4, [R3] ;"])
        t.run(0, c[0], addrs=[0x1000])
        t.run(1, r[0], addrs=[0x1000])
        rows, *_ = t.analyze()
        got = row(rows, "cpasync_write->generic_read", c[0], r[0])
        self.assertEqual((got["occurrences"], got["completion_unchecked"]), (1, 1))

    def test_later_fenced_link_preferred(self):
        t = Trace()
        t.code[0x3F00 // 16] = "NOP ;"
        r = t.program(
            [
                "LDS R4, [R2] ;",
                "SYNCS.ARRIVE.TRANS64.A1T0 RZ, [UR9+0x0], RZ ;",
                "FENCE.VIEW.ASYNC.S ;",
                "SYNCS.ARRIVE.TRANS64.A1T0 RZ, [UR9+0x10], RZ ;",
            ]
        )
        wx = t.program([*wait_code(), "NOP ;"])
        wy = t.program([*wait_code(), "NOP ;", "UTMALDG.2D [UR24], [UR10] ;"])
        t.run(1, r[0], addrs=[0x1000], mask=1)
        t.run(1, r[1], addrs=[X], mask=1)
        t.run(1, r[2], mask=1)
        t.run(1, r[3], addrs=[Y], mask=1)
        succeed_wait(t, 0, wx, X, 0)
        t.run(0, wx[2])
        succeed_wait(t, 0, wy, Y, 0)
        t.run(0, wy[2])
        t.run(0, wy[3], uregs={24: 0x1000, 10: 0x7F0000000000})
        rows, *_ = t.analyze()
        got = row(rows, "generic_read->async_write", r[0], wy[3])
        self.assertEqual((got["sync_linked"], got["fence_and_link"]), (1, 1))

    def test_same_warp_different_lanes(self):
        t = Trace()
        p = t.program(
            [
                "LDS R4, [R2] ;",
                "FENCE.VIEW.ASYNC.S ;",
                "UGETNEXTWORKID.BROADCAST [UR4], [UR5] ;",
            ]
        )
        t.run(0, p[0], addrs=[0x1000], mask=0x1)
        t.run(0, p[1], mask=0x2)
        t.run(0, p[2], mask=0x2, uregs={4: 0x1000, 5: Y})
        rows, *_ = t.analyze()
        got = row(rows, "generic_read->async_write", p[0], p[2])
        self.assertEqual(
            (got["same_warp_lanes_unresolved"], got["same_warp_fence_seen"]), (1, 0)
        )

    def test_undecodable_wait_is_reported(self):
        t = Trace()
        t.code[0x3F00 // 16] = "NOP ;"
        w = t.program(
            [
                "SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [UR8+0x0], R6 ;",
                "@!P1 BRA 0x3f00 ;",
                "NOP ;",
            ]
        )
        t.run(0, w[0], addrs=[X], regs={6: 0})
        t.run(0, w[2])
        _rows, _w, unresolved, _b = t.analyze()
        self.assertTrue(
            any(
                k[0] == hex(w[0]) and k[2] == "wait_success_unresolved"
                for k in unresolved
            )
        )


class LaneTests(unittest.TestCase):
    """Cross-warp evidence must hold for the lanes that made the accesses."""

    def build(
        self, *, read_mask, fence_mask, warpsync=False, wait_mask=FULL, write_mask=1
    ):
        t = Trace()
        t.code[0x3F00 // 16] = "NOP ;"
        r = t.program(
            [
                "LDS.128 R4, [UR9+0x10] ;",
                "FENCE.VIEW.ASYNC.S ;",
                "WARPSYNC.ALL ;",
                "SYNCS.ARRIVE.TRANS64.A1T0 RZ, [UR9+0x0], RZ ;",
            ]
        )
        w = t.program(
            [*wait_code(), "NOP ;", "UGETNEXTWORKID.BROADCAST [UR4], [UR5] ;"]
        )
        t.run(1, r[0], addrs=[0x1000] * 32, mask=read_mask)
        t.run(1, r[1], mask=fence_mask)
        if warpsync:
            t.run(1, r[2])
        t.run(1, r[3], addrs=[X], mask=1)
        t.run(0, w[0], addrs=[X] * 32, regs={6: 0}, mask=wait_mask)
        t.run(0, w[2], mask=wait_mask)
        t.run(0, w[3], mask=write_mask, uregs={4: 0x1000, 5: Y})
        rows, *_ = t.analyze()
        return row(rows, "generic_read->async_write", r[0], w[3])

    def test_fence_by_another_lane_does_not_count(self):
        got = self.build(read_mask=0x1, fence_mask=0x2)
        self.assertEqual((got["fence_seen"], got["fence_and_link"]), (0, 0))

    def test_acquire_lane_must_cover_writer(self):
        got = self.build(read_mask=0x1, fence_mask=0x1, wait_mask=0x1, write_mask=0x2)
        self.assertEqual((got["sync_linked"], got["sync_lanes_unresolved"]), (0, 1))

    def test_all_lane_read_then_elected_arrive_is_unresolved(self):
        got = self.build(read_mask=FULL, fence_mask=FULL)
        self.assertEqual((got["sync_linked"], got["sync_lanes_unresolved"]), (0, 1))

    def test_warpsync_joins_lanes_before_elected_arrive(self):
        got = self.build(read_mask=FULL, fence_mask=FULL, warpsync=True)
        self.assertEqual((got["sync_linked"], got["fence_and_link"]), (1, 1))

    def test_warpsync_must_include_release_lane(self):
        """Lane 1 reads, fences and syncs with itself; lane 0 arrives outside the sync."""
        t = Trace()
        t.code[0x3F00 // 16] = "NOP ;"
        r = t.program(
            [
                "LDS.128 R4, [UR9+0x10] ;",
                "FENCE.VIEW.ASYNC.S ;",
                "WARPSYNC 0x2 ;",
                "SYNCS.ARRIVE.TRANS64.A1T0 RZ, [UR9+0x0], RZ ;",
            ]
        )
        w = t.program(
            [*wait_code(), "NOP ;", "UGETNEXTWORKID.BROADCAST [UR4], [UR5] ;"]
        )
        t.run(1, r[0], addrs=[0, 0x1000], mask=0x2)
        t.run(1, r[1], mask=0x2)
        t.run(1, r[2], mask=0x2)
        t.run(1, r[3], addrs=[X], mask=1)
        t.run(0, w[0], addrs=[X], regs={6: 0}, mask=1)
        t.run(0, w[2], mask=1)
        t.run(0, w[3], mask=1, uregs={4: 0x1000, 5: Y})
        rows, *_ = t.analyze()
        got = row(rows, "generic_read->async_write", r[0], w[3])
        self.assertEqual((got["fence_and_link"], got["sync_lanes_unresolved"]), (0, 1))

    def fence_after_join(self, *, fence_first, depth2=False):
        """Lane 1 reads; WARPSYNC 0x3 joins lanes 0-1; lane 1 fences; lane 0 arrives on X.
        With fence_first, the fence precedes the WARPSYNC instead."""
        t = Trace()
        t.code[0x3F00 // 16] = "NOP ;"
        body = ["FENCE.VIEW.ASYNC.S ;", "WARPSYNC 0x3 ;"]
        r = t.program(
            [
                "LDS R4, [R2] ;",
                *(body if fence_first else body[::-1]),
                "SYNCS.ARRIVE.TRANS64.A1T0 RZ, [UR9+0x0], RZ ;",
            ]
        )
        t.run(1, r[0], addrs=[0, 0x1000], mask=0x2)
        t.run(1, r[1], mask=0x2 if fence_first else 0x3)
        t.run(1, r[2], mask=0x3 if fence_first else 0x2)
        t.run(1, r[3], addrs=[X], mask=1)
        bar = X
        if depth2:
            v = t.program(
                [*wait_code(), "NOP ;", "SYNCS.ARRIVE.TRANS64.A1T0 RZ, [UR9+0x0], RZ ;"]
            )
            t.run(2, v[0], addrs=[X], regs={6: 0}, mask=1)
            t.run(2, v[2], mask=1)
            t.run(2, v[3], addrs=[Z], mask=1)
            bar = Z
        w = t.program(
            [*wait_code(), "NOP ;", "UGETNEXTWORKID.BROADCAST [UR4], [UR5] ;"]
        )
        t.run(0, w[0], addrs=[bar], regs={6: 0}, mask=1)
        t.run(0, w[2], mask=1)
        t.run(0, w[3], mask=1, uregs={4: 0x1000, 5: Y})
        rows, *_ = t.analyze()
        return row(rows, "generic_read->async_write", r[0], w[3])

    def test_fence_after_lane_join_is_not_joined(self):
        for depth2 in (False, True):
            got = self.fence_after_join(fence_first=False, depth2=depth2)
            self.assertEqual(
                (got["fence_seen"], got["sync_linked"], got.get("fence_and_link", 0)),
                (1, 1, 0),
            )

    def test_fence_before_lane_join_is_joined(self):
        for depth2 in (False, True):
            got = self.fence_after_join(fence_first=True, depth2=depth2)
            self.assertEqual((got["sync_linked"], got["fence_and_link"]), (1, 1))

    def test_wait_lane_on_other_barrier_is_not_linked(self):
        """Warp 0's lane 0 waits on X, lane 1 on Z; lane 1 then writes."""
        t = Trace()
        t.code[0x3F00 // 16] = "NOP ;"
        z = t.program(["SYNCS.ARRIVE.TRANS64.A1T0 RZ, [R9], RZ ;"])
        r = t.program(
            [
                "LDS R4, [R2] ;",
                "FENCE.VIEW.ASYNC.S ;",
                "SYNCS.ARRIVE.TRANS64.A1T0 RZ, [R9], RZ ;",
            ]
        )
        w = t.program(
            [*wait_code("R8"), "NOP ;", "UGETNEXTWORKID.BROADCAST [UR4], [UR5] ;"]
        )
        t.run(3, z[0], addrs=[Z], mask=1)
        t.run(1, r[0], addrs=[0x1000], mask=1)
        t.run(1, r[1], mask=1)
        t.run(1, r[2], addrs=[X], mask=1)
        t.run(0, w[0], addrs=[X, Z], regs={6: 0}, mask=0x3)
        t.run(0, w[2], mask=0x3)
        t.run(0, w[3], mask=0x2, uregs={4: 0x1000, 5: Y})
        rows, _, _, barriers = t.analyze()
        got = row(rows, "generic_read->async_write", r[0], w[3])
        self.assertEqual(
            (got.get("sync_linked", 0), got.get("fence_and_link", 0)), (0, 0)
        )
        self.assertIn(
            "nonuniform_lane_addresses", barriers.problems[("mbar", (0, 0, 0), X)]
        )

    def test_raw_wait_lane_on_other_barrier_is_not_linked(self):
        t = Trace()
        t.code[0x3F00 // 16] = "NOP ;"
        z = t.program(["SYNCS.ARRIVE.TRANS64.A1T0 RZ, [R9], RZ ;"])
        a = t.program(["UGETNEXTWORKID.BROADCAST [UR4], [UR5] ;"])
        r = t.program([*wait_code("R8"), "NOP ;", "LDS R4, [R2] ;"])
        t.run(3, z[0], addrs=[Z], mask=1)
        t.run(1, a[0], mask=1, uregs={4: 0x1000, 5: X})
        t.run(0, r[0], addrs=[X, Z], regs={6: 0}, mask=0x3)
        t.run(0, r[2], mask=0x3)
        t.run(0, r[3], addrs=[0, 0x1000], mask=0x2)
        rows, *_ = t.analyze()
        got = row(rows, "async_write->generic_read", a[0], r[3])
        self.assertEqual(got.get("wait_linked", 0), 0)

    def test_release_lane_on_other_barrier_is_not_linked(self):
        """Lane 1 reads and fences but its arrive goes to Z; lane 0 arrives on X."""
        t = Trace()
        t.code[0x3F00 // 16] = "NOP ;"
        r = t.program(
            [
                "LDS R4, [R2] ;",
                "FENCE.VIEW.ASYNC.S ;",
                "SYNCS.ARRIVE.TRANS64.A1T0 RZ, [R9], RZ ;",
            ]
        )
        w = t.program(
            [*wait_code("R8"), "NOP ;", "UGETNEXTWORKID.BROADCAST [UR4], [UR5] ;"]
        )
        t.run(1, r[0], addrs=[0, 0x1000], mask=0x2)
        t.run(1, r[1], mask=0x2)
        t.run(1, r[2], addrs=[X, Z], mask=0x3)
        t.run(0, w[0], addrs=[X], regs={6: 0}, mask=1)
        t.run(0, w[2], mask=1)
        t.run(0, w[3], mask=1, uregs={4: 0x1000, 5: Y})
        rows, *_ = t.analyze()
        got = row(rows, "generic_read->async_write", r[0], w[3])
        self.assertEqual(
            (got.get("sync_linked", 0), got.get("fence_and_link", 0)), (0, 0)
        )

    def test_fenced_depth2_preferred_over_unfenced_direct(self):
        """Read; arrive X; fence; arrive Z. Warp 2 waits Z, arrives Y. Writer waits X and Y."""
        t = Trace()
        t.code[0x3F00 // 16] = "NOP ;"
        arrive = "SYNCS.ARRIVE.TRANS64.A1T0 RZ, [R9], RZ ;"
        r = t.program(["LDS R4, [R2] ;", arrive, "FENCE.VIEW.ASYNC.S ;", arrive])
        v = t.program([*wait_code(), "NOP ;", arrive])
        w = t.program(
            [
                *wait_code(),
                "NOP ;",
                *wait_code(),
                "NOP ;",
                "UGETNEXTWORKID.BROADCAST [UR4], [UR5] ;",
            ]
        )
        t.run(1, r[0], addrs=[0x1000], mask=1)
        t.run(1, r[1], addrs=[X], mask=1)
        t.run(1, r[2], mask=1)
        t.run(1, r[3], addrs=[Z], mask=1)
        t.run(2, v[0], addrs=[Z], regs={6: 0}, mask=1)
        t.run(2, v[2], mask=1)
        t.run(2, v[3], addrs=[Y], mask=1)
        t.run(0, w[0], addrs=[X], regs={6: 0}, mask=1)
        t.run(0, w[2], mask=1)
        t.run(0, w[3], addrs=[Y], regs={6: 0}, mask=1)
        t.run(0, w[5], mask=1)
        t.run(0, w[6], mask=1, uregs={4: 0x1000, 5: 0x20530})
        rows, *_ = t.analyze()
        got = row(rows, "generic_read->async_write", r[0], w[6])
        self.assertEqual(
            (got["fence_and_link"], got["linked_via_intermediate"]), (1, 1)
        )

    def r2p_between_wait_and_branch(self, raw):
        t = Trace()
        t.code[0x3F00 // 16] = "NOP ;"
        if raw:
            r = t.program(["UGETNEXTWORKID.BROADCAST [UR4], [UR5] ;"])
            t.run(1, r[0], mask=1, uregs={4: 0x1000, 5: X})
        else:
            r = t.program(
                [
                    "LDS R4, [R2] ;",
                    "FENCE.VIEW.ASYNC.S ;",
                    "SYNCS.ARRIVE.TRANS64.A1T0 RZ, [R9], RZ ;",
                ]
            )
            t.run(1, r[0], mask=1, addrs=[0x1000])
            t.run(1, r[1], mask=1)
            t.run(1, r[2], mask=1, addrs=[X])
        w = t.program(
            [
                "SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [R8], R6 ;",
                "R2P PR, R7, 0x7f ;",
                "@!P0 BRA 0x3f00 ;",
                "NOP ;",
                "LDS R4, [R2] ;" if raw else "UGETNEXTWORKID.BROADCAST [UR4], [UR5] ;",
            ]
        )
        t.run(0, w[0], mask=1, addrs=[X], regs={6: 0})
        t.run(0, w[1], mask=1, regs={7: 1})
        t.run(0, w[3], mask=1)
        if raw:
            t.run(0, w[4], mask=1, addrs=[0x1000])
        else:
            t.run(0, w[4], mask=1, uregs={4: 0x1000, 5: Y})
        rows, _, unresolved, _ = t.analyze()
        kind = "async_write->generic_read" if raw else "generic_read->async_write"
        return row(rows, kind, r[0], w[4]), unresolved

    def test_r2p_leaves_wait_unresolved(self):
        for raw in (False, True):
            got, unresolved = self.r2p_between_wait_and_branch(raw)
            self.assertEqual(
                (got.get("sync_linked", 0), got.get("wait_linked", 0)), (0, 0)
            )
            self.assertTrue(any(k[2] == "wait_success_unresolved" for k in unresolved))

    def register_bar(self, id0, id1):
        t = Trace()
        r = t.program(["LDS R4, [R2] ;", "FENCE.VIEW.ASYNC.S ;", "BAR.SYNC R8, 0x40 ;"])
        w = t.program(
            ["BAR.SYNC R9, 0x40 ;", "UGETNEXTWORKID.BROADCAST [UR4], [UR5] ;"]
        )
        t.run(1, r[0], addrs=[0x1000])
        t.run(1, r[1])
        t.run(1, r[2], regs={} if id0 is None else {8: id0})
        t.run(0, w[0], regs={} if id1 is None else {9: id1})
        t.run(0, w[1], mask=1, uregs={4: 0x1000, 5: Y})
        rows, *_ = t.analyze()
        return row(rows, "generic_read->async_write", r[0], w[1])

    def test_register_barrier_ids_are_read(self):
        self.assertEqual(self.register_bar(0, 1).get("sync_linked", 0), 0)
        self.assertEqual(self.register_bar(1, 1)["fence_and_link"], 1)

    def test_unrecorded_barrier_register_is_unresolved(self):
        self.assertEqual(self.register_bar(None, 1).get("sync_linked", 0), 0)
        self.assertEqual(self.register_bar(None, None).get("sync_linked", 0), 0)

    def test_warpsync_register_groups(self):
        """WARPSYNC R3 with lanes 0-15 and 16-31 in separate groups joins neither half to the other."""
        groups = [0xFFFF] * 16 + [0xFFFF0000] * 16

        def war(r3):
            t = Trace()
            t.code[0x3F00 // 16] = "NOP ;"
            r = t.program(
                [
                    "LDS R4, [R2] ;",
                    "FENCE.VIEW.ASYNC.S ;",
                    "WARPSYNC R3 ;",
                    "SYNCS.ARRIVE.TRANS64.A1T0 RZ, [R9], RZ ;",
                ]
            )
            w = t.program(
                [*wait_code("R8"), "NOP ;", "UGETNEXTWORKID.BROADCAST [UR4], [UR5] ;"]
            )
            t.run(1, r[0], addrs=[0] * 16 + [0x1000], mask=0x10000)
            t.run(1, r[1], mask=0x10000)
            t.run(1, r[2])
            t.records[-1]["regs_indices"], t.records[-1]["regs"] = [3], [r3]
            t.run(1, r[3], addrs=[X], mask=1)
            t.run(0, w[0], addrs=[X], regs={6: 0}, mask=1)
            t.run(0, w[2], mask=1)
            t.run(0, w[3], mask=1, uregs={4: 0x1000, 5: Y})
            rows, *_ = t.analyze()
            return row(rows, "generic_read->async_write", r[0], w[3])

        got = war(groups)
        self.assertEqual(
            (got.get("sync_linked", 0), got.get("fence_and_link", 0)), (0, 0)
        )
        got = war([0xFFFFFFFF] * 32)
        self.assertEqual(got["fence_and_link"], 1)

    def test_other_branch_between_wait_and_guard(self):
        """BRA.U between the wait and its guarded branch may skip it: wait unresolved."""
        for raw in (False, True):
            t = Trace()
            t.code[0x3F00 // 16] = "NOP ;"
            if raw:
                r = t.program(["UGETNEXTWORKID.BROADCAST [UR4], [UR5] ;"])
                t.run(1, r[0], mask=1, uregs={4: 0x1000, 5: X})
            else:
                r = t.program(
                    [
                        "LDS R4, [R2] ;",
                        "FENCE.VIEW.ASYNC.S ;",
                        "SYNCS.ARRIVE.TRANS64.A1T0 RZ, [R9], RZ ;",
                    ]
                )
                t.run(1, r[0], mask=1, addrs=[0x1000])
                t.run(1, r[1], mask=1)
                t.run(1, r[2], mask=1, addrs=[X])
            w = t.program(
                [
                    "SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [R8], R6 ;",
                    "BRA.U UP0, 0x40 ;",
                    "@!P0 BRA 0x3f00 ;",
                    "NOP ;",
                    (
                        "LDS R4, [R2] ;"
                        if raw
                        else "UGETNEXTWORKID.BROADCAST [UR4], [UR5] ;"
                    ),
                ]
            )
            t.run(0, w[0], mask=1, addrs=[X], regs={6: 0})
            t.run(0, w[1], mask=1)
            t.run(0, w[3], mask=1)
            if raw:
                t.run(0, w[4], mask=1, addrs=[0x1000])
            else:
                t.run(0, w[4], mask=1, uregs={4: 0x1000, 5: Y})
            rows, *_ = t.analyze()
            kind = "async_write->generic_read" if raw else "generic_read->async_write"
            got = row(rows, kind, r[0], w[4])
            self.assertEqual(
                (got.get("sync_linked", 0), got.get("wait_linked", 0)), (0, 0)
            )

    def test_brxu_between_wait_and_guard(self):
        for raw in (False, True):
            t = Trace()
            t.code[0x3F00 // 16] = "NOP ;"
            if raw:
                r = t.program(["UGETNEXTWORKID.BROADCAST [UR4], [UR5] ;"])
                t.run(1, r[0], mask=1, uregs={4: 0x1000, 5: X})
            else:
                r = t.program(
                    [
                        "LDS R4, [R2] ;",
                        "FENCE.VIEW.ASYNC.S ;",
                        "SYNCS.ARRIVE.TRANS64.A1T0 RZ, [R9], RZ ;",
                    ]
                )
                t.run(1, r[0], mask=1, addrs=[0x1000])
                t.run(1, r[1], mask=1)
                t.run(1, r[2], mask=1, addrs=[X])
            w = t.program(
                [
                    "SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [R8], R6 ;",
                    "BRXU UR4 -0x420 ;",
                    "@!P0 BRA 0x3f00 ;",
                    "NOP ;",
                    (
                        "LDS R4, [R2] ;"
                        if raw
                        else "UGETNEXTWORKID.BROADCAST [UR4], [UR5] ;"
                    ),
                ]
            )
            t.run(0, w[0], mask=1, addrs=[X], regs={6: 0})
            t.run(0, w[1], mask=1)
            t.run(0, w[3], mask=1)
            if raw:
                t.run(0, w[4], mask=1, addrs=[0x1000])
            else:
                t.run(0, w[4], mask=1, uregs={4: 0x1000, 5: Y})
            rows, *_ = t.analyze()
            kind = "async_write->generic_read" if raw else "generic_read->async_write"
            got = row(rows, kind, r[0], w[4])
            self.assertEqual(
                (got.get("sync_linked", 0), got.get("wait_linked", 0)), (0, 0)
            )

    def test_expect_tx_is_not_a_release(self):
        """Warp 1's only barrier operation is expect_tx (.RED.A0TR); warp 2 arrives."""
        t = Trace()
        t.code[0x3F00 // 16] = "NOP ;"
        r = t.program(
            [
                "LDS R4, [R2] ;",
                "FENCE.VIEW.ASYNC.S ;",
                "SYNCS.ARRIVE.TRANS64.RED.A0TR RZ, [UR8], R0 ;",
            ]
        )
        v = t.program(
            [
                "SYNCS.ARRIVE.TRANS64.A1T0 RZ, [UR8], RZ ;",
                "SYNCS.ARRIVE.TRANS64.RED.A0TX RZ, [UR8], R0 ;",
            ]
        )
        w = t.program(
            [*wait_code("R8"), "NOP ;", "UGETNEXTWORKID.BROADCAST [UR4], [UR5] ;"]
        )
        t.run(1, r[0], mask=1, addrs=[0x1000])
        t.run(1, r[1], mask=1)
        t.run(1, r[2], mask=1, addrs=[X], uregs={8: X})
        t.run(2, v[0], mask=1, addrs=[X], uregs={8: X})
        t.run(2, v[1], mask=1, addrs=[X], uregs={8: X})
        t.run(0, w[0], mask=1, addrs=[X], regs={6: 0})
        t.run(0, w[2], mask=1)
        t.run(0, w[3], mask=1, uregs={4: 0x1000, 5: Y})
        rows, *_ = t.analyze()
        got = row(rows, "generic_read->async_write", r[0], w[3])
        self.assertEqual(
            (got.get("sync_linked", 0), got.get("fence_and_link", 0)), (0, 0)
        )

    def test_remote_arrive_excludes_barrier(self):
        t = Trace()
        t.code[0x3F00 // 16] = "NOP ;"
        r = t.program(
            [
                "LDS R4, [R2] ;",
                "FENCE.VIEW.ASYNC.S ;",
                "SYNCS.ARRIVE.TRANS64.RED.A1T0 RZ, [R0+URZ], RZ ;",
            ]
        )
        w = t.program(
            [*wait_code("R8"), "NOP ;", "UGETNEXTWORKID.BROADCAST [UR4], [UR5] ;"]
        )
        t.run(1, r[0], mask=1, addrs=[0x1000])
        t.run(1, r[1], mask=1)
        t.run(1, r[2], mask=1, addrs=[X])
        t.run(0, w[0], mask=1, addrs=[X], regs={6: 0})
        t.run(0, w[2], mask=1)
        t.run(0, w[3], mask=1, uregs={4: 0x1000, 5: Y})
        rows, _, _, barriers = t.analyze()
        got = row(rows, "generic_read->async_write", r[0], w[3])
        self.assertEqual(got.get("sync_linked", 0), 0)
        self.assertIn("remote_arrive", barriers.problems[("mbar", (0, 0, 0), X)])

    def test_warpsync_must_include_acquire_lane(self):
        """Lane 0 waits; lane 1 syncs alone and reads."""
        t = Trace()
        t.code[0x3F00 // 16] = "NOP ;"
        w = t.program(["UGETNEXTWORKID.BROADCAST [UR4], [UR5] ;"])
        r = t.program([*wait_code(), "NOP ;", "WARPSYNC 0x2 ;", "LDS R4, [R2] ;"])
        t.run(0, w[0], mask=1, uregs={4: 0x1000, 5: X})
        t.run(1, r[0], addrs=[X], regs={6: 0}, mask=0x1)
        t.run(1, r[2], mask=0x1)
        t.run(1, r[3], mask=0x2)
        t.run(1, r[4], addrs=[0, 0x1000], mask=0x2)
        rows, *_ = t.analyze()
        got = row(rows, "async_write->generic_read", w[0], r[4])
        self.assertEqual(
            (got.get("wait_linked", 0), got["wait_lanes_unresolved"]), (0, 1)
        )

    def test_witness_keeps_chain(self):
        t = Trace()
        t.code[0x3F00 // 16] = "NOP ;"
        w = t.program(["UGETNEXTWORKID.BROADCAST [UR4], [UR5] ;"])
        r = t.program([*wait_code(), "NOP ;", "WARPSYNC.ALL ;", "LDS R4, [R2] ;"])
        t.run(0, w[0], mask=1, uregs={4: 0x1000, 5: X})
        t.run(1, r[0], addrs=[X], regs={6: 0}, mask=0x1)
        t.run(1, r[2], mask=0x1)
        t.run(1, r[3])
        t.run(1, r[4], addrs=[0x1000] * 32)
        rows, witness, *_ = t.analyze()
        key = ("async_write->generic_read", hex(w[0]), hex(r[4]))
        self.assertEqual(rows[key]["wait_linked"], 1)
        self.assertEqual([s[0] for s in witness[key][-1]], ["wait", "sync"])

    def test_raw_wait_lane_must_cover_reader(self):
        t = Trace()
        t.code[0x3F00 // 16] = "NOP ;"
        w = t.program(["UGETNEXTWORKID.BROADCAST [UR4], [UR5] ;"])
        r = t.program([*wait_code(), "NOP ;", "LDS R4, [R2] ;"])
        t.run(0, w[0], mask=1, uregs={4: 0x1000, 5: X})
        t.run(1, r[0], addrs=[X], regs={6: 0}, mask=0x1)
        t.run(1, r[2], mask=0x1)
        t.run(1, r[3], addrs=[0, 0x1000], mask=0x2)
        rows, *_ = t.analyze()
        got = row(rows, "async_write->generic_read", w[0], r[3])
        self.assertEqual(
            (got.get("wait_linked", 0), got["wait_lanes_unresolved"]), (0, 1)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
