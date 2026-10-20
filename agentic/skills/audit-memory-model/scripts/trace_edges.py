#!/usr/bin/env python3
# MIT License
# Copyright (c) Meta Platforms, Inc. and affiliates.
"""List candidate cross-path shared-memory edges from a CUTracer trace (NVIDIA sm_90+).

Input: one NDJSON trace from `cutracer trace -i reg_trace,mem_addr_trace`.

The output is a list of candidates and observations for a manual audit, never
a verdict. For each candidate pair of accesses to overlapping shared-memory
bytes through different paths, it reports which mechanisms it saw:

- fence_seen: a shared-memory proxy fence (FENCE.VIEW.ASYNC.S) executed by
  every lane of the first access, after it and before the release that links
  to the second warp (or before the first release, when no link is found);
- sync_linked: a release in the first warp that the second warp observably
  acquired before the second access, matched by phase on the same barrier
  (directly, or through one intermediate warp), with lane coverage at each
  hop; a match that holds by warp but not by lane is sync_lanes_unresolved;
- fence_and_link: fence_seen and sync_linked, with the fence itself joined by
  lane to the release (the release covers the access lanes, or a warp-wide
  sync after the fence includes both);
- wait_linked (async write, then generic read): a successful wait in the
  reader on the barrier the async operation completes, matched by ordinal,
  with lane coverage; otherwise wait_lanes_unresolved.

Lane coverage: an instruction covers an access if its active mask includes
the access's lanes. Otherwise a warp-wide sync between them must include both
sets of lanes: a BAR.SYNC's active mask, or a WARPSYNC's active mask
intersected with its member mask (which must be known and uniform). A release executed by one elected lane
after a read by all 32 lanes, with no such sync between them, is unresolved.
The JSON witness lists the matched chain (fence, release, acquire, sync) with
warps, PCs, and masks.

How the evidence is computed:

- Order within a warp comes from trace_index filtered to that warp (program
  order). Order across warps (receive order) is used only to discover
  candidate pairs; it never counts as evidence.
- A wait counts as successful only if the warp's executed instructions show
  the predicate-guarded branch after it falling through, with an unchanged
  active mask, and no other branch or write to the predicate between them. A warp's successive successful waits on one barrier must show
  alternating phase tokens (the wait's parity register); otherwise the barrier
  is excluded from ordinal matching.
- Phases come from the parity tokens. A waiter whose first parity is 1 starts
  on the pre-completed phase before phase 0. Arrive k of a warp belongs to
  phase k, and the k-th async operation completing a barrier completes phase k.
  These are protocol assumptions (one arrive per participating warp record per
  phase, no skipped phases): check them for the kernel you audit. They have
  been checked against kernel source only for one Blackwell GEMM's
  cluster-launch-control barriers and its named barriers 0 and 1; treat links
  through other barriers as assumption-based.

- An arrive with an arrival count (SYNCS.ARRIVE A1T0, ART0, or the plain
  arrive.expect_tx form) is treated as a release, and a successful wait as an
  acquire. ptxas emits the same SASS for relaxed and release arrives and for
  relaxed and acquire waits, so check the PTX for `.relaxed` on both.
  expect_tx and complete_tx forms (A0TR, A0TX) are not arrivals, and arrives
  through a cluster address (.RED) exclude that barrier from matching.

Shared addresses carry the CTA's rank in its cluster above bit 24. The script
takes each CTA's own rank to be the one most of its addresses carry and turns
those addresses into offsets. Accesses to another rank's shared memory are
listed as remote_shared.

Anything the script cannot resolve is listed as unresolved: descriptor-based
async reads, bulk copies whose extent is unknown, operands it cannot decode,
and barriers that fail the consistency checks.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import itertools
import json
import re
import sys

GRANULE = 16
FULL_MASK = 0xFFFFFFFF
SHARED_LIMIT = 1 << 24  # shared-window offsets
# Shared addresses carry the CTA's rank in its cluster above bit 24 (observed on sm_100a:
# rank-1 CTAs of a 2-CTA cluster record every shared address with bit 24 set).
CLUSTER_WINDOW = 16 * SHARED_LIMIT  # at most 16 CTAs per cluster
BRACKET = re.compile(r"\[([^\]]+)\]")
ACCESS_KINDS = (
    "generic_r",
    "generic_w",
    "generic_rw",
    "cpasync_w",
    "async_w",
    "async_r",
)

# mnemonic regex -> (kind, operand source, size in bytes; 0 = extent unknown)
OPCODES = [
    (re.compile(r"^LDSM\b"), "generic_r", "lanes", 16),
    (re.compile(r"^LDS\b"), "generic_r", "lanes", None),
    (re.compile(r"^STSM\b"), "generic_w", "lanes", 16),
    (re.compile(r"^STS\b"), "generic_w", "lanes", None),
    (re.compile(r"^(ATOMS|REDS)\b"), "generic_rw", "lanes", None),
    (re.compile(r"^LDGSTS\b"), "cpasync_w", "lanes", None),
    (re.compile(r"^UGETNEXTWORKID\b"), "async_w", "bracket0", 16),
    (re.compile(r"^UTMALDG\b"), "async_w", "bracket0", 0),
    (re.compile(r"^(UTMASTG|UTMAREDG)\b"), "async_r", "bracket0", 0),
    (re.compile(r"^(UBLKCP|UBLKRED)\b"), "async_unresolved", None, 0),
    # Descriptor-based MMA and copy forms: the sm_100 and sm_90 tables in NVIDIA's CUDA Binary
    # Utilities, plus UTCMMA as named by upstream Triton.
    (
        re.compile(
            r"^(UTCHMMA|UTCIMMA|UTCQMMA|UTCOMMA|UTCMMA|UTCCP|HGMMA|IGMMA|QGMMA|BGMMA)\b"
        ),
        "async_desc",
        None,
        0,
    ),
    (re.compile(r"^FENCE\.VIEW\.ASYNC\.S\b"), "fence_s", None, 0),
    # Observed with ptxas 12.8 and 13.0 for sm_100a: .A0TR (expect_tx) and .A0TX (complete_tx)
    # add no arrival; .RED forms target a barrier through a cluster address (mapa), possibly
    # in another CTA. Plain arrives (A1T0, ART0, no suffix for arrive.expect_tx) count.
    (re.compile(r"^SYNCS\.ARRIVE\.TRANS64\.RED\.A0"), "mbar_tx", None, 0),
    (re.compile(r"^SYNCS\.ARRIVE\.TRANS64\.A0"), "mbar_tx", None, 0),
    (re.compile(r"^SYNCS\.ARRIVE\.TRANS64\.RED\b"), "mbar_remote", None, 0),
    (re.compile(r"^SYNCS\.ARRIVE"), "mbar_arrive", None, 0),
    (re.compile(r"^SYNCS\.PHASECHK"), "mbar_wait", None, 0),
    (re.compile(r"^SYNCS\.EXCH\.64\b"), "mbar_init", None, 0),
    (re.compile(r"^BAR\.(SYNC|RED)\b"), "bar_sync", None, 0),
    (re.compile(r"^BAR\.ARV\b"), "bar_arv", None, 0),
    (re.compile(r"^WARPSYNC\b"), "warp_sync", None, 0),
]


def split_sass(sass: str) -> tuple[str | None, str, str]:
    """Return (guard predicate or None, mnemonic, operand text)."""
    s = sass.strip().rstrip(";").strip()
    guard = None
    if s.startswith("@"):
        guard, s = s.split(None, 1)
    parts = s.split(None, 1)
    return guard, parts[0], parts[1] if len(parts) > 1 else ""


def classify(mnemonic: str):
    for rx, kind, src, size in OPCODES:
        if rx.match(mnemonic):
            return kind, src, size
    return None, None, None


def access_size(mnemonic: str) -> int:
    for width, size in (
        (".128", 16),
        (".64", 8),
        (".U16", 2),
        (".S16", 2),
        (".U8", 1),
        (".S8", 1),
    ):
        if width in mnemonic:
            return size
    return 4


def bracket_value(operands: str, which: int, uregs: dict) -> int | None:
    """Value of a bracketed operand made of uniform registers plus an optional immediate."""
    found = BRACKET.findall(operands)
    if which >= len(found):
        return None
    total = 0
    for term in found[which].replace(" ", "").split("+"):
        if term.startswith("0x"):
            total += int(term, 16)
        elif term in ("URZ", "RZ"):
            continue
        elif term.startswith("UR") and term[2:].isdigit() and int(term[2:]) in uregs:
            total += uregs[int(term[2:])]
        else:
            return None  # thread register or unknown term
    return total


class Event:
    __slots__ = (
        "bar",
        "bar_count",
        "bar_id",
        "bar_set",
        "cta",
        "flags",
        "guard",
        "kind",
        "mask",
        "members",
        "mn",
        "ops",
        "pc",
        "pos",
        "ranges",
        "raw_bars",
        "sass",
        "src_bar",
        "success",
        "t",
        "token",
        "warp",
    )

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


def reg_value(rec, n, mask):
    """Value of register Rn in rec if recorded and equal across the active lanes, else None."""
    idx = rec.get("regs_indices") or []
    if n not in idx:
        return None
    vals = (rec.get("regs") or [])[idx.index(n)]
    seen = {vals[i] for i in range(min(32, len(vals))) if mask >> i & 1}
    return seen.pop() if len(seen) == 1 else None


def build_events(path: str):
    """Merge each dynamic instruction's mem_addr_trace and reg_trace records into one event.

    In CUTracer traces, an instruction's mem_addr_trace record arrives just before its
    reg_trace record in the same warp."""
    per_warp = collections.defaultdict(list)
    pc_sass = {}  # static listing: pc (int) -> (sass, guard, mnemonic, operands)
    with open(path) as f:
        meta = json.loads(f.readline())
        static = {}
        listing = {}
        for oid, v in meta["instructions"].items():
            guard, mn, ops = split_sass(v["sass"])
            static[oid] = (v["sass"], guard, mn, ops) + classify(mn)
            listing[int(oid) * 16] = (v["sass"], guard, mn, ops)
        listing_ok = [True]

        def emit(key, rec, mem):
            sass, guard, mn, ops, kind, _src, size = static[str(rec["opcode_id"])]
            if int(rec["pc"], 16) != int(rec["opcode_id"]) * 16:
                listing_ok[0] = False
            uregs = dict(zip(rec.get("uregs_indices") or [], rec.get("uregs") or []))
            mask = int((mem or rec).get("active_mask", "0xffffffff"), 16)
            ev = Event(
                cta=key[0],
                warp=key[1],
                pos=len(per_warp[key]),
                t=rec.get("trace_index"),
                pc=rec["pc"],
                sass=sass,
                guard=guard,
                mn=mn,
                ops=ops,
                kind=kind,
                ranges=[],
                mask=mask,
                flags=set(),
            )
            lanes = []
            if mem is not None:
                lanes = [
                    a for i, a in enumerate(mem.get("addrs") or []) if mask >> i & 1
                ]
            if kind in ("generic_r", "generic_w", "generic_rw", "cpasync_w"):
                shared = [a for a in lanes if 0 <= a < CLUSTER_WINDOW]
                if lanes and not shared:
                    ev.kind = None  # a global access
                elif not shared:
                    ev.flags.add("operand_unresolved")
                sz = size or access_size(mn)
                ev.ranges = sorted({(a, a + sz) for a in shared})
                if mn.startswith(("LDSM", "STSM")):
                    ev.flags.add("matrix_footprint_approx")
            elif kind in ("async_w", "async_r"):
                base = bracket_value(ops, 0, uregs)
                if base is None or not (0 <= base < CLUSTER_WINDOW):
                    ev.flags.add("operand_unresolved")
                else:
                    ev.ranges = [(base, base + (size or 1))]
                    if not size:
                        ev.flags.add("footprint_unresolved")
                if mn.startswith("UGETNEXTWORKID"):
                    ev.src_bar = bracket_value(ops, 1, uregs)
            elif kind in (
                "mbar_arrive",
                "mbar_wait",
                "mbar_init",
                "mbar_tx",
                "mbar_remote",
            ):
                addrs = sorted({a for a in lanes if 0 <= a < CLUSTER_WINDOW})
                if len(addrs) > 1:
                    # Lanes used different barriers: matching by warp would credit lanes
                    # with another lane's barrier, so every barrier involved is excluded.
                    ev.bar_set = addrs
                    ev.flags.add("barrier_nonuniform")
                else:
                    ev.bar = addrs[0] if addrs else bracket_value(ops, 0, uregs)
                if kind == "mbar_wait":
                    m = re.search(r"\],\s*R(\d+)", ops)
                    idx = rec.get("regs_indices") or []
                    if m and int(m.group(1)) in idx:
                        vals = (rec.get("regs") or [])[idx.index(int(m.group(1)))]
                        toks = {
                            vals[i] for i in range(min(32, len(vals))) if mask >> i & 1
                        }
                        # A token that differs across lanes is not a valid encoding.
                        ev.token = toks.pop() if len(toks) == 1 else "nonuniform"
            elif kind == "warp_sync":
                # Member mask: .ALL, an immediate, or a register that must hold the same value
                # in every active lane (per-lane groups, as in __syncwarp(mask) with a runtime
                # mask, are not modeled). Unknown members join nothing.
                o = ops.strip()
                if mn.startswith("WARPSYNC.ALL"):
                    ev.members = FULL_MASK
                elif re.fullmatch(r"0x[0-9a-fA-F]+|\d+", o):
                    ev.members = int(o, 0)
                elif re.fullmatch(r"R\d+", o):
                    ev.members = reg_value(rec, int(o[1:]), mask) or 0
                else:
                    ev.members = 0
            elif kind in ("bar_sync", "bar_arv"):
                # Operands are the barrier id and an optional thread count, each an immediate
                # or a register. A register is read from reg_trace; if its value is missing or
                # differs across lanes, the operand is unresolved.
                vals = []
                for o in [o.strip() for o in ops.split(",") if o.strip()][:2]:
                    if re.fullmatch(r"0x[0-9a-fA-F]+|\d+", o):
                        vals.append(int(o, 0))
                    elif re.fullmatch(r"R\d+", o):
                        vals.append(reg_value(rec, int(o[1:]), mask))
                    else:
                        vals.append(None)
                if not vals or None in vals:
                    ev.flags.add("operand_unresolved")
                ev.bar_id = vals[0] if vals else None
                ev.bar_count = vals[1] if len(vals) > 1 else None
            per_warp[key].append(ev)

        pending_mem = {}
        for line in f:
            rec = json.loads(line)
            typ = rec.get("type")
            if typ not in ("reg_trace", "mem_addr_trace"):
                continue
            key = (tuple(rec["cta"]), rec["warp"])
            if typ == "mem_addr_trace":
                old = pending_mem.get(key)
                if old is not None and old["pc"] == rec["pc"]:
                    # Instructions with several memory operands produce one record per operand.
                    mask = int(rec.get("active_mask", "0xffffffff"), 16)
                    old.setdefault("extra_addrs", []).extend(
                        a for i, a in enumerate(rec.get("addrs") or []) if mask >> i & 1
                    )
                    continue
                if old is not None:
                    emit(key, pending_mem.pop(key), old)
                pending_mem[key] = rec
            else:
                mem = pending_mem.pop(key, None)
                if mem is not None and mem["pc"] != rec["pc"]:
                    emit(key, mem, mem)
                    mem = None
                emit(key, rec, mem)
        for key, mem in pending_mem.items():
            emit(key, mem, mem)
    # The metadata lists every instruction of the root function, including ones that never
    # executed, at pc = opcode_id * 16. Use it as the static listing only if that held throughout.
    if listing_ok[0]:
        pc_sass = listing
    rebase_cluster_addresses(per_warp)
    return per_warp, pc_sass


def rebase_cluster_addresses(per_warp):
    """Turn each CTA's own shared addresses into offsets in its shared window.

    A CTA's own rank is the rank most of its CTA-local addresses carry: generic shared
    accesses, cp.async destinations, and mbarrier waits, which can only address the
    issuing CTA. Async operations can target other CTAs, so they vote only when a CTA
    has no CTA-local address. An access or a
    barrier with another rank names another CTA's shared memory: it moves above the
    cluster window (CLUSTER_WINDOW + address), so it overlaps none of this CTA's
    offsets, and accesses are flagged remote_shared (unresolved)."""
    local_kinds = ("generic_r", "generic_w", "generic_rw", "cpasync_w", "mbar_wait")
    votes = collections.defaultdict(
        lambda: (collections.Counter(), collections.Counter())
    )
    for (cta, _w), evs in per_warp.items():
        for ev in evs:
            vote = votes[cta][0 if ev.kind in local_kinds else 1]
            for lo, _hi in ev.ranges or ():
                vote[lo // SHARED_LIMIT] += 1
            if ev.bar is not None:
                vote[ev.bar // SHARED_LIMIT] += 1
    for (cta, _w), evs in per_warp.items():
        local_votes, other_votes = votes[cta]
        if not (local_votes or other_votes):
            continue
        own = (local_votes or other_votes).most_common(1)[0][0]
        base = own * SHARED_LIMIT

        def local(a):
            if a is None or not 0 <= a < CLUSTER_WINDOW:
                return a
            return a - base if a // SHARED_LIMIT == own else CLUSTER_WINDOW + a

        for ev in evs:
            if ev.kind == "mbar_remote":
                # Keep the encoded target (rank and offset): its exclusion must reach the
                # target CTA, whose name for the barrier differs from the issuer's.
                ev.raw_bars = [
                    a for a in [ev.bar, *(ev.bar_set or ())] if a is not None
                ]
            if ev.ranges:
                if any(lo // SHARED_LIMIT != own for lo, _ in ev.ranges):
                    ev.flags.add("remote_shared")
                ev.ranges = [(local(lo), local(lo) + (hi - lo)) for lo, hi in ev.ranges]
            ev.bar, ev.src_bar = local(ev.bar), local(ev.src_bar)
            if ev.bar_set:
                ev.bar_set = [local(a) for a in ev.bar_set]


def guarded_branch(pc_sass, wait_pc, pred, window=32):
    """First branch after wait_pc guarded on pred, if the predicate is not redefined before it."""
    pc = wait_pc
    for _ in range(window):
        pc += 16
        info = pc_sass.get(pc)
        if info is None:
            return None
        _sass, guard, mn, ops = info
        if mn.split(".")[0] == "BRA":
            # Only a plain branch guarded on pred qualifies. Any other branch (BRA.U UP0, an
            # unguarded BRA, a branch on another predicate) can skip it, so decoding stops.
            if mn == "BRA" and guard in ("@" + pred, "@!" + pred) and "," not in ops:
                return (pc, guard, ops.split()[0] if ops else None)
            return None
        if mn.startswith(
            (
                "BR",
                "JM",
                "CALL",
                "RET",
                "EXIT",
                "KILL",
                "BPT",
                "BSSY",
                "BSYNC",
                "WARPSYNC",
            )
        ):
            return None  # other control flow (BRX, BRXU, BREAK, JMX, ...) may skip the branch
        operands = [o.strip().lstrip("!") for o in ops.split(",")]
        if pred in operands or "PR" in operands or "UPR" in operands:
            # The predicate may be redefined: as a named operand (destination predicates come
            # first in SASS, but checking every position is the safe choice), or through the
            # whole predicate register (R2P PR, ...).
            return None
    return None


def mark_wait_success(per_warp, pc_sass):
    """Decide whether each wait succeeded from the executed path around the branch on its predicate.

    Instructions whose guard predicate is false leave no trace record. So if the branch
    "@!P BRA retry" left no record and execution continued past it, P was true and the wait
    succeeded. If the branch left a record with the wait's full mask, the guard was true in every
    lane and the wait failed. A change in the active mask (lanes diverging) leaves the result
    unresolved."""
    for evs in per_warp.values():
        for i, ev in enumerate(evs):
            if ev.kind != "mbar_wait":
                continue
            ev.success = None
            m = re.match(r"(P\d+)", ev.ops)
            if not m:
                continue
            br = guarded_branch(pc_sass, int(ev.pc, 16), m.group(1))
            if br is None:
                continue
            bra_pc, guard, _target = br
            on_failure = guard.startswith("@!")  # "@!P BRA" leaves the success path
            j = i + 1
            while j < len(evs) and int(ev.pc, 16) < int(evs[j].pc, 16) < bra_pc:
                j += 1  # straight-line instructions between the wait and the branch
            # Lanes must not diverge on the way. A guarded instruction's own predicate may
            # shrink its mask, so only unguarded instructions are compared.
            if j >= len(evs) or any(
                e.mask != ev.mask for e in evs[i + 1 : j + 1] if e.guard is None
            ):
                continue
            nxt = evs[j]
            # A recorded guarded branch executed in the lanes whose guard was true. With the
            # wait's full mask, the guard was true in every lane, so the branch was taken.
            taken = int(nxt.pc, 16) == bra_pc
            if taken and nxt.mask != ev.mask:
                continue  # some lanes took the branch, others did not
            if not taken and not (bra_pc < int(nxt.pc, 16) <= bra_pc + 16 * 8):
                continue
            ev.success = (not taken) if on_failure else taken


class Barriers:
    """Per-CTA ordinal matching of releases to acquires."""

    def __init__(self, per_warp):
        self.arrive = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )  # (cta,bar) -> warp -> [pos]
        self.wait = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )  # (cta,bar) -> warp -> [(pos,tok)]
        self.issue = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )  # (cta,bar) -> warp -> [pos]
        self.bar = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )  # (cta,id,count) -> warp -> [(pos,kind)]
        self.inits = collections.Counter()
        for (cta, warp), evs in per_warp.items():
            for ev in evs:
                if ev.kind == "mbar_arrive" and ev.bar is not None:
                    self.arrive[(cta, ev.bar)][warp].append(ev.pos)
                elif ev.kind == "mbar_wait" and ev.bar is not None and ev.success:
                    self.wait[(cta, ev.bar)][warp].append((ev.pos, ev.token))
                elif ev.kind == "mbar_init" and ev.bar is not None:
                    self.inits[(cta, ev.bar)] += 1
                elif ev.kind == "async_w" and ev.src_bar is not None:
                    self.issue[(cta, ev.src_bar)][warp].append(ev.pos)
                elif ev.kind in ("bar_sync", "bar_arv"):
                    self.bar[(cta, ev.bar_id, ev.bar_count)][warp].append(
                        (ev.pos, ev.kind)
                    )
        self.problems = {}
        for (cta, _warp), evs in per_warp.items():
            for ev in evs:
                if (
                    ev.kind in ("bar_sync", "bar_arv")
                    and "operand_unresolved" in ev.flags
                ):
                    # An unresolved id or count could belong to any named barrier.
                    self.problems.setdefault(("bar", cta), set()).add(
                        "operand_unresolved"
                    )
                if ev.kind == "mbar_remote":
                    # The address may name a barrier in another CTA of the cluster, so arrive
                    # ordinals for it cannot be counted per CTA. Exclude it in every CTA under
                    # both names it can have there: the offset (in the target CTA) and the
                    # remote name (in any other CTA).
                    raw = ev.raw_bars
                    if raw is None:
                        raw = [
                            a for a in [ev.bar, *(ev.bar_set or ())] if a is not None
                        ]
                    names = set()
                    for a in raw:
                        names.add(a)
                        if 0 <= a < CLUSTER_WINDOW:
                            names |= {a % SHARED_LIMIT, CLUSTER_WINDOW + a}
                    for a in names:
                        for c in {k[0] for k in per_warp}:
                            self.problems.setdefault(("mbar", c, a), set()).add(
                                "remote_arrive"
                            )
                for a in ev.bar_set or ():
                    self.problems.setdefault(("mbar", cta, a), set()).add(
                        "nonuniform_lane_addresses"
                    )
        # Phase observed by each successful wait. The token is the wait's parity operand,
        # observed as 0 or 0x80000000 (parity in bit 31). A waiter whose first parity is 1
        # starts on the pre-completed phase before phase 0 (a "start empty" producer), so its
        # n-th successful wait observes phase n - 1. Arrives are numbered from phase 0.
        self.phase = {}  # (cta, bar, warp) -> {phase: wait pos}
        for key, waiters in self.wait.items():
            for warp, ws in waiters.items():
                toks = [t for _, t in ws]
                if any(t not in (0, 0x80000000) for t in toks):
                    self.problems.setdefault(("mbar",) + key, set()).add(
                        "token_encoding_unknown"
                    )
                    continue
                if any(a == b for a, b in itertools.pairwise(toks)):
                    self.problems.setdefault(("mbar",) + key, set()).add(
                        "phase_token_repeat"
                    )
                    continue
                origin = -1 if toks and toks[0] == 0x80000000 else 0
                self.phase[key + (warp,)] = {
                    origin + n: pos for n, (pos, _) in enumerate(ws)
                }
            if self.inits[key] > 1:
                self.problems.setdefault(("mbar",) + key, set()).add("reinitialized")
        for key, users in self.bar.items():
            if len({len(v) for v in users.values()}) > 1:
                self.problems.setdefault(("bar",) + key, set()).add(
                    "unequal_invocations"
                )
            need = key[2] // 32 if key[2] else None
            if need is not None and len(users) > need:
                self.problems.setdefault(("bar",) + key, set()).add(
                    "more_warps_than_count"
                )

    def mbar_ok(self, cta, bar):
        return ("mbar", cta, bar) not in self.problems

    def links_from(self, cta, warp, release_pos):
        """Yield (other warp, acquire pos) for acquires that ordinal-match the release at release_pos."""
        for (c, bar), arrivers in self.arrive.items():
            if (
                c != cta
                or release_pos not in arrivers.get(warp, ())
                or not self.mbar_ok(cta, bar)
            ):
                continue
            k = arrivers[warp].index(release_pos)  # this arrive's phase, counted from 0
            for w2 in self.wait.get((cta, bar), {}):
                pos = self.phase.get((cta, bar, w2), {}).get(k)
                if pos is not None:
                    yield w2, pos
        for key, users in self.bar.items():
            if (
                key[0] != cta
                or warp not in users
                or ("bar",) + key in self.problems
                or ("bar", cta) in self.problems
            ):
                continue
            mine = [p for p, _ in users[warp]]
            if release_pos not in mine:
                continue
            k = mine.index(release_pos)
            for w2, lst in users.items():
                if w2 != warp and k < len(lst) and lst[k][1] == "bar_sync":
                    yield w2, lst[k][0]


def releases_after(evs, pos):
    for ev in evs[pos + 1 :]:
        if ev.kind in ("mbar_arrive", "bar_sync", "bar_arv"):
            yield ev


def covers(mask, lanes):
    return (mask & lanes) == lanes


def join(evs, lo, hi, lanes, anchor):
    """Lane evidence that the lanes in `lanes` are ordered with the instruction `anchor`, which
    sits between positions lo and hi in this warp. Returns [] if anchor's mask covers them, [sync]
    if a warp-wide sync (BAR.SYNC or WARPSYNC) between lo and hi covers both them and anchor's
    lanes, and None otherwise."""
    if covers(anchor.mask, lanes):
        return []
    need = lanes | anchor.mask
    for e in evs[lo + 1 : hi]:
        if e.kind == "bar_sync" and covers(e.mask, need):
            return [e]
        if e.kind == "warp_sync" and covers(e.mask & (e.members or 0), need):
            return [e]
    return None


def step(role, ev):
    return (role, ev.warp, ev.pc, hex(ev.mask), ev.t)


def handshake(per_warp, barriers, cta, w1, p1, w2, p2):
    """Evidence that access (w1, p1) is ordered before (w2, p2): a release in w1 after p1 that w2
    acquires before p2, directly or through one intermediate warp. All releases after p1 are tried,
    and a link whose release follows a covering proxy fence is preferred.

    Lanes are tracked at every hop: the fence must be executed by every lane of the first access;
    the release must cover those lanes, or a warp-wide sync between them must cover both the access
    lanes and the release lanes; likewise between each acquire and the next release or the second
    access. A link that holds only by warp, not by lane, is reported as lanes_unresolved.
    The fence counts toward fence_and_link only if it is itself joined to the release: the
    release covers the access lanes, or a warp-wide sync after the fence includes both.
    Returns (fence_seen, fence_joined, link, chain): link is one of "linked", "linked_depth2",
    "lanes_unresolved", None; chain lists the (role, warp, pc, mask, trace_index) steps of the
    chosen link.
    """
    evs1 = per_warp[(cta, w1)]
    evs2 = per_warp[(cta, w2)]
    x_lanes, y_lanes = evs1[p1].mask, evs2[p2].mask
    fences = [e for e in evs1 if e.kind == "fence_s" and covers(e.mask, x_lanes)]
    i = bisect.bisect_right([e.pos for e in fences], p1)
    first_fence = fences[i] if i < len(fences) else None
    fallback_fence = None
    best = best_score = None  # best: (fence, fence_joined, link, chain)
    rank = {"lanes_unresolved": 0, "linked_depth2": 1, "linked": 2}
    for rel in releases_after(evs1, p1):
        fence = first_fence is not None and first_fence.pos < rel.pos
        if fallback_fence is None:
            fallback_fence = fence
        j_rel = join(evs1, p1, rel.pos, x_lanes, rel)
        head, joined = [], False
        for f in fences[i:]:
            if f.pos >= rel.pos:
                break
            j_f = join(evs1, f.pos, rel.pos, x_lanes, rel)
            if j_f is not None:
                head, joined, j_rel = [step("fence", f)], True, j_f
                break
        if fence and not joined:
            head = [step("fence", first_fence)]
        link = chain = None
        for wx, acq in barriers.links_from(cta, w1, rel.pos):
            if wx == w2 and acq < p2:
                j_acq = join(evs2, acq, p2, y_lanes, evs2[acq])
                ok = j_rel is not None and j_acq is not None
                link = "linked" if ok else "lanes_unresolved"
                chain = (
                    head
                    + [step("sync", e) for e in j_rel or []]
                    + [step("release", rel), step("acquire", evs2[acq])]
                    + [step("sync", e) for e in j_acq or []]
                )
                break
        if link != "linked":
            for wx, acq in barriers.links_from(cta, w1, rel.pos):
                evx = per_warp[(cta, wx)]
                for rel2 in releases_after(evx, acq):
                    j_mid = join(evx, acq, rel2.pos, rel2.mask, evx[acq])
                    for wy, acq2 in barriers.links_from(cta, wx, rel2.pos):
                        if wy == w2 and acq2 < p2:
                            j_acq = join(evs2, acq2, p2, y_lanes, evs2[acq2])
                            ok = None not in (j_rel, j_mid, j_acq)
                            cand = "linked_depth2" if ok else "lanes_unresolved"
                            if link is None or rank[cand] > rank[link]:
                                link = cand
                                chain = (
                                    head
                                    + [step("sync", e) for e in j_rel or []]
                                    + [step("release", rel), step("acquire", evx[acq])]
                                    + [step("sync", e) for e in j_mid or []]
                                    + [
                                        step("release", rel2),
                                        step("acquire", evs2[acq2]),
                                    ]
                                    + [step("sync", e) for e in j_acq or []]
                                )
        # Prefer any lane-resolved link, then a joined fence, then a direct link.
        score = None if link is None else (rank[link] > 0, joined, rank[link])
        if score is not None and (best is None or score > best_score):
            best, best_score = (fence, joined, link, chain), score
            if joined and link == "linked":
                break
    if best is not None:
        return best
    return bool(fallback_fence), False, None, []


def wait_link(per_warp, barriers, cta, ww, wpos, bar, rw, rpos):
    """Async write (ww, wpos) completing on bar, then generic read (rw, rpos). Returns (result,
    chain): result is "linked" if rw's successful wait on the phase this write completes comes
    before the read and covers the reading lanes (directly or through a warp-wide sync that also
    covers the waiting lanes), "lanes_unresolved" if only the warp matches, False if no such wait
    precedes the read, None if not decidable. Assumes the k-th async op on bar completes phase k.
    """
    if bar is None or not barriers.mbar_ok(cta, bar):
        return None, []
    issues = barriers.issue[(cta, bar)].get(ww, [])
    if wpos not in issues:
        return None, []
    pos = barriers.phase.get((cta, bar, rw), {}).get(issues.index(wpos))
    if pos is None or pos >= rpos:
        return False, []
    evs = per_warp[(cta, rw)]
    j = join(evs, pos, rpos, evs[rpos].mask, evs[pos])
    chain = [step("wait", evs[pos])] + [step("sync", e) for e in j or []]
    return ("linked" if j is not None else "lanes_unresolved"), chain


def overlaps(a, b):
    return any(lo1 < hi2 and lo2 < hi1 for lo1, hi1 in a for lo2, hi2 in b)


MAX_LATER = (
    64  # later accesses paired with one access per granule; truncation is reported
)

GENERIC_READ = ("generic_r", "generic_rw")
GENERIC_WRITE = ("generic_w", "generic_rw", "cpasync_w")


def pair_kind(a, b):
    """Edge kind for access a followed by access b, or None if the pair needs no check here."""
    if a.kind in GENERIC_READ and b.kind == "async_w":
        return "generic_read->async_write"
    if a.kind in GENERIC_WRITE and b.kind == "async_r":
        return "generic_write->async_read"
    if a.kind in GENERIC_WRITE and b.kind == "async_w":
        return "generic_write->async_write"
    if a.kind == "async_w" and b.kind in GENERIC_READ:
        return "async_write->generic_read"
    if a.kind == "async_r" and b.kind in GENERIC_WRITE:
        return "async_read->generic_write"
    if a.kind.startswith("async") and b.kind.startswith("async"):
        return "async->async"
    if a.kind == "cpasync_w" and b.kind in GENERIC_READ:
        return "cpasync_write->generic_read"
    return None


def candidate_pairs(seq, capped):
    """Pair each access with every later access of a relevant kind in the same granule (receive
    order, discovery only). No access is retired by an intervening one: retirement would need an
    ordering chain, which is what the evidence checks establish afterwards."""
    for i, a in enumerate(seq):
        n = 0
        for b in seq[i + 1 :]:
            kind = pair_kind(a, b)
            if kind is None:
                continue
            if n == MAX_LATER:
                capped[(a.pc, a.kind)] += 1
                break
            n += 1
            yield kind, a, b


def analyze(path: str):
    per_warp, pc_sass = build_events(path)
    mark_wait_success(per_warp, pc_sass)
    barriers = Barriers(per_warp)
    unresolved = collections.Counter()
    by_cta = collections.defaultdict(list)
    for (cta, _warp), evs in per_warp.items():
        for ev in evs:
            if ev.kind in ("async_desc", "async_unresolved"):
                unresolved[
                    (
                        ev.pc,
                        ev.sass.strip(),
                        "descriptor" if ev.kind == "async_desc" else "unsupported",
                    )
                ] += 1
            elif ev.kind in ACCESS_KINDS and ev.flags & {
                "operand_unresolved",
                "footprint_unresolved",
                "remote_shared",
            }:
                unresolved[
                    (
                        ev.pc,
                        ev.sass.strip(),
                        ",".join(
                            sorted(
                                ev.flags
                                & {
                                    "operand_unresolved",
                                    "footprint_unresolved",
                                    "remote_shared",
                                }
                            )
                        ),
                    )
                ] += 1
            if ev.kind == "mbar_wait" and ev.success is None:
                unresolved[(ev.pc, ev.sass.strip(), "wait_success_unresolved")] += 1
            if ev.kind in ACCESS_KINDS and ev.ranges:
                by_cta[cta].append(ev)

    rows = collections.defaultdict(collections.Counter)
    witness = {}
    capped = collections.Counter()
    for cta, evs in by_cta.items():
        evs.sort(key=lambda e: e.t or 0)
        index = collections.defaultdict(list)
        for e in evs:
            for lo, hi in e.ranges:
                for g in range(lo // GRANULE, (hi - 1) // GRANULE + 1):
                    index[g].append(e)
        seen = set()
        for seq in index.values():
            for kind, x, y in candidate_pairs(seq, capped):
                sig = (kind, x.warp, x.pos, y.warp, y.pos)
                if sig in seen or not overlaps(x.ranges, y.ranges):
                    continue
                seen.add(sig)
                key = (kind, x.pc, y.pc)
                r = rows[key]
                r["occurrences"] += 1
                # The witness is the first occurrence, replaced by the first one with a link chain.
                this = [
                    list(cta),
                    x.warp,
                    x.t,
                    hex(x.mask),
                    y.warp,
                    y.t,
                    hex(y.mask),
                    [],
                ]
                witness.setdefault(key, this)
                if "footprint_unresolved" in (x.flags | y.flags):
                    r["footprint_unresolved"] += 1
                if x.kind == "cpasync_w" or kind == "async_read->generic_write":
                    r["completion_unchecked"] += 1
                if x.warp == y.warp:
                    r["same_warp"] += 1
                    if x.mask & ~y.mask:
                        # Lanes that made the first access are not the lanes that made the
                        # second: different threads of one warp, ordered only by a warp sync.
                        r["same_warp_lanes_unresolved"] += 1
                    elif any(
                        e.kind == "fence_s"
                        and x.pos < e.pos < y.pos
                        and (e.mask & x.mask) == x.mask
                        for e in per_warp[(cta, x.warp)]
                    ):
                        r["same_warp_fence_seen"] += 1
                    continue
                if kind == "async->async":
                    r["manual"] += 1
                elif kind == "async_write->generic_read":
                    wl, chain = wait_link(
                        per_warp, barriers, cta, x.warp, x.pos, x.src_bar, y.warp, y.pos
                    )
                    r[
                        {
                            None: "barrier_unknown",
                            False: "no_wait_link",
                            "linked": "wait_linked",
                            "lanes_unresolved": "wait_lanes_unresolved",
                        }[wl]
                    ] += 1
                    if chain and not witness[key][-1]:
                        this[-1] = chain
                        witness[key] = this
                else:
                    fence, fence_joined, link, chain = handshake(
                        per_warp, barriers, cta, x.warp, x.pos, y.warp, y.pos
                    )
                    linked = link in ("linked", "linked_depth2")
                    if chain and not witness[key][-1]:
                        this[-1] = chain
                        witness[key] = this
                    if kind != "async_read->generic_write":
                        r["fence_seen"] += fence
                        r["fence_and_link"] += fence_joined and linked
                    r["sync_linked" if linked else "no_sync_link"] += 1
                    if link == "linked_depth2":
                        r["linked_via_intermediate"] += 1
                    if link == "lanes_unresolved":
                        r["sync_lanes_unresolved"] += 1
                    if not linked and any(
                        e.kind == "mbar_wait" and e.success is None and e.pos < y.pos
                        for e in per_warp[(cta, y.warp)]
                    ):
                        r["unresolved_wait_in_second_warp"] += 1
    for (pc, kind), n in capped.items():
        unresolved[(pc, kind, f"discovery_capped_at_{MAX_LATER}")] += n
    return rows, witness, unresolved, barriers


COLUMNS = [
    "occurrences",
    "fence_seen",
    "sync_linked",
    "fence_and_link",
    "no_sync_link",
    "linked_via_intermediate",
    "wait_linked",
    "wait_lanes_unresolved",
    "sync_lanes_unresolved",
    "no_wait_link",
    "barrier_unknown",
    "same_warp",
    "same_warp_fence_seen",
    "same_warp_lanes_unresolved",
    "unresolved_wait_in_second_warp",
    "footprint_unresolved",
    "completion_unchecked",
    "manual",
]


WAR_KINDS = ("generic_read->async_write", "async_read->generic_write")
NO_REUSE_WARNING = (
    "No write-after-read candidate in this trace: no CTA wrote a buffer through one "
    "path after reading it through another. If the kernel reuses buffers, this launch "
    "was too small for a CTA to reuse them, and reuse edges are missing from this "
    "list. Trace a launch with more work per CTA (for a persistent kernel, more tiles "
    "than SMs)."
)


def reuse_warning(rows):
    """The no-reuse warning if no candidate row is a read followed by a write, else None."""
    return None if any(k[0] in WAR_KINDS for k in rows) else NO_REUSE_WARNING


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "trace", help="NDJSON trace from `cutracer trace -i reg_trace,mem_addr_trace`"
    )
    p.add_argument("--json", help="also write the results as JSON to this path")
    args = p.parse_args()
    rows, witness, unresolved, barriers = analyze(args.trace)

    print("Candidate edges (observations, not verdicts):")
    for key in sorted(rows):
        r = rows[key]
        print(
            f"  {key[0]:28s} {key[1]:>8s} -> {key[2]:<8s} "
            + ", ".join(f"{c}={r[c]}" for c in COLUMNS if r[c])
        )
    warning = reuse_warning(rows)
    if warning:
        print(f"\nWarning: {warning}")
    if unresolved:
        print("\nUnresolved operations (check by hand):")
        for (pc, sass, why), n in sorted(unresolved.items()):
            print(f"  {pc:>8s} x{n:<6d} [{why}] {sass}")
    if barriers.problems:
        print("\nBarriers excluded from ordinal matching:")
        for key, why in sorted(barriers.problems.items(), key=str):
            print(f"  {key}: {', '.join(sorted(why))}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                {
                    "warnings": [warning] if warning else [],
                    "edges": [
                        dict(
                            kind=k[0],
                            first_pc=k[1],
                            second_pc=k[2],
                            **rows[k],
                            witness=dict(
                                zip(
                                    (
                                        "cta",
                                        "first_warp",
                                        "first_trace_index",
                                        "first_mask",
                                        "second_warp",
                                        "second_trace_index",
                                        "second_mask",
                                        "chain",
                                    ),
                                    witness.get(k, ()),
                                )
                            ),
                        )
                        for k in sorted(rows)
                    ],
                    "unresolved": [
                        {"pc": pc, "sass": s, "reason": w, "count": n}
                        for (pc, s, w), n in unresolved.items()
                    ],
                    "excluded_barriers": [
                        {"barrier": str(k), "reasons": sorted(v)}
                        for k, v in barriers.problems.items()
                    ],
                },
                f,
                indent=1,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
