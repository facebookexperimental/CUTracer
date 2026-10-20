# What CUTracer traces show (NVIDIA)

This file describes what a `cutracer trace -i reg_trace,mem_addr_trace`
capture contains, what it can establish, and what it cannot. Check it against
the CUTracer version you use; record the version in the audit.

## Record order and timestamps

- `trace_index` counts records per kernel launch in the order the host
  received them from the device channel. `timestamp` is the host clock when
  the host processed the record. Neither is a device timestamp.
- Within one warp, records arrive in program order, so the order of one warp's
  instructions is reliable.
- Across warps, the order is the order in which records reached the channel.
  It approximates the device order but does not establish execution order or
  memory ordering. Use it to pair candidate accesses, never as proof that one
  access happened before another.

## Observed record structure

Observed with CUTracer at commit `b9cdb15` on sm_100a; re-check on your version.

- An instruction's `mem_addr_trace` record arrives just before its
  `reg_trace` record in the same warp. An instruction with several memory
  operands (for example `UGETNEXTWORKID`, which has the response address and
  the mbarrier) produces one `mem_addr_trace` record per operand.
- An instruction whose guard predicate is false for every active lane leaves
  no record. A skipped `@!P BRA retry` after a `try_wait` therefore shows up as
  a jump straight to the branch's fall-through.
- For a kernel without calls, `pc = opcode_id * 16`, so the trace's first line
  is a complete static listing, including instructions that never executed.
- The `try_wait` parity operand is recorded in `reg_trace` as `0` or
  `0x80000000` (parity in bit 31). Parity gives the phase only modulo 2. In the
  validated CLC and TMA pipelines, a waiter whose first successful wait has
  parity 1 started on the pre-completed phase before phase 0 (a producer that
  starts with the buffer empty). In another protocol, parity 1 can also mean
  phase 1 or any odd phase, so establish each barrier's first logical phase
  from the source before relying on this. Waiters on one barrier can start at
  different phases.

## What each mode records

| Data | Source | Notes |
|---|---|---|
| Every executed instruction of each warp, including fences, barriers, and arrives | `reg_trace` | Includes uniform-register values (`uregs`, `uregs_indices`) and register values. The trace's first line maps each `opcode_id` to its SASS text. |
| Per-lane addresses of ordinary shared and global accesses | `mem_addr_trace` (`addrs[32]`, `active_mask`) | Shared addresses are offsets in the CTA's shared window. Kernel buffers start after a reserved region (observed at `0x400`). |
| mbarrier operations (`SYNCS.*`) | `mem_addr_trace` | Carry the mbarrier's address, so waits and arrives can be tied to a barrier. Whether a `try_wait` succeeded is not recorded directly. It can be shown by the executed control flow when all of these hold: the next branch tests the predicate this wait produced, nothing in between changes that predicate, the branch's taken and fall-through targets separate failure from success, the predicate and branch are uniform across the lanes that matter (or lane-level evidence covers them), and the warp's next executed instruction is the fall-through. The wait's parity operand (a register value in `reg_trace`) gives the checked phase modulo 2; with a validated first phase and no skipped phases, the sequence of parities identifies the phase. Any other pattern leaves success unresolved. |
| `UGETNEXTWORKID` (CLC response) | `mem_addr_trace` and `reg_trace` | Its `mem_addr_trace` lanes can carry the mbarrier operand as well as the response address. Take the response address from the first bracketed operand's uniform register. |
| TMA, bulk-copy, and other uniform-operand async ops | `reg_trace` | Absent from `mem_addr_trace`. Read the shared-memory operand from the uniform register named in the SASS operand, plus any immediate offset. |
| Descriptor-based MMA (`UTCHMMA`, `HGMMA`) | `reg_trace` | Shared-memory operands are descriptors, not addresses. Decode the descriptors or map the buffers from the PTX. |
| TMEM accesses (`LDTM`, `STTM`) | `reg_trace`, `mem_addr_trace` | `mem_addr_trace` records a value (for example `0x200000`) that is not a byte address. It encodes TMEM lane and column fields; decode and validate the encoding for your CUTracer version before matching accesses. |

## What a trace cannot show

- Whether a predicated instruction actually took effect in every lane that
  matters. Check `active_mask`, and read predication from the SASS.
- Phase completion and transaction totals, directly. A successful wait matched to the right phase under a validated protocol can establish completion; the trace alone does not.
- Synchronization that exists in the PTX but compiles to no SASS instruction.
  For example, ptxas can emit no `WARPSYNC` for a `bar.warp.sync` in
  converged code, so the trace shows no lane join even though the PTX has
  one. Check every `sync_lanes_unresolved`, `wait_lanes_unresolved`, and
  `same_warp_lanes_unresolved` row against the PTX before you classify it.
- Paths that the traced launch did not execute, and other configurations.
- Device-side timing. Instrumentation changes timing, so a trace that shows
  no conflicting interleaving is not evidence that none can occur.

## The bundled script

`scripts/trace_edges.py` turns one capture into a list of candidate edges
between access paths, a list of unresolved operations (descriptor reads, bulk
copies of unknown extent, undecodable operands and waits), and a list of
barriers excluded from matching. Its counts report which mechanisms it saw in
each warp's program order. They are
candidates for the manual check in the worksheet, not a verdict. See the
script's `--help` for the exact checks.
