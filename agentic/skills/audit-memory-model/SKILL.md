---
name: audit-memory-model
description: Use when a GPU kernel's correctness is in question but no test fails, for example when the reproducer passes and someone asks whether the kernel is correct; timing, stress, random-delay, or sanitizer runs found nothing; a debugging workflow is about to conclude "no defect"; a synchronization change to warp-specialized or pipelined code needs review before it lands; or a missing fence, wait, or barrier is suspected. Applies to NVIDIA Hopper and Blackwell kernels that use TMA, wgmma, tcgen05, cluster launch control, or mbarriers, and to AMD CDNA3 and CDNA4 kernels that use LDS, direct-to-LDS loads, s_waitcnt, or s_barrier.
license: MIT
compatibility: Requires the kernel's generated intermediate code and machine ISA for the exact configuration that runs (for Triton, the compile cache holds both). CUTracer traces are optional; they list candidate edges and the mechanisms each warp executed.
---

# Audit a GPU kernel against the memory model

## When to use this

Use this skill when correctness has to be established from the rules, not from
a failure:

- the reproducer passes, but someone asks whether the kernel is correct;
- random-delay or stress searches found nothing;
- a synchronization change needs review before it lands;
- someone suspects a missing fence, wait, or barrier.

A passing run does not show that a kernel is correct. Some defects in this
class never produce wrong output on a given GPU, driver, or compiler, and then
fail on another one. Do not end an audit with "no defect" because runtime tests
pass. Runtime results bound how often a failure occurs. They cannot show that
a required ordering exists.

If a failure exists (hang, wrong output, fault), debug that first with the
matching CUTracer debugging workflow. Come back to this audit when the failure
is explained or when runtime methods stop finding anything.

## Core idea

A memory location can be reached through more than one access path. Threads
use ordinary loads and stores. Copy engines, matrix units, and other hardware
units read or write the same memory through paths of their own. Each path has
its own completion and ordering rules.

> Thread synchronization alone does not supply the completion or the ordering
> between access paths that these operations need. This covers workgroup
> barriers, arrive and wait, data dependences, control dependences, and
> predicates, unless the vendor model names that mechanism for the edge (for
> example, a successful wait on the mbarrier phase that an async copy
> completes). Each edge between two accesses to the same bytes needs the
> mechanism the vendor model names for that pair: a fence, a completion signal,
> or a counter wait, and a synchronization that connects the two participants.
> Check every direction (read then write, write then read, write then write),
> including reuse of a buffer in the next loop iteration.

The common wrong arguments are the same on every vendor: "a barrier sits
between them", "the second access depends on the first one's value", and "it
never fails in testing". None of them establishes the required ordering.

## Pick the hardware reference

Identify the target first, for example from `nvidia-smi`, `rocminfo`, or the
compiled binary's target string. Then read its reference before you start:

| Target | Read |
|---|---|
| NVIDIA sm_90 (Hopper) | [references/nvidia-common.md](references/nvidia-common.md), then [references/nvidia-sm90.md](references/nvidia-sm90.md) |
| NVIDIA sm_100 (datacenter Blackwell) | [references/nvidia-common.md](references/nvidia-common.md), then [references/nvidia-sm100.md](references/nvidia-sm100.md) |
| AMD gfx942 (CDNA3, MI300) | [references/amd-common.md](references/amd-common.md), then [references/amd-gfx942.md](references/amd-gfx942.md) |
| AMD gfx950 (CDNA4, MI350) | [references/amd-common.md](references/amd-common.md), then [references/amd-gfx950.md](references/amd-gfx950.md) |

The AMD references have not yet been checked against an audit of a real AMD
kernel. Treat AMD findings as provisional, and mark every AMD rule you rely on
as inferred unless you confirmed it in the ISA guide or the LLVM memory-model
tables.

For any other target, start from the vendor's common file, then list the
target's extra access paths and ordering mechanisms from its ISA and
memory-model documents before you audit. Treat that list as inferred until
someone checks it against the documents. Add a target file when the list is
checked.

## Procedure

Work through [references/audit-worksheet.md](references/audit-worksheet.md).
On NVIDIA, a CUTracer trace lists candidate edges and the accesses behind them.
On AMD, and for anything the trace cannot resolve, read the compiled code.

1. **Collect the artifacts.** Get the intermediate code (PTX or LLVM IR) and
   the machine ISA (SASS or AMDGCN) for the exact configuration that runs.
   Compile with a fresh cache so the binary matches the source you audit.
2. **Capture a trace (NVIDIA).** Run one launch with enough work that each
   CTA reuses its buffers at least once (for a persistent kernel, more tiles
   than SMs):

   ```bash
   cutracer trace -i reg_trace,mem_addr_trace -k <kernel> --output-dir <dir> -- <command>
   ```

   `reg_trace` records every executed instruction, including fences,
   barriers, and arrives. `mem_addr_trace` records per-lane shared-memory
   addresses. The shared-memory operands of TMA, CLC, and bulk-copy
   instructions come from their uniform registers in `reg_trace`. If the
   script in step 3 warns that the trace has no write-after-read candidate,
   the launch was too small: trace a larger one.
3. **List the cross-path edges.** Run the bundled script on the trace:

   ```bash
   python3 scripts/trace_edges.py <dir>/<kernel>.ndjson
   ```

   It lists candidate edges: pairs of accesses to overlapping bytes through
   different paths, per CTA. For each, it reports which mechanisms it saw in
   each warp's own instruction stream (a fence, a release, a successful wait)
   and whether the synchronization ordinals connect the two warps. These are
   candidates and observations, not verdicts. Every candidate, including one
   with all mechanisms seen, needs the manual check in step 4.
4. **Check each edge against the rules.** Map the PCs to SASS and source.
   For every candidate and every unresolved operation, confirm it with the rules in the
   hardware reference. Then cover what the script cannot see:
   - async operations that read shared memory through descriptors (MMA
     operands), which the script lists separately: map their buffers from the
     descriptors or the PTX;
   - the async-write-then-generic-read rows. The script matches the reader's
     successful wait to the phase the async operation completes only when it
     can decode that barrier from the instruction's operands; otherwise it
     reports `barrier_unknown`. Either way, confirm it is the barrier the async
     operation completes;
   - rows the script reports as lanes unresolved: the lane join may exist in
     the PTX (for example `bar.warp.sync`) and compile to no SASS instruction.
     The contract is the PTX, so classify from the PTX;
   - code paths the trace did not execute, other configurations, and cluster
     shapes. Read them in the compiled code.

   On AMD, build the buffer map and edge list from the AMDGCN by hand, as
   described in the worksheet. List every access to a reused buffer between
   two synchronization points; the earliest write after a read, not the first
   one you find, decides the edge.
5. **Classify each gap, then test the fix.** Classify each finding from the
   rule and the ordering chain you found in step 4, not from the ISA. Then add
   the missing instruction to a copy of the intermediate code, recompile with
   the same compiler and flags, and diff the machine ISA. Report the diff as
   evidence next to the classification: an added instruction shows that the
   shipped binary lacked it, and a change that adds nothing shows only that the
   modified input produced no additional instruction (also compare scheduling
   and control fields). Neither result decides the classification: an added
   instruction can be redundant with another valid mechanism, and a change
   that adds nothing does not show that the lowering provides the ordering or
   that other targets tolerate the gap. On NVIDIA, re-run steps 2 and 3
   on the fixed build: the edge must now show the required mechanism, and no
   other edge may get worse. To show that the gap can change results, delay
   one participant at the gap and run the test with and without the fix (see
   the worksheet); first check that the test can detect a corrupted buffer.
6. **Report.** For each finding, give the buffer, the two instructions with
   their PCs and paths, the direction, the missing mechanism, the fix
   position, the ISA evidence, and the trace counts before and after the fix.
   Keep three things apart: the contract violation, what the generated machine
   code does, and any runtime consequence.

## Evidence rules

- Label each claim observed (read from an artifact), inferred (reasoned from
  artifacts or documentation), or causal (shown by an intervention).
- A requirement taken from documentation that you cannot open is inferred.
  Say which document it comes from.
- A finding needs no reproduced failure. Its evidence is the rule, the
  instruction pair, and the ISA triage.
- Rank findings by the contract gap and the behavior it can affect (for
  example, a buffer whose overwrite corrupts results ranks above a missing
  initialization fence that only matters for remote users). Report the ISA
  diff separately; it is not a ranking.
- Name the exact binary you audited (cubin or code object hash). A fresh
  compile helps match source to binary, but it must not silently replace the
  binary under investigation.
- State what the audit did not cover: targets, configurations, cluster or
  workgroup shapes, and code paths you did not compile or read.

## CUTracer on each vendor

On NVIDIA, CUTracer is the primary evidence source for steps 2 and 3. Other
commands that help with step 4:

```bash
# all executions of one instruction, by PC
cutracer query <dir>/<kernel>.ndjson --filter "pc=<pc>" --all-lines

# SASS of the traced cubin
cutracer sass <dir>/*.cubin
```

The trace's first line maps each `opcode_id` to its SASS, so every record can
be tied back to an instruction.

On AMD, CUTracer's capture does not run, because it is built on NVBit. Read
the AMDGCN from the compiled code object instead, and capture executed order
with the ROCm tools (ROCgdb, ROCprofiler) when you need it.
