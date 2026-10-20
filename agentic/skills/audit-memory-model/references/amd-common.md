# AMD: rules common to CDNA3 and CDNA4

AMD has no proxies. Memory instructions complete through per-wave counters,
and a wave must wait on the right counter before another wave may rely on the
result. This file covers what gfx942 and gfx950 share. Read it together with
the file for your target:

| Target | Also read |
|---|---|
| gfx942 (CDNA3, MI300) | [amd-gfx942.md](amd-gfx942.md) |
| gfx950 (CDNA4, MI350) | [amd-gfx950.md](amd-gfx950.md) |
| Other targets | List their access paths and counters from the ISA guide first. Newer targets add asynchronous tensor movers with counters of their own (for example a tensor counter waited on with `s_wait_tensorcnt`). |

**Provisional:** no audit of a real AMD kernel has used these rules yet. The
rules below were checked against the documents cited in the next section.

These rules summarize the AMD Instinct CDNA3 and CDNA4 ISA guides and the
AMDGPU memory-model section of the LLVM
[`AMDGPUUsage`](https://github.com/llvm/llvm-project/blob/main/llvm/docs/AMDGPUUsage.rst)
document, which lists the exact code sequence for each scope and operation.
When a finding depends on an exact rule, cite the section. If you cannot open
the document, mark the claim inferred.

## Cited rules

From the AMD Instinct MI300 (CDNA3) ISA guide; the CDNA4 guide has the same
text:

| Section | Rule |
|---|---|
| 4.4 Data dependency resolution | "Instructions of a given type return in order, but instructions of different types can complete out-of-order." |
| 4.4, VM_CNT | "Memory reads and writes return in the order they were issued, including mixing reads and writes." |
| 4.4, LGKM_CNT | "Incremented by 1 for every LDS or GWS instruction issued, as well as by Dword-count for scalar-memory reads (1 for 1-dword loads, 2 for 2-dword or larger loads)." |
| 8.3 Dependency checking (scalar memory) | Scalar reads "can return out-of-order"; "the only sensible way to use this counter is to implement S_WAITCNT 0". |
| 10.2 FLAT | FLAT instructions "increment both VM_CNT and LGKM_CNT" and "can complete out of order with each other"; "the only sensible S_WAITCNT value to use after FLAT instructions is zero". |
| S_BARRIER | "Barrier instructions do not wait for any counters to go to zero before issuing. If the barrier is being used to protect an outstanding memory operation use the appropriate S_WAITCNT instruction before the barrier." |

From LLVM `AMDGPUUsage`, "Memory Model GFX942":

- "All LDS operations of a CU are performed as wavefront wide operations in a
  global order and involve no caching. Completion is reported to a wavefront
  in execution order."
- "A `s_waitcnt lgkmcnt(0)` is required to ensure synchronization between LDS
  operations and vector memory operations between wavefronts of a work-group,
  but not between operations performed by the same wavefront."

## Wait counters

| Counter | Tracks |
|---|---|
| `vmcnt` | Vector memory operations, including direct global-to-LDS loads. One count per instruction, whatever its width. |
| `lgkmcnt` | LDS, GDS, scalar memory, and messages. LDS and GDS count one per instruction; scalar loads count by dwords (1 or 2). FLAT instructions count on both counters. |

## Access paths to LDS

| Path | Instructions | Completion counter |
|---|---|---|
| Wave LDS access | `ds_read*`, `ds_write*`, `ds_*` atomics | `lgkmcnt` |
| Direct global-to-LDS load | `buffer_load_* ... lds`, `global_load_lds_*` | `vmcnt` |

MFMA instructions read and write VGPRs and AGPRs, not LDS. On these targets no
matrix unit reads LDS on its own path.

Scalar memory loads (`s_load*`) also count on `lgkmcnt`, and they can return
out of order relative to LDS operations. When both are outstanding, only
`lgkmcnt(0)` guarantees that a given LDS operation has completed.

## What orders what

| Edge | Accepted ordering | Not sufficient |
|---|---|---|
| LDS write by wave A, then LDS read by wave B (same workgroup) | A: `s_waitcnt lgkmcnt(0)`, then `s_barrier`. B reads after the barrier. | `s_barrier` alone. It does not wait for outstanding memory operations. |
| LDS read by wave A, then LDS write by wave B (buffer reuse) | A: `s_waitcnt lgkmcnt(0)` (the reads have returned), then `s_barrier`. B writes after the barrier. | `s_barrier` alone |
| **Direct-to-LDS load, then LDS read by any wave** | The issuing wave: `s_waitcnt vmcnt(N)` that covers that load, then `s_barrier`. Readers read after the barrier. Buffer and global vector-memory reads and writes return in issue order, so N counts the vector-memory instructions issued after the load. FLAT instructions complete out of order; if any were issued after the load, require `vmcnt(0)`. | `s_waitcnt lgkmcnt(0)` (wrong counter); `s_barrier` alone |
| **LDS read, then a direct-to-LDS load into the same bytes** | Readers: `s_waitcnt lgkmcnt(0)`, then `s_barrier`. The load is issued after the barrier. | `s_barrier` alone; a data dependence on the read value without the wait |
| Global write by one workgroup, then global read by another (agent scope) | Release at agent scope before the flag store or atomic, and acquire after the flag load or atomic. The publishing wave must first be ordered after every wave whose writes it publishes (for example, those waves wait on their own `vmcnt` and then synchronize with it). The cache instructions that implement release and acquire are target-specific; see the target file. Data read with scalar loads also needs scalar-cache handling and an `lgkmcnt` wait. | `s_barrier` (workgroup only); a workgroup-scope fence; a flag without release and acquire |
| Global accesses across XCDs | Agent-scope release and acquire as above. Each XCD has its own L2, so workgroup- or XCD-local ordering is not enough. | Ordering that only covers one XCD's L2 |

The two bold rows are the easiest to miss. A direct-to-LDS load writes LDS
through the vector-memory path, so `lgkmcnt` does not track it. Code that
waits on `lgkmcnt` before a barrier, as ordinary LDS code does, is correct for
`ds_write` and wrong for a direct-to-LDS load.

## Within one wave

LDS operations from one wave complete in execution order, so a wave's own
`ds_write` followed by its `ds_read` of the same address is ordered without a
wait (LLVM `AMDGPUUsage`, Memory Model GFX942). Results still need the counter
wait before the wave uses the loaded value. The compiler inserts that wait
when the value is used.

## How waits appear in the ISA

- A workgroup barrier in source (`__syncthreads()`, `tl.debug_barrier()`, or
  a compiler-inserted barrier) usually lowers to `s_waitcnt` on the needed
  counters followed by `s_barrier`. Check that the counters cover every
  outstanding operation that the barrier is meant to order, including
  direct-to-LDS loads.
- Scoped fences and atomics in LLVM IR (`fence syncscope("agent") release`,
  `syncscope("workgroup")`) lower to `s_waitcnt` plus cache instructions as
  listed in the LLVM memory-model table.
- For triage, add the missing fence or `llvm.amdgcn.s.waitcnt` to a copy of
  the LLVM IR, recompile for the same `--mcpu`, and diff the AMDGCN.

## Artifacts

- Triton's compile cache holds `.ttgir`, `.llir`, `.amdgcn` (assembly), and
  `.hsaco` (code object).
- Disassemble a code object with `llvm-objdump -d --mcpu=<target>`.
- CUTracer's capture does not run on AMD. For executed order, use ROCgdb or
  ROCprofiler.

## Arguments that do not establish ordering

- "An `s_barrier` sits between them." The barrier does not wait for
  outstanding memory operations.
- "We wait on `lgkmcnt(0)` before the barrier." That does not cover a
  direct-to-LDS load, which counts on `vmcnt`.
- "The waves run in lockstep" or "the workgroup is small." Neither is
  guaranteed.
- "The second access depends on the first one's value." The compiler waits
  for that value in the reading wave only. Other waves are not ordered by it.
- "It never fails in testing." Runtime results cannot show that an ordering
  exists.
