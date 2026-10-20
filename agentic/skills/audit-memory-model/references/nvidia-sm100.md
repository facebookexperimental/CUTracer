# NVIDIA sm_100 (datacenter Blackwell)

Read [nvidia-common.md](nvidia-common.md) first. This file adds what is
specific to sm_100 and sm_100a: `tcgen05`, Tensor Memory (TMEM), and cluster
launch control. When a finding depends on an exact rule, cite the
[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
sections on `tcgen05` and `clusterlaunchcontrol`. Items marked **(check)**
need confirmation against the PTX ISA.

## Extra async-proxy paths to shared memory

| Unit | Shared-memory access | Completion | SASS (observed) |
|---|---|---|---|
| `tcgen05.mma` | Reads operand matrices from shared memory through descriptors | `tcgen05.commit` to an mbarrier: a plain arrive when the committed operations complete, not a byte-counted `complete_tx` | `UTCHMMA`, `UTCQMMA`, and related forms |
| `tcgen05.cp` | Reads shared memory, writes TMEM | `tcgen05.commit`, as above | `UTCCP` **(check)** |
| `clusterlaunchcontrol.try_cancel` | Writes a 16-byte response to shared memory | `complete_tx` on an mbarrier | `UGETNEXTWORKID` |

The shared-memory rules in the common file apply to all three, in every
direction.

### Cluster launch control (PTX ISA 9.4, section 9.7.15.18)

- With `.multicast::cluster::all`, "the response is asynchronously written
  using weak async-proxy writes to the corresponding local shared memory addr
  of each CTA in the requesting cluster." The section does not name the proxy
  for the form without `.multicast`; treat it as an async write and mark the
  proxy inferred.
- Completion: "No other synchronization mechanisms described in Memory
  Consistency Model can be used to guarantee the completion of the
  asynchronous copy operations." Only a successful wait on the mbarrier phase
  that the `complete_tx` completes shows the response is written.
- The instruction "accesses its mbarrier operand using generic-proxy."
- Reusing the response buffer for the next request is the common file's
  generic-read-then-async-write edge.

## Tensor Memory

TMEM is a separate memory with its own access paths. Threads move data
between registers and TMEM with `tcgen05.ld` and `tcgen05.st`. The MMA unit
reads and writes TMEM with `tcgen05.mma`, and `tcgen05.cp` copies shared
memory into TMEM. All of these are asynchronous with respect to the issuing
thread.

| Access | Instruction | SASS (observed) | Completion, and what it covers |
|---|---|---|---|
| Thread read | `tcgen05.ld` | `LDTM` | `tcgen05.wait::ld` waits for the issuing thread's own prior `tcgen05.ld` operations only |
| Thread write | `tcgen05.st` | `STTM` | `tcgen05.wait::st` waits for the issuing thread's own prior `tcgen05.st` operations only |
| MMA read or write | `tcgen05.mma` | `UTCHMMA` and related forms | `tcgen05.commit` arrives on an mbarrier when the committed MMAs and copies complete. It does not cover `tcgen05.ld` or `tcgen05.st`. |
| Copy from shared memory | `tcgen05.cp` | `UTCCP` **(check)** | `tcgen05.commit`, as above |

Neither `wait::ld`/`wait::st` nor `commit` waits for another warp's
operations. A handoff between warps always needs a synchronization that
connects the two participants.

A warp can only access the TMEM lanes that belong to its position in the
warpgroup (warp `w` reaches lanes `32 * (w % 4)` to `32 * (w % 4) + 31`), so
warps `w` and `w + 4` share a lane group. Whether two accesses overlap depends
on the lanes, the columns, the instruction shape, and the data packing, not on
logical tensor indices or register numbers. Two warps that touch the same
columns in the same lanes, for example through aliased TMEM allocations,
conflict even though each warp's code looks correct on its own.

### What orders what in TMEM

| Edge (overlapping TMEM lanes and columns) | Accepted ordering | Not sufficient |
|---|---|---|
| MMA writes an accumulator, then threads read it with `tcgen05.ld` | `tcgen05.commit` to an mbarrier; the reader waits on it successfully, then `tcgen05.fence::after_thread_sync`, then `tcgen05.ld` | The wait without the fence |
| Threads read with `tcgen05.ld`, then the MMA (or a `tcgen05.cp`) overwrites the same TMEM | Reader: `tcgen05.wait::ld`, `tcgen05.fence::before_thread_sync`, then the release (arrive or barrier). MMA side: a successful wait on that release, then `tcgen05.fence::after_thread_sync` | Releasing right after issuing `tcgen05.ld` |
| Threads write with `tcgen05.st`, then the MMA or other threads read it | Writer: `tcgen05.wait::st`, `tcgen05.fence::before_thread_sync`, then the release. Reader: a successful wait, then `tcgen05.fence::after_thread_sync` | Releasing right after issuing `tcgen05.st` |
| **Warp A reads with `tcgen05.ld`, then warp B writes the same TMEM with `tcgen05.st`** (aliased buffers) | A: `tcgen05.wait::ld`, `tcgen05.fence::before_thread_sync`, then a release. B: a successful wait on that release, then `tcgen05.fence::after_thread_sync`, then `tcgen05.st` | No synchronization between the two warps; program order within each warp |
| One warp: `tcgen05.st` then `tcgen05.ld` or `tcgen05.st`; `tcgen05.ld` then `tcgen05.st` | Pipelined: no wait needed (PTX ISA 9.4, 9.7.18.6.2). Other same-thread pairs need the completion mechanism (`wait::ld`/`wait::st`, or `commit`) | Treating a register dependence on a `tcgen05.ld` result as memory ordering (9.7.18.6.4.5) |

The bold row is the TMEM counterpart of the shared-memory read-then-write
edge. It is easy to miss when two logical buffers alias the same TMEM columns.

Cited from PTX ISA 9.4, section 9.7.18.6 (tcgen05 memory consistency):

- "Instructions issued by different warps are always non-pipelined." A
  handoff between warps always needs the canonical pattern (9.7.18.6.4.4):
  completion, `tcgen05.fence::before_thread_sync`, synchronization, then
  `tcgen05.fence::after_thread_sync` on the other side.
- `tcgen05.commit` implicitly performs `tcgen05.fence::before_thread_sync`.
- The canonical patterns use `mbarrier.arrive.relaxed` and
  `mbarrier.try_wait.relaxed`: the `tcgen05` fences "compose with execution
  ordering instructions" to order the `tcgen05` operations. A `.relaxed`
  arrive or wait is therefore not a defect for a `tcgen05` handoff that has
  both fences. It is still a defect for ordinary shared or global data handed
  off through the same barrier, which needs release and acquire.

## Barrier initialization

Barriers that `tcgen05.commit` or `try_cancel` signal follow the
initialization rules in the common file.

## SASS hints (observed mappings)

- `FENCE.VIEW.ASYNC.T` orders TMEM. One audit run observed it as the lowering
  of `tcgen05.wait::st`. It is not a shared-memory proxy fence. Do not assume
  that every `.T` fence implements a `tcgen05` thread-sync fence.
- `UTCHMMA` reads shared memory through descriptors, so its shared-memory
  addresses do not appear in a CUTracer `mem_addr_trace`.
- How `LDTM` and `STTM` appear in traces is described in
  [cutracer-traces.md](cutracer-traces.md).
