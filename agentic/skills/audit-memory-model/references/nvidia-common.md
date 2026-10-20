# NVIDIA: rules common to sm_90 and later

NVIDIA calls the access paths to memory *proxies*. This file covers what all
targets from sm_90 on share. Read it together with the file for your target:

| Target | Also read |
|---|---|
| sm_90, sm_90a (Hopper) | [nvidia-sm90.md](nvidia-sm90.md) |
| sm_100, sm_100a (datacenter Blackwell) | [nvidia-sm100.md](nvidia-sm100.md) |
| Other targets | List their extra paths from the PTX ISA first. For example, sm_120 (consumer Blackwell) has no `tcgen05` and no Tensor Memory. |

These rules summarize the [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
memory model sections on proxies, `fence`, `mbarrier`, `cp.async`, and the bulk
asynchronous copies. When a finding depends on an exact rule, cite the PTX ISA
section and version. If you cannot open it, mark the claim inferred. What
CUTracer traces can and cannot show is in [cutracer-traces.md](cutracer-traces.md).

## Cited rules (PTX ISA 9.4)

These quotes were checked against PTX ISA 9.4. Section numbers change between
versions; cite the version you read.

| Section | Rule |
|---|---|
| 8.6 Proxies | "A proxy fence is required to synchronize memory operations across different proxies." |
| 8.9.1 Program Order | Program order "forms a total order over the operations performed by the thread, but does not relate operations from different threads." |
| 9.7.15.4 membar / fence | "bi-directional proxy fences take effect within a single thread and therefore do not have a .scope qualifier. Their cross-proxy synchronizing effect composes with other forms of synchronization according to the rules of the Memory Consistency Model." |
| 9.7.15.4 membar / fence | "A .release proxy fence can form a release sequence that synchronizes with an acquire sequence that contains a .acquire proxy fence." |
| 9.7.15.2 bar.warp.sync | "bar.warp.sync also guarantee memory ordering among threads participating in barrier." |
| 9.7.15.16.16 mbarrier.arrive | "If the .sem qualifier is absent, .release is assumed by default." `.relaxed` "does not provide any memory ordering semantics and visibility guarantees." |
| mbarrier.test_wait / try_wait | "If the .sem qualifier is absent, .acquire is assumed by default." |


## Access paths to shared memory

| Proxy | Shared-memory accesses that use it |
|---|---|
| Generic | `ld.shared`, `st.shared`, `atom.shared`, `red.shared`, `ld.shared::cluster` / `st.shared::cluster`, `ldmatrix` / `stmatrix` |
| Generic, asynchronous | Non-bulk `cp.async` (sm_80 form, SASS `LDGSTS`). It writes shared memory through the generic proxy but completes later; see the edge table. |
| Async | Bulk and tensor copies (`cp.async.bulk*`, `cp.reduce.async.bulk*`, TMA load, store, and reduce), which read or write shared memory |
| Async | Target-specific matrix units and schedulers; see the target file |
| Tensormap | Accesses to a tensor map (TMA descriptor) by TMA operations |

## How async operations complete

Async operations do not all complete the same way. Match each one to its own
mechanism:

| Operation | Completion mechanism |
|---|---|
| TMA and bulk copies into shared memory | An mbarrier transaction count: `expect_tx` adds the bytes the phase still expects, and the copy's `complete_tx` credits the completed bytes against them. The phase completes when its pending arrivals and expected bytes both reach zero. |
| Bulk stores and reduces that read shared memory | `cp.async.bulk.commit_group` and `cp.async.bulk.wait_group.read` (the source may be reused) or `wait_group` (the writes are complete) |
| Non-bulk `cp.async` | `cp.async.commit_group` with `cp.async.wait_group` / `wait_all`, or `cp.async.mbarrier.arrive` |
| Target-specific units | See the target file (for example, `tcgen05.commit` gives a plain mbarrier arrive, and `wgmma` uses commit and wait groups) |

### What a valid mbarrier wait requires

Finding a wait instruction is not enough. For an mbarrier to order an async
write before a read, check all of these:

- the async operation signals this mbarrier, in the phase the reader waits on;
- the phase's expected arrival count and expected transaction bytes
  (`expect_tx`) match what the producers actually deliver;
- the reader's wait succeeded on that phase (`try_wait` and `test_wait` can
  return false; only a successful check acquires), with the correct parity;
- the barrier is valid for its whole lifetime (initialized and published
  before first use, not invalidated while in use), and the reuse protocol lets
  each waiter observe its intended phase. Phases advance automatically when
  they complete; what matters is that a waiter cannot miss its phase or match
  a later one (for example, through wrong parity or a phase that a fast
  producer completes twice).

## What orders what

| Edge (same bytes) | Accepted ordering | Not sufficient |
|---|---|---|
| Async write, then generic read | The async op's completion is observed by a successful wait (see the checklist above), either by the reader or by a thread that then synchronizes with the reader | `bar.sync` alone, with no thread observing the completion; a wait on another barrier or phase |
| Async write, then async read | A completion that the reading side waits on before issuing | n/a |
| Generic write, then async read | `fence.proxy.async` after the writes, then synchronization that connects the writers to the thread that issues the async op | Synchronization without the fence |
| **Generic read, then async write** | Each reading thread executes `fence.proxy.async` after its reads, then releases the buffer (arrive or `bar.sync`) to the thread that issues the async write, which waits on that release. When one elected thread releases for its warp, the other reading threads must first synchronize with it (`bar.warp.sync`, or each thread arrives itself). Alternatively, a uni-directional pair: a `.release` proxy fence (`fence.proxy.async::generic.release...`) on the reading side and a matching `.acquire` proxy fence on the issuing side, joined by synchronization | Synchronization without the fence; a data dependence on the loaded value; a fence with no release that the writer waits on; a fence and release by one elected thread with no synchronization joining the other reading threads to it |
| Generic write, then async write (WAW) | `fence.proxy.async` after the generic writes, then synchronization to the issuing thread, as for generic write then async read | Synchronization without the fence |
| Async read, then any write (buffer reuse) | The async read has completed: for a bulk store, `wait_group.read`; for a matrix unit, its own completion (see the target file). Then synchronization to the writer. | Releasing the buffer right after issuing the async op |
| Non-bulk `cp.async` write, then read | `cp.async.wait_group` / `wait_all` (or the `cp.async.mbarrier.arrive` phase) in the issuing threads, then synchronization to the readers | `bar.sync` alone. Both accesses use the generic proxy, but the copy completes later. |
| Generic access, then generic access | Synchronization that orders the two threads (`bar.sync`, mbarrier release and acquire) | n/a |

In every row, "synchronization" means a direct or transitive chain of
release and acquire between the participants (for example, the waiting thread
publishes completion to the readers through a barrier). When both accesses
belong to the same thread, no synchronization between threads is needed, but
the completion and proxy rules still apply.

The generic-read-then-async-write row is the easiest to miss. It arises when
threads read a buffer and then release it to an async writer, for example
when a pipeline stage or a staging area is reused for the next async
operation. The release is correct for execution order and wrong for proxy
order. Without the fence, the async write may land before the generic reads
complete.

## Global memory across CTAs

Ordering between CTAs through global memory needs release and acquire at the
right scope (`.cta`, `.cluster`, `.gpu`, `.sys`), either through scoped
`ld.acquire` / `st.release` / `atom` or a `fence` of that scope. A fence or
`bar.sync` scoped to one CTA does not order accesses for another CTA.

- A TMA store writes global memory through the async proxy. Wait for it to
  complete (`cp.async.bulk.wait_group`) before another agent relies on the
  data.
- Non-coherent loads (`ld.global.nc`, `__ldg`) may return stale data for
  memory that another thread writes during the kernel. Do not use them for
  data that changes during the kernel.
- A tensor map written in global memory during the kernel (or by another
  kernel without a full synchronization) must be published to the tensormap
  proxy before TMA uses it, with the `fence.proxy.tensormap::generic` release
  and acquire forms **(check the exact forms and scopes)**.

## Barrier initialization and cluster scope

| Situation | Requirement |
|---|---|
| `mbarrier.init` followed by use from other threads, other CTAs in the cluster, or async operations | Publish the initialization before first use, then synchronize with the users. For remote (cluster) users, `fence.mbarrier_init.release.cluster` followed by a cluster barrier is the documented pattern. Whether a given async operation also needs `fence.proxy.async` depends on that instruction's contract; do not treat the two fences as interchangeable. **(check per instruction)** |
| Producer and consumer in different CTAs of a cluster | Write down the producer's CTA, the consumer's CTA, the state space (`.shared::cta` or `.shared::cluster`), and the synchronization scope. A fence's state-space qualifier is not its scope. Cluster communication often uses a local proxy fence in each CTA followed by a cluster-scope synchronization; it does not always need a cluster-scope fence. |

## How fences appear in SASS

Observed mappings (verify on your toolchain):

- The shared-memory proxy fence is `FENCE.VIEW.ASYNC.S`. On sm_100a,
  `fence.proxy.async.shared::cta` compiled to `MEMBAR.ALL.CTA` followed by
  `FENCE.VIEW.ASYNC.S`.
- Other `FENCE.VIEW.ASYNC.*` forms order other memories (see the target file)
  and do not count for a shared-memory edge.
- Some fences compile to no additional instruction. That shows only that the
  input change produced no new instruction; it does not show that the lowering
  provides the ordering, and it says nothing about other targets.

mbarrier operations, observed with ptxas 12.8 and 13.0 for sm_100a:

| PTX | SASS |
|---|---|
| `mbarrier.arrive` (release, CTA scope) and `mbarrier.arrive.relaxed.cta` | both `SYNCS.ARRIVE.TRANS64.A1T0` |
| `mbarrier.try_wait.parity` with `.acquire` or `.relaxed` | both `SYNCS.PHASECHK.TRANS64.TRYWAIT` |
| `mbarrier.arrive` with a count | `SYNCS.ARRIVE.TRANS64.ART0` |
| `mbarrier.arrive.expect_tx` | `SYNCS.ARRIVE.TRANS64` (no suffix) |
| `mbarrier.expect_tx.relaxed` / `mbarrier.complete_tx.relaxed` | `SYNCS.ARRIVE.TRANS64.RED.A0TR` / `.RED.A0TX`; no arrival |
| `mbarrier.arrive.release.cluster` | `MEMBAR.ALL.CTA; MEMBAR.ALL.GPU` before the arrive |
| Arrive on a `shared::cluster` address (after `mapa`) | `SYNCS.ARRIVE.TRANS64.RED.A1T0` |

Relaxed and release arrives produce the same SASS, and so do relaxed and
acquire waits. Neither the SASS nor a trace shows whether an arrive is a
release or a successful wait is an acquire. Read the PTX (or the source) for
`.relaxed` before you count either one, and if it is relaxed, look for the
fence that supplies the ordering.

## Arguments that do not establish ordering

- "A `bar.sync` sits between them." That orders execution and generic-proxy
  accesses only.
- "The arrive is predicated on the loaded value, so the load has completed."
  That is a data dependence in the generic proxy. It does not order the access
  against the async proxy.
- "There is a wait on an mbarrier first." That covers async write then read
  only if the checklist under "What a valid mbarrier wait requires" holds.
- "The fence is there." A fence orders only what the participants then
  synchronize on. Check that the other side waits on the release that follows
  the fence.
- "It never fails in testing." Runtime results cannot show that an ordering
  exists.
