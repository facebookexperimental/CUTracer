# NVIDIA sm_90 (Hopper)

Read [nvidia-common.md](nvidia-common.md) first. This file adds what is
specific to sm_90 and sm_90a. When a finding depends on an exact rule, cite
the PTX ISA section on `wgmma`. Items marked **(check)** need confirmation
against the PTX ISA.

## Extra async-proxy path

| Unit | Shared-memory access | SASS (observed) |
|---|---|---|
| `wgmma.mma_async` (sm_90a) | Reads operand matrices A and B from shared memory through descriptors, on the async proxy | `HGMMA` and related `*GMMA` forms |

`wgmma` completes through `wgmma.commit_group` and `wgmma.wait_group`. There is
no `wgmma` form that signals an mbarrier.

## What orders what

| Edge | Accepted ordering | Not sufficient |
|---|---|---|
| Generic writes to an operand buffer, then `wgmma` reads it | `fence.proxy.async` after the writes, then synchronization to the warpgroup that issues the `wgmma` | Synchronization without the fence; `wgmma.fence` (it does not order shared memory) |
| TMA writes an operand buffer, then `wgmma` reads it | The TMA completes on an mbarrier and the `wgmma` warpgroup waits on that phase successfully | n/a |
| `wgmma` reads an operand buffer, then a write (TMA or generic) reuses it | `wgmma.commit_group` and `wgmma.wait_group` with a count that covers that `wgmma`, then the release to the writer | Releasing the buffer right after issuing the `wgmma` |
| Register accesses to accumulators or register operands, then a `wgmma` that uses them | `wgmma.fence` before the first `wgmma` and after any change to those registers | n/a |
| A `wgmma`'s accumulator results, then ordinary instructions read or overwrite those registers | `wgmma.wait_group` covering that `wgmma` | A later `wgmma.fence` (it does not wait for completion) |

## SASS hints (observed mappings)

- `HGMMA` instructions take shared-memory operands as descriptors, so their
  addresses do not appear in a CUTracer `mem_addr_trace`. Map the descriptors
  from the PTX or the descriptor registers.
- `WARPGROUP.ARRIVE` and `WARPGROUP.DEPBAR` are the usual SASS forms of
  `wgmma.fence` and `wgmma.wait_group`. Do not rely on the mnemonic alone;
  lowering can differ across toolchains.
