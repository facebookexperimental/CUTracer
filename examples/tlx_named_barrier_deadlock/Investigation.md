# CUTracer investigates a TLX named-barrier kernel hang on B200: divergent `WARPSYNC.ALL` blocks execution before `BAR.SYNC`

**Status:** Confirmed · **Tags:** TLX, named barrier, deadlock, `WARPSYNC.ALL`, LLVM jump threading, convergent intrinsic · **Reproducer:** [repro.py](repro.py)

## TL;DR

- **Symptom:** `named_barrier_divergence` hangs when lane 0 performs a store before and after `tlx.named_barrier_wait(14, 32)`.
- **Conclusion:** LLVM jump threading duplicated a non-convergent inline-assembly barrier across divergent control flow. The generated SASS splits warp 6 into 1-lane and 31-lane paths. Each path stops at a separate predicated `WARPSYNC.ALL` before it reaches its copied `BAR.SYNC`.
- **Outcome:** Triton commit [`a6ffe3ce`](https://github.com/facebookexperimental/triton/commit/a6ffe3ce5ee1ade3596332c85ce3a1e6ea377630) lowers named barriers to convergent NVVM intrinsics. An exact parent-versus-fix test changed the result from a 15-second hang to `PASS: kernel completed with output [5, 99]`.
- **CUTracer contribution:** A register trace recorded active masks `0x1` and `0xfffffffe` at the two terminal `WARPSYNC.ALL` instructions. It also showed zero dynamic records at the later named-barrier PCs. This evidence moved the investigation from a general named-barrier hang to divergent generated synchronization before the barrier.

## Investigation timeline

**Total investigation time:** 14 minutes. The blind diagnosis took 4 minutes. The exact fix validation completed at T+14 minutes.

**Coding agent:** Codex, GPT-5. The agent received only the symptom, reproducer path, and runnable environment. It did not receive the known cause or fix.

| Elapsed | Action and evidence | Decision or next action |
|---|---|---|
| T+0 min | The baseline exceeded 15 seconds and exited 124. | Trace the launched kernel. |
| T+1 min | CUTracer identified `named_barrier_divergence`: grid `(1,1,1)`, block `(256,1,1)`, and 8 warps. A sync-only capture produced zero dynamic records. | Treat the zero-record capture as inconclusive. Capture all opcodes. |
| T+2 min | The all-opcode capture recorded 172 instructions. Warp 6 stopped at two predicated `WARPSYNC.ALL` PCs. | Capture active masks with `reg_trace`. |
| **T+4 min — runtime cause localized** | CUTracer recorded mask `0x1` at PC `0x220` and mask `0xfffffffe` at PC `0x280`. It recorded no later warp-6 instruction and no dynamic named-barrier instruction. | Bound the blind conclusion to generated synchronization under divergence. |
| **T+14 min — defect boundary confirmed** | The public fix identified non-convergent inline assembly and LLVM jump threading. An exact source-revision A/B changed hang to pass. | Confirm the compiler lowering defect and the fix. |

## Symptom and reproducer

The reproducer launches one CTA with 256 threads. One TLX asynchronous task uses one warp. Lane 0 stores a value, all 32 lanes wait on named barrier 14, and lane 0 stores another value.

Build the affected public Triton revision:

```bash
git clone https://github.com/facebookexperimental/triton.git
cd triton
git checkout d077424a67a74634e64ce35f644b2ac36eccd229
python3 -m venv .venv
.venv/bin/pip install -r python/requirements.txt torch
.venv/bin/pip install -e . --no-build-isolation
```

From the CUTracer repository root, run:

```bash
/path/to/triton/.venv/bin/python \
  examples/tlx_named_barrier_deadlock/repro.py --timeout 15
```

The affected revision prints:

```text
HANG REPRODUCED: kernel exceeded 15 seconds
```

The [README](README.md) also gives the current `fbtriton` control command.

## Evidence summary

| Claim | State | Producing tool | Evidence | Boundary |
|---|---|---|---|---|
| The kernel hangs. | Observed | Reproducer | The child remained alive for 15 seconds and exited 124 after termination. | One run on one B200. |
| The target kernel launches and begins device execution. | Observed | CUTracer launch logger and opcode trace | CUTracer identified launch ID 1 and recorded 172 dynamic instructions from 8 warps. | The trace ended by timeout. |
| Warp 6 stops at complementary full-warp synchronization instructions. | Observed | CUTracer `reg_trace` | PC `0x220`: `@!P0 WARPSYNC.ALL`, mask `0x1`. PC `0x280`: `@P0 WARPSYNC.ALL`, mask `0xfffffffe`. No later warp-6 record exists. | This is the instrumented execution. |
| The named-barrier instructions are not the immediate blocking instructions. | Observed | CUTracer query | The decisive trace contains zero records at `BAR.SYNC` PCs `0x270` and `0x2c0`. | Zero is meaningful for warp 6 because the same one-record-flush trace contains its preceding PCs. |
| The generated synchronization around the divergent region is the runtime defect boundary. | Inferred during the blind phase | CUTracer trace plus SASS | The 1-lane and 31-lane masks partition one warp at separate `WARPSYNC.ALL` instructions. | CUTracer did not identify the compiler pass that created the control flow. |
| LLVM jump threading duplicated the barrier because inline assembly lacked `convergent`. | Causal after reveal | Triton source diff and exact A/B | Commit `a6ffe3ce` replaces inline assembly with convergent NVVM intrinsics. Its parent hangs; the fix commit passes. | Same LLVM hash, script, GPU, environment, and fresh cache for both variants. |

The decisive CUTracer chain was:

```text
PC 0x220, active_mask 0x1, @!P0 WARPSYNC.ALL
+ PC 0x280, active_mask 0xfffffffe, @P0 WARPSYNC.ALL
+ zero dynamic records at BAR.SYNC PCs 0x270 and 0x2c0
-> warp 6 splits into complementary lane subsets and stops before the named barriers
-> inspect generated control flow and compiler lowering instead of the barrier runtime state
```

CUTracer's deadlock analysis classified 7 of 8 warps as barrier-blocked, 1 as unknown, and 0 as progressing or exited. It also marked the trace as truncated. Therefore, the aggregate detector result alone was not proof of the mechanism. The per-PC records and active masks supplied the decisive evidence.

## How CUTracer helped

CUTracer first isolated the launched kernel and confirmed device execution. The one-record-flush trace then preserved the terminal instruction sequence before timeout.

The active masks made the divergence concrete. Lane 0 reached one `WARPSYNC.ALL`. Lanes 1 through 31 reached the other. The absence of later records placed the immediate stall before both named-barrier instructions.

This changed the next action. The investigation moved from barrier-state debugging to generated control-flow and lowering inspection.

## Blind investigation and human report comparison

### Blind phase

The agent considered three main boundaries: launch or compilation, the named barrier, and generated synchronization around divergent control flow. Launch logging rejected the first boundary. The register trace narrowed the failure to the third boundary.

The blind conclusion did not name LLVM jump threading. It also did not claim that the named barrier itself blocked. The trace did not reach either copied `BAR.SYNC` instruction.

### Post-investigation validation

The public [Triton fix report](https://github.com/facebookexperimental/triton/commit/a6ffe3ce5ee1ade3596332c85ce3a1e6ea377630) identified the compiler cause. Named barriers were lowered through `PTXBuilder` to `llvm.inline_asm`. The operation did not have the `convergent` attribute. LLVM jump threading could therefore duplicate the barrier into both sides of divergent control flow.

The blind result aligned with the human report at the executable boundary. Both found split control flow and separate synchronization paths. CUTracer added dynamic lane masks and showed that this build stopped at `WARPSYNC.ALL` before either copied barrier executed.

The blind agent missed two facts that required compiler evidence:

- LLVM jump threading was the pass that duplicated the barrier.
- A convergent NVVM intrinsic was the correct lowering and fix.

These facts were not available in the runtime trace. The later source diff supplied them.

## Resolution and validation

The fix replaces PTX inline assembly with these NVVM intrinsics:

- `llvm.nvvm.barrier.cta.sync.aligned.count`
- `llvm.nvvm.barrier.cta.arrive.aligned.count`

LLVM defines these intrinsics as convergent. Jump threading must not duplicate or move them across divergent control flow.

The validation changed only the Triton source revision. Both builds used LLVM `f6ded0be897e2878612dd903f7e8bb85448269e5`, Python 3.13, PyTorch `2.13.0+cu130`, CUDA 12.8 tools, NVIDIA B200, driver `580.82.07`, the same reproducer, and separate fresh Triton caches.

| Gate | Parent `d077424a` | Fix `a6ffe3ce` | Acceptance criterion |
|---|---:|---:|---|
| Reproducer | Exit 124 after 15 seconds | Exit 0 | Fixed build completes and returns `[5, 99]`. |
| Output | No result | `[5, 99]` | Both stores are visible. |
| LLVM revision | `f6ded0be` | `f6ded0be` | Toolchain is held constant. |
| CUTracer | Runtime stall localized on parent | Not rerun on fix | Reproducer A/B is the fix oracle. |

The fix commit also adds `test_named_barrier_wait_1warp_async_deadlock` and `test_named_barrier_wait_1warp_async_deadlock_single_proc`. These tests preserve the divergent named-barrier pattern as a compiler regression.
