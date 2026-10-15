# CUTracer investigates a TLX named-barrier kernel hang on B200: the trace ends at complementary `WARPSYNC.ALL` sites before `BAR.SYNC`

**Status:** Bounded · **Tags:** CUTracer, TLX, deadlock, B200, named barrier · **Example:** [repro.py](repro.py)

## TL;DR

- **Symptom:** `named_barrier_divergence` exceeded 15 seconds on one NVIDIA B200 run and the reproducer exited 124.
- **Conclusion:** In the decisive instrumented run, warp 6 ended at two predicated `WARPSYNC.ALL` instructions with complementary active masks. No later warp-6 instruction or dynamic record for the following named barriers was captured. This bounds the immediate failure to generated synchronization and control flow before `BAR.SYNC`; it does not prove that the named barrier instruction blocked.
- **Outcome:** The blind investigation left compiler ownership and the TLX source contract unresolved. A later comparison found a public Triton fix for non-convergent named-barrier lowering, but this investigation did not preserve a matched runtime run of that fix.
- **CUTracer contribution:** CUTracer recorded the terminal PCs, active masks, launch shape, and capture completeness. Those records changed the next action from named-barrier state inspection to generated control-flow and compiler-lowering inspection.

## Investigation timeline

**Coding agent:** Codex · **Model:** GPT-5 · **Total investigation time:** T+4 min to the bounded runtime localization.

| Elapsed | Action and evidence | Decision or next action |
|---|---|---|
| T+0 min | The baseline exceeded 15 seconds and exited 124. | Trace the launched kernel without assuming that the named barrier blocked. |
| T+1 min | CUTracer identified `named_barrier_divergence`, one CTA, 256 threads, and 8 warps. A sync-only capture produced no dynamic records. | Treat the zero-record capture as inconclusive and capture all opcodes. |
| T+2 min | The all-opcode capture recorded 172 instructions. Warp 6 ended at two predicated `WARPSYNC.ALL` PCs. | Capture active masks with `reg_trace`. |
| **T+4 min, runtime boundary localized** | CUTracer recorded mask `0x1` at PC `0x220` and mask `0xfffffffe` at PC `0x280`. It recorded no later warp-6 instruction and no dynamic named-barrier instruction. | Bound the conclusion to generated synchronization and control flow before the named barriers. |

The preserved evidence has no matched run of the fixed compiler. The investigation therefore has no defect-boundary confirmation milestone.

## Case identity

| Field | Value |
|---|---|
| Reproducer | [repro.py](repro.py) at this report revision |
| Affected component/configuration | `named_barrier_divergence`; grid `(1, 1, 1)`; block `(256, 1, 1)`; one TLX asynchronous task with one warp on NVIDIA B200 |
| Original artifact | SASS target `sm_100a`; kernel checksum `44f7b5fccc466eb5`; cubin SHA-256 `67edcfb2f1bc98de2dc20dc19e37bd28c2ddf4a1cbb6eaccf9427e91f3986376` |
| Treatment artifact | None. Triton commit `a6ffe3ce` was inspected later as source evidence only. |
| Recognition signature | Warp 6 at PC `0x220` with mask `0x1` and PC `0x280` with mask `0xfffffffe`; zero records at PCs `0x270` and `0x2c0` |
| Validated action | Inspect generated control flow and named-barrier lowering. A matched parent-versus-fix runtime test remains open. |

The blind artifacts omit the exact Triton, PyTorch, Python, driver, and CUTracer package versions. The commands selected CUDA 12.8 tools. A later check found the retained Triton checkout at the affected parent revision. That check does not establish the compiler version used during the original run.

## Symptom and reproducer

The reproducer launches one CTA with 256 threads. One TLX asynchronous task uses one warp. Lane 0 stores one value before `tlx.named_barrier_wait(14, 32)` and another value after it. All 32 lanes request the barrier.

The case depends on affected Triton source revision `d077424a67a74634e64ce35f644b2ac36eccd229`. The preserved evidence does not identify an equivalent `fbtriton` wheel. A current repository check found no exact release tag for the affected revision. Build the source revision to select the affected compiler:

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

The observed affected run printed:

```text
HANG REPRODUCED: kernel exceeded 15 seconds
```

The reproducer terminated the child after 15 seconds. It exited after 15.33 seconds with status 124. This result comes from one run on one B200.

| Variant | Changed variable | Result | Evidence |
|---|---|---|---|
| Original | No CUTracer instrumentation | Exceeded 15 seconds; exit 124 | One affected run confirms the high-level hang. |
| Instrumented diagnostic, not a treatment | `reg_trace`, one-record flushing, 8-second kernel timeout | 172 records; target terminated; trace marked truncated | One affected capture localizes the recorded execution. It does not validate a fix. |

## Evidence summary

| Claim | State | Producing tool | Evidence | Boundary |
|---|---|---|---|---|
| The kernel hangs. | Observed | Reproducer | The child remained alive for 15 seconds and the reproducer exited 124 after termination. | One uninstrumented run on one B200. |
| The target kernel launches and begins device execution. | Observed | CUTracer launch logger and register trace | Launch ID 1; kernel checksum `44f7b5fccc466eb5`; 172 dynamic records from 8 warps. | The capture ended by CUTracer timeout. |
| The last recorded warp-6 instructions are complementary predicated full-warp synchronization sites. | Observed | CUTracer `reg_trace` | PC `0x220`: `@!P0 WARPSYNC.ALL`, mask `0x1`. PC `0x280`: `@P0 WARPSYNC.ALL`, mask `0xfffffffe`. | The trace records instruction state at the `before` instrumentation point and is truncated. |
| The following named barriers were not dynamically recorded. | Observed | CUTracer query plus SASS | Zero records at `BAR.SYNC` PCs `0x270` and `0x2c0`; SASS places both after a recorded `WARPSYNC.ALL`. | The zero count applies only to this one-record-flush capture. It does not cover every execution. |
| Generated synchronization around the divergent region is the narrowest runtime boundary. | Inferred | CUTracer trace plus SASS | The masks partition warp 6 into lane 0 and lanes 1-31 at separate full-warp synchronization PCs, with no later warp-6 record. | Instrumentation can affect timing. The uninstrumented terminal PC was not measured. |
| The analyzer reported no named-barrier pairing mismatch finding. | Observed | CUTracer deadlock analyzer | The analyzer's `deadlock_barrier_mismatch` verdict was `clean` with zero findings. It separately classified 7 warps as barrier-blocked, 1 as unknown, 0 as progressing, and 0 as exited. | The analyzer marked the 172-record trace as truncated and the hang as soft/potential. `clean` here is not a safety verdict. |
| LLVM jump threading caused the failure. | Unresolved during the blind phase | None in the blind phase | CUTracer did not identify a compiler pass or determine whether the source violated a TLX contract. | Compared with source evidence only after the blind conclusion. |

The decisive CUTracer chain was:

```text
PC 0x220, active_mask 0x1, @!P0 WARPSYNC.ALL
+ PC 0x280, active_mask 0xfffffffe, @P0 WARPSYNC.ALL
+ zero dynamic records at BAR.SYNC PCs 0x270 and 0x2c0
-> warp 6 is recorded in complementary lane subsets at separate full-warp sync sites
-> inspect generated control flow and compiler lowering instead of claiming that BAR.SYNC blocked
```

The trace does not prove that either `WARPSYNC.ALL` instruction retired. It also does not show whether an uninstrumented run ends at the same PCs. It establishes only the last recorded instruction state for warp 6 in this capture.

## How CUTracer helped

CUTracer isolated `named_barrier_divergence` and confirmed device execution. One-record channel flushing retained the final warp-6 records before the 8-second instrumented timeout.

The active masks showed the lane split. Lane 0 was active at one `WARPSYNC.ALL`. Lanes 1 through 31 were active at the other. The following named-barrier PCs had no records.

These records ruled out a launch failure as the immediate boundary. They did not show that `BAR.SYNC` blocked. The next step was to inspect generated synchronization and compiler lowering.

## Blind investigation and later validation

### Blind phase

| Input or milestone | Evidence or decision available at that time |
|---|---|
| Starting prompt and artifacts | The runnable reproducer and symptom only. Adjacent documentation, history, and known fixes were not inspected. |
| Baseline | One uninstrumented run exceeded 15 seconds and exited 124. |
| Initial boundaries | Compilation or launch failure; the named barrier; generated synchronization around divergent control flow. |
| Launch and opcode traces | Launch succeeded. Eight warps began device execution. A sync-only capture produced no records, so it was inconclusive. |
| Decisive register trace | Warp 6 ended at PCs `0x220` and `0x280` with masks `0x1` and `0xfffffffe`; PCs `0x270` and `0x2c0` had zero records. |
| Blind conclusion | The narrowest supported boundary was generated synchronization and control flow before the named barriers. Compiler-pass ownership and source-contract validity remained unresolved. |

The blind diagnosis took about four minutes, from the 00:09:49 baseline start to the 00:13:45 decisive capture completion. The later fix comparison does not strengthen that conclusion.

### Post-investigation validation

After the blind conclusion, the investigation compared Triton commit [`a6ffe3ce`](https://github.com/facebookexperimental/triton/commit/a6ffe3ce5ee1ade3596332c85ce3a1e6ea377630) with its parent, `d077424a`. The diff:

- It replaces PTXBuilder inline assembly for named-barrier wait with `llvm.nvvm.barrier.cta.sync.aligned.count`.
- It replaces PTXBuilder inline assembly for named-barrier arrive with `llvm.nvvm.barrier.cta.arrive.aligned.count`.
- It adds regression tests with the same divergent stores around `tlx.named_barrier_wait(14, 32)`.

The fix commit identifies missing convergent semantics and LLVM jump threading as the compiler cause. That explanation is consistent with the blind boundary. Both concern generated synchronization around divergent control flow. The blind capture adds dynamic evidence: this compiled artifact reached complementary lane masks at two `WARPSYNC.ALL` sites before the trace recorded either copied `BAR.SYNC`.

The blind investigation did not establish:

- CUTracer did not identify LLVM jump threading.
- CUTracer did not decide whether the source violated a documented TLX divergence contract.
- CUTracer did not validate the convergent NVVM intrinsic change.
- The source changes and tests do not replace a runtime trace of the fixed artifact.

The evidence set contains no runtime output from the fixed revision. This report therefore does not claim an exact parent-versus-fix A/B or upgrade the blind result to Confirmed.

## Comparison with the human investigation

Human report: [`[TLX] Fix named barrier deadlock caused by LLVM jump threading (#1040)`](https://github.com/facebookexperimental/triton/commit/a6ffe3ce5ee1ade3596332c85ce3a1e6ea377630)

| Topic | Agent with CUTracer | Human report | Gap or next test |
|---|---|---|---|
| Root cause | Bounded the affected runtime to generated synchronization and control flow before the named barriers. | Identifies LLVM jump threading and missing convergent semantics in the inline-assembly lowering. | Run the same reproducer on the parent and fix with one environment and separate fresh caches. |
| Hazard or failure coverage | Traced one `named_barrier_divergence` artifact on one B200. | Changes both named-barrier wait and arrive lowering and adds wait regression tests. | Trace the fixed wait artifact and test the arrive path separately. |
| Finding unit | Reports dynamic PCs and active masks for warp 6 in one truncated capture. | Reports compiler IR, source lowering, and regression-test structure. | Map the fixed IR to its SASS and complete runtime trace. |
| Independent validation | Preserves one affected baseline and one affected instrumented capture. | Records a test command and before-and-after compiler IR in the commit message. | Preserve public parent and fix outputs, generated-code identities, and exit statuses from a matched rerun. |

The CUTracer capture adds the exact active-mask values `0x1` and `0xfffffffe` at the terminal recorded PCs. The public human report does not record those mask values. The agent investigation did not identify the compiler pass, prove the source-level cause, or validate the fix at runtime.

## Resolution and validation

The known fix changes named-barrier lowering to convergent NVVM intrinsics. The exact parent-to-fix diff shows that code change. This investigation did not run the fixed kernel.

| Gate | Original | Treatment | Acceptance criterion |
|---|---:|---:|---|
| Reproducer | Exit 124 after 15 seconds | Not run in this investigation | Fixed revision exits 0 and returns `[5, 99]`. |
| Independent oracle | Not preserved | Not run in this investigation | A matched parent-versus-fix run changes only the compiler revision. |
| CUTracer | Runtime boundary localized on the affected artifact | Not run | The fixed artifact has no terminal split at complementary full-warp synchronization sites. |
| Regression/performance | Tests not run; performance not measured | Matching tests are present in commit `a6ffe3ce`, but were not run here | Tests pass on a supported GPU. No performance claim is required. |

To confirm causality, rebuild the parent and fix commits with the same GPU, driver, CUDA tools, Python packages, and launch configuration. Use separate fresh Triton caches. Preserve both outputs and generated-code identities.
