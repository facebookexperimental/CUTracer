# CUTracer investigates a TLX named-barrier kernel hang on B200: the trace ends at complementary `WARPSYNC.ALL` sites before `BAR.SYNC`

**Status:** Bounded | **Example:** [repro.py](repro.py)

## TL;DR

- **Symptom:** `named_barrier_divergence` exceeded 15 seconds on one NVIDIA B200 run and the reproducer exited 124.
- **Conclusion:** In the decisive instrumented run, warp 6 ended at two predicated `WARPSYNC.ALL` instructions with complementary active masks. No later warp-6 instruction or dynamic record for the following named barriers was captured. This bounds the immediate failure to generated synchronization and control flow before `BAR.SYNC`; it does not prove that the named barrier instruction blocked.
- **Outcome:** The blind investigation left compiler ownership and the TLX source contract unresolved. A later comparison found a public Triton fix for non-convergent named-barrier lowering, but this investigation did not preserve a matched runtime run of that fix.
- **CUTracer contribution:** CUTracer recorded the terminal PCs, active masks, launch shape, and capture completeness. Those records changed the next action from named-barrier state inspection to generated control-flow and compiler-lowering inspection.

## Case identity

| Field | Value |
|---|---|
| Reproducer | [repro.py](repro.py) |
| Kernel | `named_barrier_divergence`; grid `(1, 1, 1)`; block `(256, 1, 1)`; 8 warps |
| GPU | NVIDIA B200; SASS target `sm_100a` |
| Capture software | CUTracer `reg_trace`; one-record channel flushing; CUDA 12.8 `ptxas` selected by the command manifest |
| Original artifact | Kernel checksum `44f7b5fccc466eb5`; cubin SHA-256 `67edcfb2f1bc98de2dc20dc19e37bd28c2ddf4a1cbb6eaccf9427e91f3986376` |
| Treatment artifact | None in the blind phase. Triton commit `a6ffe3ce` was inspected later as source evidence only. |
| Recognition signature | Warp 6 at PC `0x220` with mask `0x1` and PC `0x280` with mask `0xfffffffe`; zero records at PCs `0x270` and `0x2c0` |
| Discriminating test | Open: run the same reproducer and environment at the affected revision and fix revision with separate caches |
| Validated action | Inspect generated control flow and named-barrier lowering; no runtime fix was validated in the blind artifact set |

The blind artifacts did not record exact Triton, PyTorch, Python, driver, or CUTracer package versions. The commands selected CUDA 12.8 tools, and the retained Triton checkout was later found at the affected parent revision. These later checks do not retroactively add version metadata to the original run.

## Symptom and reproducer

The reproducer launches one CTA with 256 threads. One TLX asynchronous task uses one warp. Lane 0 stores a value, all 32 lanes request `tlx.named_barrier_wait(14, 32)`, and lane 0 stores another value.

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

The observed affected run printed:

```text
HANG REPRODUCED: kernel exceeded 15 seconds
```

It completed after 15.33 seconds with exit status 124. This is one run on one B200.

| Run | Changed variable | Result | Evidence boundary |
|---|---|---|---|
| Baseline | No CUTracer instrumentation | Exceeded 15 seconds; exit 124 | Confirms the high-level hang without instrumentation |
| Decisive capture | `reg_trace`, one-record flushing, 8-second kernel timeout | 172 records; target terminated; trace marked truncated | Localizes the instrumented execution only; it is not a treatment or fix |

## Evidence summary

| Claim | State | Producing tool | Evidence | Boundary |
|---|---|---|---|---|
| The kernel hangs. | Observed | Reproducer | The child remained alive for 15 seconds and the reproducer exited 124 after termination. | One uninstrumented run on one B200. |
| The target kernel launches and begins device execution. | Observed | CUTracer launch logger and register trace | Launch ID 1; kernel checksum `44f7b5fccc466eb5`; 172 dynamic records from 8 warps. | The capture ended by CUTracer timeout. |
| The last recorded warp-6 instructions are complementary predicated full-warp synchronization sites. | Observed | CUTracer `reg_trace` | PC `0x220`: `@!P0 WARPSYNC.ALL`, mask `0x1`. PC `0x280`: `@P0 WARPSYNC.ALL`, mask `0xfffffffe`. | The trace records instruction state at the `before` instrumentation point and is truncated. |
| The following named barriers were not dynamically recorded. | Observed | CUTracer query plus SASS | Zero records at `BAR.SYNC` PCs `0x270` and `0x2c0`; SASS places both after a recorded `WARPSYNC.ALL`. | Meaningful for this one-record-flush capture; not proof about every execution. |
| Generated synchronization around the divergent region is the narrowest runtime boundary. | Inferred | CUTracer trace plus SASS | The masks partition warp 6 into lane 0 and lanes 1-31 at separate full-warp synchronization PCs, with no later warp-6 record. | Instrumentation can affect timing. The uninstrumented terminal PC was not measured. |
| A named-barrier pairing mismatch was detected. | Unresolved / not established | CUTracer deadlock analyzer | The analyzer's `deadlock_barrier_mismatch` verdict was `clean` with zero findings. It separately classified 7 warps as barrier-blocked, 1 as unknown, 0 as progressing, and 0 as exited. | The analyzer marked the 172-record trace truncated and the hang soft/potential. `clean` here is not a safety verdict. |
| LLVM jump threading caused the failure. | Unresolved during the blind phase | None in the blind phase | CUTracer did not identify a compiler pass or determine whether the source violated a TLX contract. | Compared with source evidence only after the blind conclusion. |

The decisive CUTracer chain was:

```text
PC 0x220, active_mask 0x1, @!P0 WARPSYNC.ALL
+ PC 0x280, active_mask 0xfffffffe, @P0 WARPSYNC.ALL
+ zero dynamic records at BAR.SYNC PCs 0x270 and 0x2c0
-> warp 6 is recorded in complementary lane subsets at separate full-warp sync sites
-> inspect generated control flow and compiler lowering instead of claiming that BAR.SYNC blocked
```

The trace does not prove that either `WARPSYNC.ALL` instruction retired or that the same terminal PCs occur without instrumentation. It proves the last recorded instruction state for warp 6 in this capture.

## How CUTracer helped

CUTracer isolated `named_barrier_divergence` and confirmed device execution. One-record channel flushing preserved the final warp-6 records before the 8-second instrumented timeout.

The active masks made the divergence concrete. Lane 0 was active at one `WARPSYNC.ALL`. Lanes 1 through 31 were active at the other. The following named-barrier PCs had no records.

This evidence bounded the next action. It rejected CUDA launch as the immediate boundary and did not support blaming the named-barrier instruction itself. It directed the investigation toward generated synchronization and compiler lowering.

## Blind investigation and later fix comparison

### Blind phase

| Input or milestone | Evidence or decision available at that time |
|---|---|
| Starting prompt and artifacts | The runnable reproducer and symptom only. Adjacent documentation, history, and known fixes were not inspected. |
| Baseline | One uninstrumented run exceeded 15 seconds and exited 124. |
| Initial boundaries | Compilation or launch failure; the named barrier; generated synchronization around divergent control flow. |
| Launch and opcode traces | Launch succeeded. Eight warps began device execution. A sync-only zero-record capture was treated as inconclusive. |
| Decisive register trace | Warp 6 ended at PCs `0x220` and `0x280` with masks `0x1` and `0xfffffffe`; PCs `0x270` and `0x2c0` had zero records. |
| Blind conclusion | The narrowest supported boundary was generated synchronization and control flow before the named barriers. Compiler-pass ownership and source-contract validity remained unresolved. |

The blind diagnosis took about four minutes, from the 00:09:49 baseline start to the 00:13:45 decisive capture completion. No later evidence is used to strengthen that conclusion.

### Post-investigation comparison with the known fix

After the blind conclusion, Triton commit [`a6ffe3ce`](https://github.com/facebookexperimental/triton/commit/a6ffe3ce5ee1ade3596332c85ce3a1e6ea377630) was inspected. Its parent is `d077424a`. The source diff does three relevant things:

- It replaces PTXBuilder inline assembly for named-barrier wait with `llvm.nvvm.barrier.cta.sync.aligned.count`.
- It replaces PTXBuilder inline assembly for named-barrier arrive with `llvm.nvvm.barrier.cta.arrive.aligned.count`.
- It adds regression tests with the same divergent stores around `tlx.named_barrier_wait(14, 32)`.

The fix commit identifies missing convergent semantics and LLVM jump threading as the compiler cause. That explanation is consistent with the blind boundary: both point to duplicated or separated synchronization in generated control flow. The blind capture adds dynamic evidence that this compiled artifact reached complementary lane masks at two `WARPSYNC.ALL` sites before either copied `BAR.SYNC` was recorded.

The comparison also shows what the blind investigation did not establish:

- CUTracer did not identify LLVM jump threading.
- CUTracer did not decide whether the source violated a documented TLX divergence contract.
- CUTracer did not validate the convergent NVVM intrinsic change.
- The fix commit's source and test changes are a different evidence class from the blind runtime trace.

No runtime output from the fixed revision is included in the handed-off artifacts. Therefore, this report does not claim an exact parent-versus-fix A/B or upgrade the blind result to Confirmed.

## Resolution and validation

The known fix changes named-barrier lowering to convergent NVVM intrinsics. This report verifies that source change by inspecting the exact parent-to-fix diff. It does not verify the fixed kernel at runtime.

| Gate | Affected artifact | Fix artifact | Acceptance criterion |
|---|---:|---:|---|
| Reproducer | Exit 124 after 15 seconds | Not run in this investigation | Fixed revision completes and returns `[5, 99]` |
| Generated code | Two predicated `WARPSYNC.ALL` sites precede two predicated `BAR.SYNC` sites | Not captured | One converged named-barrier operation for the participating warp |
| CUTracer | Runtime boundary localized on affected artifact | Not run | No terminal split at complementary full-warp synchronization sites |
| Regression tests | Not run | Matching tests present in commit `a6ffe3ce` | Tests execute and pass on a supported GPU |

A future causal confirmation should rebuild the parent and fix commits with the same GPU, driver, CUDA tools, Python packages, launch configuration, and separate fresh Triton caches. It should preserve both outputs and generated-code identities.
