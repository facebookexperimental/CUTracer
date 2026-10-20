# Codex: conclusion excerpt

Excerpt of the agent's frozen `conclusion.md`. Text is verbatim except that markdown links were removed and workspace paths and GPU indices were replaced with placeholders. File names refer to the agent's workspace; only the files in this directory are published.

## Verdict

**Causal:** The selected bf16 FMHA forward kernel has a cross-wave LDS reuse race. Its final K stage and first V stage share buffer 0. A wave can overwrite K bytes with V before another wave finishes reading those K bytes. A controlled wave delay makes GPU output validation fail. Adding a workgroup barrier before the V overwrite removes the failures under the same delay. Evidence: paired results, paired trial log, ISA diff.

**Observed:** The unmodified executable passed the one baseline run performed here. The earlier 500/500 passing runs are user-provided observations. This investigation did not reproduce a failure in the unmodified executable. Evidence: baseline.

## Buffer map and offending edge

**Observed:** The source policy specializes the stage sequence to `K = 1, 2, 0` and `V = 0, 1, 2, 0`. The code object allocates 26,112 LDS bytes for a 256-thread workgroup. Evidence: policy, metadata, source excerpts.

**Inferred:** The three shared K/V buffers have stride 8,704 bytes. Each allocation includes layout padding; accesses do not fill every byte. The buffer map follows the policy descriptors and agrees with ISA offsets `0x2200` and `0x4400`. Evidence: address definitions, policy.

| Buffer | Byte allocation | Uses in one key tile | Evidence state |
|---|---|---|---|
| 0 | `0, 8704)` | K stage 2, V stage 0, V stage 3 | Inferred from descriptors and ISA |
| 1 | `[8704, 17408)` | K stage 0, V stage 1 | Inferred from descriptors and ISA |
| 2 | `[17408, 26112)` | K stage 1, V stage 2 | Inferred from descriptors and ISA |

**Observed:** K stage 2 reads at PCs `0x2F0C` through `0x2F68`, using `ds_read_b128` and base register `v177`. Each wave drains its own reads by `s_waitcnt lgkmcnt(0)` at `0x2FA0`. V stage 0 writes at PCs `0x35D4` through `0x35EC` with `ds_write2_b32` and base `v189`. There is no `s_barrier` between those reads and writes. The previous barrier is `0x2E5C`, before the K-tail reads; the next is `0x3824`, after the V writes. Evidence: [edge excerpts, complete ISA.

**Inferred:** A concrete alias is wave 0, lane 0, reading bytes `4160, 4176)` at PC `0x2F44`, and wave 1, lane 56, writing `[4160, 4164)` at PC `0x35E4`, the second store of `offset0:16 offset1:20`. The reader base is 0 plus offset 4160. The writer base is 4080 plus 80 bytes. The arithmetic follows the original register definitions. Evidence: [alias witness, calculation script, address definitions.

**Inferred:** The required edge is read-before-write across different waves, using ordinary LDS instructions. Local `lgkmcnt(0)` waits do not join the waves. The missing mechanism is a workgroup barrier after every wave has completed its K reads and before any wave issues the overlapping V stores. Scheduler barriers and the wave-local max reduction do not supply that join. Evidence: original pipeline, ISA, reference status.

## Intervention and verification

**Observed:** Experiments use source overlays under `out/skew/`, `out/skew-fix/` and `out/fix/`. Only the selected translation unit was recompiled, then linked with the existing other objects. Every variant has its exact `source.patch`, `compile.sh`, `link.sh` and `kernel.isa`. The original CK source was never edited. Evidence: variant builder, source hash check.

**Observed:** The skew adds 64 `s_sleep 10` instructions to wave 0 after the K-publication barrier and before its K-tail reads. It changes no mathematical operation in the source. The proposed fix adds `block_sync_lds()` before the first V store when the final K buffer equals the first V buffer. Evidence: skew patch, fix patch, skew ISA.

| Variant at the requested shape | GPU output validation | Evidence state and artifact |
|---|---|---|
| Original | 1/1 passed | Observed, baseline |
| Skew only | Initial run failed; 10/10 paired trials failed | Observed, initial failure, paired results |
| Same skew with fix | 10/10 paired trials passed | Observed, paired results |
| Fix without skew | 1/1 passed | Observed, fix validation |

**Observed:** The repeated skew failure reports output index 97105 as `0.5273438` versus reference `0.5117188`, one value outside the validator tolerance. Evidence: trial 0 and the other paired trial logs.

**Causal:** Adding the join removes the failure while preserving the injected delay. Both the original-to-fix and skew-to-skew-fix ISA diffs add only `s_waitcnt lgkmcnt(0)` and `s_barrier`, plus branch displacement adjustments. Arithmetic instructions, register operands and memory instructions remain unchanged within each comparison. This isolates the ordering change from numerical code changes. Evidence: original-to-fix diff, paired diff.

**Observed:** In the fix-only binary, the added wait is at `0x3458` and the added barrier at `0x345C`, before the first V stores at `0x35DC` through `0x35F4`. The compiler moved the join earlier than its source position while keeping it after the K-tail reads. Evidence: added barrier excerpt, fixed ISA.

## Proposed fix

**Inferred:** Apply fix/source.patch. Its conditional join closes the demonstrated alias without adding a barrier when the adjacent K and V stages use different buffers. The paired intervention verifies the correction for the audited shape and binary. Other policy sequences need their own review before a broader correctness claim.

```cpp
if constexpr(LdsSeq.at(number<k0_loops - 1>{}) ==
             LdsSeq.at(number<k0_loops>{}))
    block_sync_lds();
```
