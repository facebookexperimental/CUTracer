# CUTracer investigates a CK FMHA forward kernel that passes validation on MI350: missing workgroup barrier before the V tile overwrites the K tail in LDS

**Status:** Confirmed. The defect is an LDS write-after-read race across waves. With an injected wave delay it fails validation; at the default schedule no failure was observed. **Example:** [`examples/ck_fmha_kv_lds_war`](README.md).

**Tags:** CUTracer, `audit-memory-model`, AMD, ROCm, MI350, gfx950, Composable Kernel (CK), ck_tile, FMHA forward, `BlockFmhaPipelineQRKSVSAsync`, LDS, `s_barrier`, `s_waitcnt lgkmcnt`, write-after-read, cross-wave race, direct-to-LDS, blind investigation, Claude, Codex, Muse.

## TL;DR

- **Symptom:** none at the default schedule. The bf16 FMHA forward example passes GPU validation in 500 of 500 runs at the audited shape.
- **Conclusion:** in `BlockFmhaPipelineQRKSVSAsync`, the last K stage and the first V stage use the same LDS buffer. Each wave waits for its own K reads, but no barrier joins the waves before the V stores. One wave can overwrite K bytes that another wave has not read.
- **Outcome:** three coding agents audited the kernel blind with the CUTracer `audit-memory-model` skill. Claude and Codex found the defect and fixed it at the position of the upstream fix, with the same condition. Muse audited the right edge but missed the first V stores, and concluded "no defect found".
- **CUTracer contribution:** the skill's AMD rules and audit procedure. CUTracer's trace capture does not run on AMD, so the agents audited the AMDGCN and the source. No CUTracer trace was used.

## Investigation timeline

Each agent ran once, with the same prompt, its own workspace, and its own MI350 GPU. Times are from each agent's timestamped `timeline.md`, measured from the launch.

| Agent (client, model) | Total | Root cause localized | Defect boundary confirmed |
|---|---|---|---|
| Claude Code 2.1.291, `claude-opus-5-5` (solo rerun) | 57 min | **T+6 min**: source audit finds that the last K stage and the first V stage share a buffer with no barrier between the tail `gemm_0` reads and the V store; **T+8 min**: confirmed in the AMDGCN | **T+13 min**: delaying waves 0–2 makes validation fail (81.55% of outputs wrong with `-init=nf`); **T+24 min**: the same delay with its fix passes 6 of 6. The rest of the run was a 200-run stress test of the unmodified kernel (200/200 valid). |
| Codex CLI 0.162.1, `gpt-6.1-sol` | 9 min | **T+3 min**: hypothesis that the K tail in buffer 0 must finish in every wave before V stage 0 overwrites it; the ISA shows no `s_barrier` between the K-tail reads and the first V stores | **T+5 min**: a delay of wave 0 makes validation fail in 10 of 10 paired trials; the same delay with a barrier added passes 10 of 10 |
| Muse Code 1.4.11, `muse-spark-1.3-internal` | 32 min | Not localized. The agent mapped the K and V buffers correctly but took the first V store into buffer 0 to be at PC `0x3AA4`, after the barrier at `0x3824`. It missed the V stage-0 stores at `0x35D4`–`0x35EC`, before that barrier. | n/a; concluded "no defect found" |

The three agents first ran side by side on one host. In that run, Claude's `ps` printed Codex's launch command, which contains only the shared prompt. No hypothesis or finding leaked, and Claude's first run reached the same conclusion at T+9 min. To keep the reported run free of any exposure, this report uses a solo rerun of Claude in a fresh workspace. Codex and Muse never read the process table.

## Case identity

| Field | Value |
|---|---|
| Reproducer | [README](README.md) in this directory: CK `tile_example_fmha_fwd` at commits `683865895e` (affected) and `fe8b7d0c` (fixed), with [`optdim96.patch`](optdim96.patch) and [`delay_wave0.patch`](delay_wave0.patch) |
| Affected component | `ck_tile::BlockFmhaPipelineQRKSVSAsync` (`include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async.hpp`) when `LdsSeq` puts the last K stage and the first V stage in the same buffer; audited instance `fmha_fwd_d96_bf16_batch_b128x128x32x128x32x96_..._qr_async_vr_psskddv_..._lse_...` on gfx950 |
| Recognition signature | AMDGCN: K-tail `ds_read_b128` (PCs `0x2F0C`–`0x2F68`), then `s_waitcnt lgkmcnt(0)` (`0x2FA0`), then V stage-0 `ds_write2_b32` into the same bytes (`0x35D4`–`0x35EC`), with no `s_barrier` between `0x2E5C` and `0x3824`. Fixed state: one more `s_barrier` in each affected kernel. |
| Validated action | Fix: a workgroup barrier before the V store when `LdsSeq.at(k0_loops - 1) == LdsSeq.at(k0_loops)`. Upstream: [ROCm/composable_kernel@fe8b7d0c](https://github.com/ROCm/composable_kernel/commit/fe8b7d0c277f309d2a0f497c35aec3cb1e87e0dd). |

## Symptom and reproducer

The [README](README.md) builds both states from public CK commits and runs them. The command is the example's bf16 forward at b=4096, h=1, s=16, s_k=200, d=96, with the LSE output and GPU validation (`-v=2`). `-v=2` compares the output O with a reference computed on the GPU; it does not compare LSE.

| Variant | Changed variable | Result | Evidence |
|---|---|---|---|
| Affected `683865895e` | none | 0 of 500 runs invalid | [`run_results.txt`](evidence/run_results.txt) |
| Fixed `fe8b7d0c` | upstream barrier | 0 of 200 runs invalid | [`run_results.txt`](evidence/run_results.txt) |
| Affected + delay | wave 0 sleeps before its K-tail reads | 20 of 20 runs invalid | [`run_results.txt`](evidence/run_results.txt) |
| Fixed + delay | same delay | 0 of 20 runs invalid | [`run_results.txt`](evidence/run_results.txt) |

At the default schedule, the validation does not discriminate the two states. The defect is visible in the AMDGCN, and the delay makes it fail.

## Evidence summary

| Claim | State | Producing tool | Evidence | Boundary |
|---|---|---|---|---|
| The kernel passes validation at the default schedule | observed | `tile_example_fmha_fwd -v=2` | [`run_results.txt`](evidence/run_results.txt) | MI350, one shape, one instance |
| The last K stage and the first V stage share LDS buffer 0 | observed (source), inferred (byte map) | Source reading; `llvm-objdump` | `LdsSeq` gives K stages 1, 2, 0 and V stages 0, 1, 2, 0; buffer stride 8,704 bytes ([Codex excerpt](evidence/runs/codex/conclusion_excerpt.md)) | This tile and head dimension |
| No `s_barrier` separates the K-tail reads from the V stage-0 stores | observed | `roc-obj`, `llvm-objdump` | [`edge-excerpts.isa`](evidence/runs/codex/edge-excerpts.isa); the fixed state has one more `s_barrier` per affected kernel ([`isa_barriers.txt`](evidence/isa_barriers.txt)) | ROCm 7.0, gfx950 |
| A wave can overwrite another wave's unread K bytes | inferred | AMD ISA rules, as quoted in the skill references | CDNA3/CDNA4 ISA guides: "Barrier instructions do not wait for any counters to go to zero"; LLVM `AMDGPUUsage`: LDS operations of different wavefronts "can be reordered relative to each other". A concrete alias: wave 0 lane 0 reads bytes 4160–4175 at `0x2F44`; wave 1 lane 56 writes bytes 4160–4163 at `0x35E4` ([Codex excerpt](evidence/runs/codex/conclusion_excerpt.md)). | Rule-based |
| The barrier is the missing mechanism | causal | Source patches, `hipcc`, `-v=2` | Delay: affected 20/20 invalid, fixed 0/20 ([`run_results.txt`](evidence/run_results.txt)). Codex: the same delay with its barrier, 10/10 valid vs 10/10 invalid without ([`paired-results.json`](evidence/runs/codex/paired-results.json)); its fix adds only `s_waitcnt lgkmcnt(0)` and `s_barrier` to the ISA ([`original-to-fix.isa.txt`](evidence/runs/codex/original-to-fix.isa.txt)). | One shape and instance |

## How CUTracer helped

- The `audit-memory-model` skill's AMD references gave the rules the agents applied: `s_barrier` does not wait for memory operations, and LDS operations of different waves are not ordered without a barrier. All three agents read `SKILL.md`, `amd-common.md`, the gfx950 reference, and the audit worksheet.
- CUTracer's trace capture does not run on AMD, and the skill says so. No agent used a CUTracer trace. The evidence is the AMDGCN, the source, and the validation runs.
- Muse's miss shows a limit of a manual ISA audit: the agent checked the edge but took a later store as the first write into the buffer.

## Resolution and validation

The fix is [ROCm/composable_kernel@fe8b7d0c](https://github.com/ROCm/composable_kernel/commit/fe8b7d0c277f309d2a0f497c35aec3cb1e87e0dd). It adds, after `__builtin_amdgcn_sched_barrier(0x7F)` and before the V store:

```cpp
if constexpr(LdsSeq.at(number<k0_loops - 1>{}) == LdsSeq.at(number<k0_loops>{}))
{
    __builtin_amdgcn_s_barrier();
}
```

| Gate | Affected | Fixed | Acceptance criterion |
|---|---:|---:|---|
| `-v=2`, default schedule | 0/500 invalid | 0/200 invalid | No regression. Not discriminating. |
| `-v=2`, wave 0 delayed | 20/20 invalid | 0/20 invalid | The fix removes the failure under the same delay |
| AMDGCN `s_barrier` per affected kernel | see [`isa_barriers.txt`](evidence/isa_barriers.txt) | one more | One barrier added before the V stage-0 stores |

## Blind investigation and later validation

### Blind phase

| Input or milestone | Evidence or decision available at that time |
|---|---|
| Starting prompt and artifacts | The [prompt](evidence/prompt.md): the affected CK source as a snapshot without version-control history, a gfx950 build restricted to head dimension 96, the command, ROCm 7.0 tools, and "500 of 500 runs reported valid:y". It names no defect, symptom, fix, or skill. It says only that CUTracer's agent skills are installed. |
| Skills available | The CUTracer OSS skills `audit-memory-model` and `report-case-study`, found through each client's own skill mechanism. |
| Tool results | Codex: the ISA edge, a buffer map, an address alias, and a delay experiment with and without a barrier. Muse: a buffer map, a mis-mapped first V store, and a triage barrier it judged redundant. Claude: a source and ISA edge table, delay experiments with uniform and normal inputs, a fix, and a stress test. |
| Blind conclusions | Claude: a violation, with a fix and paired validation. Codex: a violation, with a fix and paired validation. Muse: "no defect found". Excerpts: [Claude](evidence/runs/claude/conclusion_excerpt.md), [Codex](evidence/runs/codex/conclusion_excerpt.md), [Muse](evidence/runs/muse/conclusion_excerpt.md). |

**What this blind run tests.** The `audit-memory-model` skill was written on NVIDIA cases. Its AMD references were checked against the AMD ISA guides and LLVM `AMDGPUUsage`, but no AMD kernel was audited before this case, and the skill contains nothing about CK or FMHA. This case is therefore the first test of the skill outside the kernels it was built from.

### Post-investigation validation: comparison with the upstream fix

| Topic | Agents | Upstream fix (`fe8b7d0c`) | Gap |
|---|---|---|---|
| Root cause | Claude and Codex: the K-tail reads and the V stage-0 stores share one LDS buffer with no workgroup barrier between them | The same: "Ensure gemm_0's LDS reads (K tile) from all threads are completed before V store" | None for Claude and Codex |
| Fix | Claude ([patch](evidence/runs/claude/fix.patch)) and Codex ([patch](evidence/runs/codex/fix.patch)): `block_sync_lds()` under `if constexpr(LdsSeq.at(number<k0_loops - 1>{}) == LdsSeq.at(number<k0_loops>{}))`, after `sched_barrier(0x7F)`. `block_sync_lds()` adds `s_waitcnt lgkmcnt(0)` before the barrier. | `__builtin_amdgcn_s_barrier()` under the same condition, at the same position | The extra wait is redundant here (each wave already waits at `0x2FA0`) but harmless |
| Missed | Muse concluded "no defect found" after mis-mapping the first V store | n/a | Check every write into a reused buffer, not only the first one found |
