# Claude: conclusion excerpt

Excerpt of the agent's frozen `conclusion.md` from its solo run. Text is verbatim except that markdown links were removed and workspace paths and GPU indices were replaced with placeholders. File names refer to the agent's workspace; only the files in this directory are published.

## Verdict

**Defect found: an LDS write-after-read race between waves.** In `BlockFmhaPipelineQRKSVSAsync`, the first V tile (V0) is stored into the same LDS buffer that the last K slice (K2) is read from by the tail `gemm_0`. No barrier separates the two. Any wave can overwrite K2 with V data while another wave of the same workgroup is still reading K2. The reading wave then computes S = QKᵀ with V values in hdim columns 64..95.

The kernel audited is `fmha_fwd_d96_bf16_batch_b128x128x32x128x32x96_r4x1x1_r4x1x1_w32x32x16_w32x32x16_qr_async_vr_psskddv_nlogits_nbias_nmask_lse_ndropout_nskip_nqscale_ntrload_nsink` (run_kname.log). It is code object 76 of `build/bin/tile_example_fmha_fwd` (binary sha256 `7d256ce0…993d`, code object sha256 `edc21541…60b6`, co/kernel76.sha256).

Boundary: the race exists when `LdsSeq[k0_loops] == LdsSeq[k0_loops-1]`. With `NumPrefetchK = NumPrefetchV = 3`, kN0=128 and kK1=32 (k1_loops=4), this holds for hdim 96 (k0_loops=3, LdsSeq = 0,1,2 | 2,0,1,2). The hdim-96 restriction in this build is the configuration that exposes it. It does not depend on s, s_k or batch: every workgroup executes the edge in every iteration of the KV loop.

Why the existing checks pass:
- Waves rarely drift that far apart after a barrier. 200 natural runs showed no failure (see below).
- The default init (uniform [0,1]) together with the bf16 tolerance (rtol=atol=1e-2) hides almost all of the error even when the race is forced. With 200 keys and narrow logits, O stays near mean(V), about 0.5, whatever K holds.
- `-v=2` checks O only. It never checks LSE.

## Claims and evidence

| # | Claim | Status | Artifact |
|---|---|---|---|
| 1 | The command selects the d96 bf16 batch `qr_async` kernel with psskddv padding, LSE on, no mask, no bias, no dropout. | observed | run_kname.log |
| 2 | `-v=2` compares O against a naive GPU kernel and never checks LSE. `-v=1` checks O and LSE. | observed | ck/example/ck_tile/01_fmha/fmha_fwd_runner.hpp:1390-1439, 1976-1990 |
| 3 | Tile 128x128x32x128x32x96 with NumKVLdsBuffers=3 gives LdsSeq = [0,1,2 \| 2,0,1,2] for K0,K1,K2 \| V0..V3. K and V buffer *i* use the same bytes, because both descriptors use stride `GetSingleSmemElementSpaceSize`. | inferred (from source) | block_fmha_pipeline_qx_ks_vs_custom_policy.hpp:276-296, 552-660 |
| 4 | In source, the tail `gemm_0` (pipeline line 449) reads K2 after the last s_barrier (line 443), and `store_tile` of V0 (line 604) writes the same buffer. Only intra-wave reductions and `sched_barrier`s lie between them. | inferred (from source) | block_fmha_pipeline_qr_ks_vs_async.hpp:442-616 |
| 5 | In the shipped ISA, the K2 reads `ds_read_b128 v177+{0..4704}` (0x2F0C-0x2F68) follow the last QK barrier at 0x2E5C. The V0 store `ds_write2_b32 v189` (0x35D4-0x35EC) writes the same 8704-byte LDS region (v189 = v157 = the V3 base; regions: K2/V0/V3 @0, K0/V1 @8704, K1/V2 @17408). No s_barrier on any path between them; the only LDS op between them is ds_bpermute. | observed | co/kernel76.dis lines 738-1105; co/excerpt_k2_v0_race.txt; co/sync_skeleton.txt |
| 6 | Every wave reads the whole K2 slice (128x32 bf16 = 8 KiB: 8 `ds_read_b128` per lane). Each wave writes a 2176-byte strip of V0 (base = warp*2176). Every other wave's V0 write therefore overlaps bytes a reading wave needs. | inferred (from ISA address arithmetic) | co/kernel76.dis lines 565-600, 1055-1062 |
| 7 | All other LDS edges in the loop are ordered: the reader waits `s_waitcnt lgkmcnt(0)` (or on its own MFMA data dependence before the back-edge) and then hits an `s_barrier` before the writer proceeds. Async K loads use `vmcnt` + `s_barrier` before reads (table below). | observed (ISA) / inferred (rule) | co/sync_skeleton.txt |
| 8 | Holding waves 0-2 for 32x`s_sleep 127` between the last QK barrier and the K2 reads makes the original kernel produce wrong O: 81.55% of elements wrong, max err 6.95 (`-init=nf`, 2/2 runs). This adds latency only, a legal schedule. | causal | patches/delay_ne3.patch, run_delay_ne3_nf_v2.log |
| 9 | The same delay with the fix applied gives valid:y in 6/6 runs (uf and nf). | causal | patches/fix_plus_delay_ne3.patch, run_fix_delay_ne3_v2.log |
| 10 | Under the default uniform init the forced race is nearly invisible: a wave-0 delay shows 1 wrong element (err 0.0156); a waves-0-2 delay shows valid:y. | observed | run_delay_v2.log, run_delay32_v2.log, run_delay_ne3_v2.log |
| 11 | The unmodified kernel passes `-init=nf` (3/3 runs, `-v=2`), so claim 8's failures come from the delay exposing the race, not from the init. | observed | run_orig_nf_v2.log |
| 12 | The unmodified kernel showed no natural failure in 200 runs with `-v=2 -init=nf` (200/200 valid:y; only the last of 25 launches per run is checked), plus 3/3 in a probe run. This bounds the natural rate on this host and load; it does not show the ordering exists. | observed | stress_orig_nf_200.log, stress_probe_orig_nf_3.log |
| 13 | `-v=1 -init=nf` reports 3 O mismatches at batch 1211 (max err 0.0137) on both the original and the fixed build, identical values. This is a CPU-reference precision artifact of that init and is unrelated to the race. | observed | run_orig_nf_v1.log, run_fix_nf_v1.log |
| 14 | Natural occurrence needs one wave to fall behind its siblings by roughly a tail-GEMM-plus-softmax interval right after a barrier. Possible causes include context-switch preemption, SIMD contention from co-resident workgroups (kBlockPerCu=2) or other processes, and clock or arbitration effects. The memory model allows it. | inferred | AMD ISA guide: s_barrier synchronizes arrival only and gives no bound on progress afterwards |

## LDS edge table (one KV iteration, shipped ISA)

| Edge | Region | First access (PC) | Second access (PC) | Mechanism between them (PC) | Verdict |
|---|---|---|---|---|---|
| K2 read -> V0 write (WAR) | @0 | ds_read_b128 0x2F0C-0x2F68 | ds_write2_b32 0x35D4-0x35EC | **none** (last barrier 0x2E5C precedes the reads) | **violation** |
| K0 read -> V1 write (WAR) | @8704 | ds_read_b128 0x2DA8-0x2DE0 | ds_write2_b32 0x3AA4 | lgkmcnt(0) 0x2FA0 (reader), then s_barrier 0x3824 | ordered |
| K1 read -> V2 write (WAR) | @17408 | ds_read_b128 0x2E18-0x2E50 | ds_write2_b32 0x3CC0 | lgkmcnt(0) 0x2FA0, s_barrier 0x3824 | ordered |
| V0 write -> V0 read (RAW) | @0 | 0x35D4 | ds_read 0x3830 | lgkmcnt(0) 0x3820 + s_barrier 0x3824 | ordered |
| V1 write -> V1 read (RAW) | @8704 | 0x3AA4 | 0x3ACC | lgkmcnt(0) 0x3AC4 + s_barrier 0x3AC8 | ordered |
| V2 write -> V2 read (RAW) | @17408 | 0x3CC0 | 0x3CE8 | lgkmcnt(0) 0x3CE0 + s_barrier 0x3CE4 | ordered |
| V0 read -> V3 write (WAR) | @0 | 0x3830.. | 0x3E9C | lgkmcnt(0) 0x3AC4 + s_barrier 0x3AC8 | ordered |
| V3 write -> V3 read (RAW) | @0 | 0x3E9C | 0x3F24 | lgkmcnt(0) 0x3F1C + s_barrier 0x3F20 | ordered |
| V1 read -> next K0 async write (WAR) | @8704 | 0x3ACC.. | buffer_load lds 0x3EF0/0x3F04 | lgkmcnt(0) 0x3CE0 + s_barrier 0x3CE4 | ordered |
| V2 read -> next K1 async write (WAR) | @17408 | 0x3CE8.. | buffer_load lds 0x2D84 (loop head 0x2D78 via s_branch 0x401C) | lgkmcnt(0) 0x3F1C + s_barrier 0x3F20 | ordered |
| V3 read -> next K2 async write (WAR) | @0 | 0x3F24.. | buffer_load lds 0x2DF4 | lgkmcnt(0) 0x3FEC + s_barrier 0x2DA4 | ordered |
| K0/K1/K2 async write -> ds_read (RAW) | all | buffer_load lds | 0x2DA8 / 0x2E18 / 0x2F0C | vmcnt(2) 0x2DA0 / vmcnt(2) / vmcnt(0) 0x2E58, each + s_barrier | ordered |
| K2 async write -> V0 write (WAW) | @0 | 0x2DF4 | 0x35D4 | vmcnt(0) 0x2E58 + s_barrier 0x2E5C | ordered |
