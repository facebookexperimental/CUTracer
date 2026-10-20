# Muse: conclusion excerpt

Excerpt of the agent's frozen `conclusion.md`. Text is verbatim except that markdown links were removed and workspace paths and GPU indices were replaced with placeholders. File names refer to the agent's workspace; only the files in this directory are published.

## Verdict

No defect found. The launched kernel is correct with respect to the AMD
LDS/vector-memory ordering rules for every cross-wave, cross-path edge its
loop structure implies at this shape, and all validation runs pass
(500/500 reported before this session plus 6/6 fresh seeds run here, all
`valid:y`). The triage-only hardening patch I tested is redundant: the
shipped binary already orders the one edge that has no explicit
source-level sync, via a compiler-placed barrier. I propose no source
change.

Narrowest boundary established: instance
`fmha_fwd_d96_bf16_batch_b128x128x32x128x32x96_r4x1x1_r4x1x1_w32x32x16_w32x32x16_qr_async_vr_psskddv_nlogits_nbias_nmask_lse_ndropout_nskip_nqscale_ntrload_nsink`
(`BlockFmhaPipelineQRKSVSAsync`, k0=3/k1=4, `LdsSeq<1,2,0,0,1,2,0>`),
as built in `<workspace>/build/bin/tile_example_fmha_fwd` for gfx950
(ROCm 7.0, `--offload-arch=gfx950 -O3`, `CK_TILE_FMHA_FWD_FAST_EXP2=1`),
run as GPU <n> with the exact flags in the task. Other instances, shapes,
dtypes, GPUs, and compilers were not audited.

## Claims and evidence (2, 4 and 8 of 9)

2. K and V tiles share one LDS carveout: same `smem_ptr` base, same
   per-buffer stride of 8704 bytes
   (`GetSingleSmemElementSpaceSize` = max(4160, 4352) bf16 elements;
   gfx950 has 64 LDS banks per `arch.hpp`), so K buffer 0 and V
   buffer 0 are the same bytes (observed in source; corroborated:
   every K `ds_read` and V `ds` op in `kernel_shipped_bf16.asm` lands
   in the buffer its `LdsSeq` index predicts).
4. K-read (buf0, last at PC 0x2F68) -> V-write (buf0, first at
   PC 0x3AA4) is ordered (observed): `s_waitcnt lgkmcnt(0)` +
   `s_barrier` at PCs 0x3820/0x3824 sit between them in program order
   on all paths (`out/excerpt_kv_barrier.txt`,
   `out/excerpt_k_tail_reads.txt`). This is the edge with no explicit
   source-level sync; the barrier is compiler-placed, so the ordering
   is real but fragile to scheduling changes (observation, not a
   violation).
8. Adding `block_sync_lds()` after the tail QK GEMM (shadow copy only,
   exact patch in `out/triage/patch.diff`) compiles to exactly one
   added `s_waitcnt lgkmcnt(0)` + `s_barrier` at PC 0x2F94/0x2F98 and
   deletes one now-redundant wait (causal: 7 -> 8 barriers, net +1
   instruction). This proves the shipped binary has no sync at that
   point, and, together with claim 4, that none is required there.
