# Example: Root-causing an illegal memory access with CUTracer

This example shows how CUTracer was used to localize a GPU illegal memory
access (IMA) down to a single SASS instruction and trace the bad address back
to its source registers -- pinning the bug on `ptxas` rather than Triton or
LLVM.

Upstream issue: pytorch/pytorch#187081.

## Background

A TorchInductor-generated Triton kernel, `triton_convolution2d_bwd_weight`,
reads ~17 GB out of bounds on Blackwell (sm100, GB200/B200), crashing a conv
backward test. `mini_repro_bwd_weight.py` is the distilled standalone kernel
(torch + triton only) for the single failing autotune config
(`BLOCK_M=256, BLOCK_N=16, BLOCK_K=16, num_warps=4, num_stages=2`).

A subtlety that makes this a good CUTracer example: CUTracer (NVBit) forces
`CUDA_MANAGED_FORCE_DEVICE_ALLOC=1`, so the OOB address lands in a mapped
managed region and the process does **not** hard-crash under CUTracer. The
buggy load still executes, so `mem_addr_trace` and `reg_trace` still capture
the out-of-bounds address and its provenance -- which is exactly how the bug
was localized without needing the crash.

## Environment

The bug is version-specific (it lives in `ptxas`), so the exact toolchain
matters when reproducing or reporting it.

| Component | Version |
|-----------|---------|
| GPU | NVIDIA GB200 (Blackwell, sm100, compute cap 10.0) |
| NVIDIA driver | 580.82.07 |
| `ptxas` (CUDA toolkit) | release 13.0, V13.0.88 |
| `cuobjdump` / `nvdisasm` | CUDA 13.0 |
| PyTorch | 2.13.0.dev20260611+cu130 (nightly) |
| Triton | 3.7.1 |
| Python | 3.14.4 |
| `compute-sanitizer` | CUDA 13.0 |

## Source kernel

`mini_repro_bwd_weight.py` -- the masked `X` load
`matrix_x = tl.load(x_ptrs, mask=mask_x)` faults at SASS offset `+0x2120`.

| Parameter   | Value |
|-------------|-------|
| BLOCK_M     | 256   |
| BLOCK_N     | 16    |
| BLOCK_K     | 16    |
| num_warps   | 4     |
| num_stages  | 2     |
| Grid        | 1,1,1 |
| Block       | 128,1,1 |

## Root cause

PTX masks the load correctly via the `cp.async` `src-size` operand (`= 0` on
the masked path), which per the PTX ISA must not access the source address.
`ptxas` lowers it to a predicated `LDGSTS` that *does* dereference the address,
and miscomputes the 64-bit address high word, so a should-be-masked lane reads
out of bounds. Confirmed `ptxas`: `DISABLE_PTXAS_OPT=1` on the same PTX gives
0 sanitizer errors and correct numerics. See `example_ptx_vs_sass.txt`.

## How to reproduce

Plain run (no CUTracer) -- hard crash:

```bash
python mini_repro_bwd_weight.py
# RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
```

Confirm it is a ptxas optimizer bug (same PTX, optimizer off -> clean):

```bash
DISABLE_PTXAS_OPT=1 compute-sanitizer --tool memcheck python mini_repro_bwd_weight.py
# ERROR SUMMARY: 0 errors
```

Capture the per-lane OOB addresses at the faulting load with CUTracer
(`mem_addr_trace`):

```bash
buck2 run //triton/tools/CUTracer:cutracer -c fbcode.nvcc_arch=b200a -- trace \
    --instrument mem_addr_trace \
    --kernel-filters triton_convolution2d_bwd_weight \
    --output-dir /tmp/ct_mem \
    -- python mini_repro_bwd_weight.py
```

Trace the bad address back to its source registers (`reg_trace`):

```bash
buck2 run //triton/tools/CUTracer:cutracer -c fbcode.nvcc_arch=b200a -- trace \
    --instrument reg_trace \
    --kernel-filters triton_convolution2d_bwd_weight \
    --output-dir /tmp/ct_reg \
    -- python mini_repro_bwd_weight.py
```

## Files

| File | Description |
|------|-------------|
| `mini_repro_bwd_weight.py` | Standalone reproducer (torch + triton only). |
| `example_mem_addr_excerpt.txt` | `mem_addr_trace` excerpt: the 32 lane addresses at `pc=0x2120` for thread (0,0,0), showing the ~17 GB OOB read. |
| `example_reg_trace_excerpt.txt` | `reg_trace` excerpt: register-by-register provenance of the bad address (negative halo index `R35=-16`, corrupted high word `R21=0xfffa`). |
| `example_ptx_vs_sass.txt` | The PTX `cp.async` masking (`src-size=0`) vs the miscompiled SASS `LDGSTS`. |
