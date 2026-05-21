# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
FP8 (e4m3) GEMM driver for the Blackwell TLX warp-specialized GEMM kernel.

This script exists primarily as a *real-data SASS source* for the CUTracer
``reg_trace`` instrumentation path. The bundled
``blackwell_gemm_ws.matmul_kernel_tma_ws_blackwell`` kernel emits ``UTCQMMA``
SASS on Blackwell (sm100) when its operand descriptors are FP8 e4m3, exercising
the non-HMMA branch of CUTracer's ``collect_utc_mma_implicit_regs`` path
introduced in D105661162.

The script itself does not invoke CUTracer; it simply runs the kernel and
verifies the results are within FP8 tolerance against an FP16 reference. The
e2e wrapper (``test_blackwell_fp8_gemm_e2e.py``) confirms this runs cleanly on
Blackwell hardware. To capture a UTCQMMA trace, wrap the invocation with
``cutracer trace -i reg_trace -k matmul_kernel_tma_ws_blackwell -- python <this
script>``.

Notes:
  * PyTorch on sm100 (as bundled in fbsource at the time of writing) does not
    expose FP8 quantization/transpose CUDA kernels. We therefore construct the
    FP8 tensors on CPU and move them to GPU rather than using ``.to(fp8)`` on
    a CUDA tensor.
  * The kernel from ``triton.language.extra.tlx.tutorials.blackwell_gemm_ws``
    uses hand-written TLX warp specialization (``tlx.async_tasks`` /
    ``tlx.async_dot``) which compiles successfully on the Triton 3.5.0+fb
    bundled in devgpu binaries, sidestepping the sm_100a / Triton 3.6+
    requirement that blocks automatic ``warp_specialize=True`` on FP8.
"""

import click
import torch
import triton
from triton.language.extra.tlx.tutorials.blackwell_gemm_ws import (
    matmul as blackwell_gemm_ws_matmul,
)

_BLACKWELL_MAJOR = 10  # sm100 = Blackwell

# Standard config from third_party/tlx/tutorials/testing/test_correctness.py
_GEMM_CONFIG = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 256,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 8,
    "NUM_SMEM_BUFFERS": 2,
    "NUM_TMEM_BUFFERS": 2,
    "NUM_MMA_GROUPS": 1,
    "EPILOGUE_SUBTILE": 1,
    "NUM_CTAS": 1,
    "SPLIT_K": 1,
}


def _make_fp8_inputs(M, N, K, seed):
    """Build FP8 e4m3 (a, b) on CPU and copy to CUDA.

    Quantization-on-GPU is unavailable in the bundled torch build, so the
    cast must happen on CPU. We seed the per-call generator for determinism
    across runs.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    a_fp16 = torch.randn((M, K), dtype=torch.float16, generator=g) * 0.1
    b_fp16 = torch.randn((K, N), dtype=torch.float16, generator=g) * 0.1
    a_fp8 = a_fp16.to(torch.float8_e4m3fn).cuda()
    b_fp8 = b_fp16.to(torch.float8_e4m3fn).cuda()
    return a_fp8, b_fp8


def _reference(a_fp8, b_fp8):
    """FP16 reference: dequantize and matmul on GPU (matmul on fp8 is unsupported)."""
    a_ref = a_fp8.to(torch.float16)
    b_ref = b_fp8.to(torch.float16)
    return torch.matmul(a_ref, b_ref)


@click.command()
@click.option(
    "--iters",
    "-i",
    default=5,
    type=int,
    help="Number of iterations to run (default: 5)",
)
@click.option(
    "--m", "M", default=1024, type=int, help="Matrix M dimension (default: 1024)"
)
@click.option(
    "--n", "N", default=1024, type=int, help="Matrix N dimension (default: 1024)"
)
@click.option(
    "--k", "K", default=1024, type=int, help="Matrix K dimension (default: 1024)"
)
@click.option(
    "--atol",
    default=0.5,
    type=float,
    help="Absolute tolerance for FP8 GEMM correctness (default: 0.5)",
)
def main(iters, M, N, K, atol):
    if not torch.cuda.is_available():
        print("Skipping: CUDA is not available")
        return
    major, _ = torch.cuda.get_device_capability()
    if major != _BLACKWELL_MAJOR:
        print(
            f"Skipping: requires Blackwell GPU (sm{_BLACKWELL_MAJOR}0), got sm{major}0"
        )
        return

    # The kernel allocates output as `dtype=a.dtype` (FP8), which `torch.matmul`
    # in the reference cannot consume. We compare against an FP16 dequantized
    # reference, then cast the FP8 output back to FP16 for the diff.
    print("=" * 60)
    print("Blackwell FP8 (e4m3) GEMM driver — UTCQMMA real-data trace source")
    print("=" * 60)
    print(f"Shape: M={M} N={N} K={K}, iters={iters}, atol={atol}")
    print("Kernel: matmul (from triton.language.extra.tlx.tutorials.blackwell_gemm_ws)")
    print()

    num_passed = 0
    num_failed = 0
    for i in range(iters):
        a, b = _make_fp8_inputs(M, N, K, seed=20 + i)
        ref = _reference(a, b)
        try:
            c_fp8 = blackwell_gemm_ws_matmul(a, b, config=dict(_GEMM_CONFIG))
        except Exception as e:
            num_failed += 1
            print(f"Run {i}: EXCEPTION {type(e).__name__}: {e}")
            continue

        c = c_fp8.to(torch.float16)
        max_diff = (c - ref).abs().max().item()
        if max_diff > atol:
            num_failed += 1
            print(f"Run {i}: FAILED, max_diff={max_diff:.4f} (atol={atol})")
        else:
            num_passed += 1
            print(f"Run {i}: PASSED, max_diff={max_diff:.4f}")

    print()
    print("=" * 60)
    print(f"Results: {num_passed} passed, {num_failed} failed out of {iters} runs")
    print("=" * 60)
    if num_failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
