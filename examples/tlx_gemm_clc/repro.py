#!/usr/bin/env python3

from __future__ import annotations

import argparse

import torch
from triton.language.extra.tlx.tutorials.blackwell_gemm_clc import matmul

CONFIG = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 256,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 8,
    "NUM_SMEM_BUFFERS": 2,
    "NUM_TMEM_BUFFERS": 2,
    "EPILOGUE_SUBTILE": True,
}
RTOL = 1e-2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    a = torch.randn((args.m, args.k), device="cuda", dtype=torch.float16)
    b = torch.randn((args.k, args.n), device="cuda", dtype=torch.float16)
    ref = torch.matmul(a.float(), b.float())
    # Inputs are O(1), so outputs are O(sqrt(K)). The absolute floor scales
    # with the output so that a zeroed, stale, or misplaced tile cannot pass.
    atol = 1e-2 * ref.abs().mean().item()
    limit = atol + RTOL * ref.abs()

    print(
        f"GPU={torch.cuda.get_device_name()} M={args.m} N={args.n} K={args.k} "
        f"iterations={args.iterations} atol={atol:.4g} rtol={RTOL}"
    )
    failures = 0
    for i in range(args.iterations):
        c = matmul(a, b, config=CONFIG)
        torch.cuda.synchronize()
        diff = (c.float() - ref).abs()
        # NaN compares false, so count elements that are not within the limit.
        bad = int((~(diff <= limit)).sum().item())
        if bad:
            failures += 1
            print(
                f"iteration {i}: MISMATCH elements={bad} "
                f"max_abs={diff.nan_to_num(float('inf')).max().item():.6g}"
            )

    print(f"{failures}/{args.iterations} iterations mismatched")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
