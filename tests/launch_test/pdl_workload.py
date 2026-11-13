# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Launches one kernel with programmatic dependent launch (PDL) and once without.

The PDL launch passes CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION, which
CUTracer records in its kernel events and trace metadata. With `--graph`, the PDL
launch is instead captured into a CUDA graph and replayed. Needs sm_90 or newer.
"""

import sys

import torch
import triton
import triton.language as tl


@triton.jit
def scale_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr, USE_GDC: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    if USE_GDC:
        tl.extra.cuda.gdc_wait()
    x = tl.load(x_ptr + offsets, mask=mask)
    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(out_ptr + offsets, x * 2, mask=mask)


def main() -> None:
    n = 4096
    x = torch.arange(n, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    grid = (triton.cdiv(n, 1024),)
    if "--graph" in sys.argv:
        # Compile outside the capture.
        scale_kernel[grid](x, out, n, BLOCK=1024, USE_GDC=True, launch_pdl=True)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            scale_kernel[grid](x, out, n, BLOCK=1024, USE_GDC=True, launch_pdl=True)
        graph.replay()
    else:
        scale_kernel[grid](x, out, n, BLOCK=1024, USE_GDC=True, launch_pdl=True)
        scale_kernel[grid](x, out, n, BLOCK=1024, USE_GDC=False, launch_pdl=False)
    torch.cuda.synchronize()
    assert torch.equal(out, x * 2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
