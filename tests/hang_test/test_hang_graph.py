# Copyright (c) Meta Platforms, Inc. and affiliates.
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    if pid == 0:
        while pid == pid:
            tl.atomic_add(a_ptr, 1)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)


def main():
    size = (1024, 1024)
    a = torch.randn(size, device="cuda", dtype=torch.float32)
    b = torch.randn(size, device="cuda", dtype=torch.float32)
    c = torch.empty_like(a)
    n_elements = a.numel()
    grid = (triton.cdiv(n_elements, 1024),)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        add_kernel[grid](a, b, c, n_elements, 1024)
    graph.replay()
    torch.cuda.synchronize()
    print("Graph kernel executed successfully")


if __name__ == "__main__":
    main()
