# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Queues a short Triton kernel behind one that spins for QUEUED_BEHIND_S seconds.

Under --nonblocking-launches the short kernel's launch returns at once while the
kernel itself waits in the stream, longer than CUTracer's default no-data
timeout. Prints DONE when both kernels finished.
"""

import os

import torch
import triton
import triton.language as tl


@triton.jit
def add_one(x_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    tl.store(x_ptr + offsets, tl.load(x_ptr + offsets, mask=mask) + 1, mask=mask)


def cycles_per_second() -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    torch.cuda._sleep(100_000_000)
    end.record()
    end.synchronize()
    return 100_000_000 / (start.elapsed_time(end) / 1000)


def main() -> None:
    n = 1024
    x = torch.zeros(n, device="cuda")
    # Compile and load before the long kernel.
    add_one[(1,)](x, n, BLOCK=n)
    torch.cuda.synchronize()
    seconds = float(os.environ.get("QUEUED_BEHIND_S", "20"))
    torch.cuda._sleep(int(seconds * cycles_per_second()))
    add_one[(1,)](x, n, BLOCK=n)
    torch.cuda.synchronize()
    assert torch.all(x == 2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
