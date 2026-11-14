# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Two kernels that finish their job only if they run at the same time.

`wait_for_flag` polls a flag that `set_flag`, launched after it on another
stream, sets. If the second launch cannot start while the first kernel runs,
the poll gives up after a bounded number of tries. The script prints
OVERLAPPED or GAVE_UP; a tool that blocks the launching thread until each
kernel finishes, as CUTracer does by default, turns OVERLAPPED into GAVE_UP.
"""

import os

import torch
import triton
import triton.language as tl

# Polls while instrumented run far slower, so a test that instruments the
# polling kernel lowers this to keep its trace small.
MAX_POLLS = int(os.environ.get("OVERLAP_MAX_POLLS", "1000000"))


@triton.jit
def wait_for_flag(flag_ptr, out_ptr, max_polls):
    seen = tl.atomic_add(flag_ptr, 0)
    polls = 0
    while (seen == 0) & (polls < max_polls):
        seen = tl.atomic_add(flag_ptr, 0)
        polls += 1
    tl.store(out_ptr, seen)


@triton.jit
def set_flag(flag_ptr):
    tl.atomic_xchg(flag_ptr, 1)


def main() -> None:
    flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    waiter_stream = torch.cuda.Stream()
    setter_stream = torch.cuda.Stream()

    # Compile and launch both kernels once with the same arguments, so the
    # measured launches below change no kernel's code. The flag is already
    # set, so this wait returns at once.
    set_flag[(1,)](flag)
    wait_for_flag[(1,)](flag, out, MAX_POLLS)
    torch.cuda.synchronize()
    flag.zero_()
    out.zero_()
    torch.cuda.synchronize()

    with torch.cuda.stream(waiter_stream):
        wait_for_flag[(1,)](flag, out, MAX_POLLS)
    with torch.cuda.stream(setter_stream):
        set_flag[(1,)](flag)
    torch.cuda.synchronize()
    print("OVERLAPPED" if out.item() else "GAVE_UP", flush=True)


if __name__ == "__main__":
    main()
