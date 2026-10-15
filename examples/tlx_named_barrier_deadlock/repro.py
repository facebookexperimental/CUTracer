#!/usr/bin/env python3

import argparse
import multiprocessing
import queue


def _run_kernel(result_queue: multiprocessing.Queue) -> None:
    try:
        import torch
        import triton
        import triton.language as tl
        import triton.language.extra.tlx as tlx

        # Reduced from Triton's `_run_kernel_diverge_both_1warp` regression,
        # added in commit a6ffe3ce.
        @triton.jit
        def named_barrier_divergence(output_ptr):
            with tlx.async_tasks():
                with tlx.async_task(num_warps=1):
                    if tlx.thread_id(axis=0) % 32 == 0:
                        tl.store(output_ptr + 1, 99)
                    tlx.named_barrier_wait(14, 32)
                    if tlx.thread_id(axis=0) % 32 == 0:
                        tl.store(output_ptr, 5)
                with tlx.async_task("default"):
                    pass

        output = torch.zeros(2, dtype=torch.int32, device="cuda")
        named_barrier_divergence[(1,)](output, num_warps=4)
        torch.cuda.synchronize()
        result_queue.put(("complete", output.cpu().tolist()))
    except Exception as error:
        result_queue.put(("error", repr(error)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(target=_run_kernel, args=(result_queue,))
    process.start()
    process.join(timeout=args.timeout)

    if process.is_alive():
        process.kill()
        process.join(timeout=5)
        print(f"HANG REPRODUCED: kernel exceeded {args.timeout:g} seconds")
        return 124

    try:
        status, detail = result_queue.get_nowait()
    except queue.Empty:
        print(f"ERROR: child exited with status {process.exitcode} and no result")
        return 1

    if status == "error":
        print(f"ERROR: {detail}")
        return 1
    if detail != [5, 99]:
        print(f"ERROR: expected [5, 99], got {detail}")
        return 1

    print(f"PASS: kernel completed with output {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
