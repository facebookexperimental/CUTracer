# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Workloads that test_runtime_identity_e2e runs under CUTracer.

``graph`` launches a CUDA-runtime kernel directly and from a CUDA graph replay.
``reload ITERATIONS`` alternately loads two PTX modules with the same entry
name, launches the entry once and unloads the module right after the launch
returns, so the driver can reuse CUfunction handles across modules.
"""

import ctypes
import sys

# On A100, later replays of an instantiated graph keep the first replay's NVBit
# launch argument, so their records would carry its launch ID.
GRAPH_REPLAYS = 1

PTX = """.version 7.0
.target sm_70
.address_size 64

.visible .entry reload_probe(.param .u64 out, .param .u32 n)
{{
  .reg .pred %p;
  .reg .b32 %r<7>;
  .reg .b64 %rd<4>;
  ld.param.u64 %rd1, [out];
  ld.param.u32 %r1, [n];
  mov.u32 %r2, %ctaid.x;
  mov.u32 %r3, %ntid.x;
  mov.u32 %r4, %tid.x;
  mad.lo.s32 %r5, %r2, %r3, %r4;
  setp.ge.s32 %p, %r5, %r1;
  @%p bra DONE;
  {value}
  cvta.to.global.u64 %rd2, %rd1;
  mul.wide.s32 %rd3, %r5, 4;
  add.s64 %rd2, %rd2, %rd3;
  st.global.u32 [%rd2], %r6;
DONE:
  ret;
}}
"""

VARIANTS = (
    PTX.format(value="mov.u32 %r6, %r5;").encode(),
    PTX.format(value="add.s32 %r6, %r5, 17;").encode(),
)


def graph() -> None:
    import torch

    x = torch.zeros(1 << 16, device="cuda")
    for _ in range(2):
        x.add_(1)
    replay = torch.cuda.CUDAGraph()
    with torch.cuda.graph(replay):
        x.add_(1)
    for _ in range(GRAPH_REPLAYS):
        replay.replay()
    torch.cuda.synchronize()
    if x[0].item() != 2 + GRAPH_REPLAYS:
        raise RuntimeError(f"unexpected value {x[0].item()}")


def _check(status: int, operation: str) -> None:
    if status != 0:
        raise RuntimeError(f"{operation} failed with CUresult {status}")


def reload(iterations: int) -> None:
    cuda = ctypes.CDLL("libcuda.so.1")
    _check(cuda.cuInit(0), "cuInit")
    device = ctypes.c_int()
    _check(cuda.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")
    context = ctypes.c_void_p()
    _check(cuda.cuDevicePrimaryCtxRetain(ctypes.byref(context), device), "retain")
    _check(cuda.cuCtxSetCurrent(context), "cuCtxSetCurrent")
    count = ctypes.c_uint32(4096)
    output = ctypes.c_uint64()
    _check(cuda.cuMemAlloc_v2(ctypes.byref(output), count.value * 4), "cuMemAlloc")
    arguments = (ctypes.c_void_p * 2)(
        ctypes.cast(ctypes.byref(output), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(count), ctypes.c_void_p),
    )
    for iteration in range(iterations):
        module = ctypes.c_void_p()
        image = VARIANTS[iteration % 2]
        _check(cuda.cuModuleLoadData(ctypes.byref(module), image), "cuModuleLoadData")
        function = ctypes.c_void_p()
        _check(
            cuda.cuModuleGetFunction(ctypes.byref(function), module, b"reload_probe"),
            "cuModuleGetFunction",
        )
        _check(
            cuda.cuLaunchKernel(
                function, count.value // 128, 1, 1, 128, 1, 1, 0, None, arguments, None
            ),
            "cuLaunchKernel",
        )
        _check(cuda.cuModuleUnload(module), "cuModuleUnload")
    _check(cuda.cuCtxSynchronize(), "cuCtxSynchronize")
    _check(cuda.cuMemFree_v2(output), "cuMemFree")
    _check(cuda.cuDevicePrimaryCtxRelease_v2(device), "release")


if __name__ == "__main__":
    if sys.argv[1] == "graph":
        graph()
    else:
        reload(int(sys.argv[2]))
