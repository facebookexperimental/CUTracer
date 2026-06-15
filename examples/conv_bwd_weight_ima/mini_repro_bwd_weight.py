"""
Standalone mini-reproducer for pytorch issue #187081:
triton_convolution2d_bwd_weight illegal memory access on B200/GB200 (sm100).

Failing autotune config: BLOCK_M=256, BLOCK_N=16, BLOCK_K=16, num_warps=4,
num_stages=2. Kernel source extracted verbatim from the inductor-generated
kernel (constexpr params baked in). Tensors match the failing test case
(in=3, out=4, groups=1, kernel=1, stride=1, padding=1, dilation=2).

Root cause: a ptxas optimization bug miscompiles the masked cp.async X load
(`matrix_x = tl.load(x_ptrs, mask=mask_x)`); for thread (0,0,0) the predicate
that should mask the load is wrong, so it reads ~17 GB out of bounds.
Plain run: hard CUDA illegal memory access. compute-sanitizer pinpoints
`triton_convolution2d_bwd_weight+0x2120` (SASS `LDGSTS.E ... P6`).
Workaround / confirmation: DISABLE_PTXAS_OPT=1 -> 0 sanitizer errors, correct
numerics. Upstream issue: pytorch/pytorch#187081.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def triton_convolution2d_bwd_weight(arg_X, arg_dY, out_ptr0):
    KERNEL_H: tl.constexpr = 1
    KERNEL_W: tl.constexpr = 1
    PADDING_H: tl.constexpr = 1
    PADDING_W: tl.constexpr = 1
    STRIDE_H: tl.constexpr = 1
    STRIDE_W: tl.constexpr = 1
    DILATION_H: tl.constexpr = 2
    DILATION_W: tl.constexpr = 2
    GROUPS: tl.constexpr = 1
    ALLOW_TF32: tl.constexpr = False
    BLOCK_M: tl.constexpr = 256
    BLOCK_N: tl.constexpr = 16
    BLOCK_K: tl.constexpr = 16
    INDEX_DTYPE: tl.constexpr = tl.int32
    X = arg_X
    dY = arg_dY

    BATCH = 2
    IN_C = 3
    IN_H = 16
    IN_W = 16
    OUT_C = 4
    OUT_H = 18
    OUT_W = 18

    stride_xn = 768
    stride_xc = 256
    stride_xh = 16
    stride_xw = 1

    stride_dyn = 1296
    stride_dyc = 324
    stride_dyh = 18
    stride_dyw = 1

    m0 = tl.program_id(0).to(INDEX_DTYPE) * BLOCK_M
    oc = tl.program_id(1).to(INDEX_DTYPE) * BLOCK_N + tl.arange(0, BLOCK_N)

    group = 0
    GROUP_IN_C = IN_C
    GROUP_OUT_C = OUT_C

    X_base = X + group * GROUP_IN_C * stride_xc
    dY_base = dY + group * GROUP_OUT_C * stride_dyc

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    K_TOTAL = BATCH * OUT_H * OUT_W
    for k in range(0, K_TOTAL, BLOCK_K):
        idx_k = k + tl.arange(0, BLOCK_K)
        n = (idx_k // (OUT_H * OUT_W)).to(tl.int32)
        rem = idx_k % (OUT_H * OUT_W)
        yh = (rem // OUT_W).to(tl.int32)
        yw = (rem % OUT_W).to(tl.int32)

        m = (m0 + tl.arange(0, BLOCK_M)).to(tl.int32)
        ic = (m // (KERNEL_H * KERNEL_W)).to(tl.int32)
        ij = m % (KERNEL_H * KERNEL_W)
        i = (ij // KERNEL_W).to(tl.int32)
        j = (ij % KERNEL_W).to(tl.int32)

        xh = yh[:, None] * STRIDE_H - PADDING_H + i[None, :] * DILATION_H
        xw = yw[:, None] * STRIDE_W - PADDING_W + j[None, :] * DILATION_W

        x_ptrs = (
            X_base
            + n[:, None] * stride_xn
            + ic[None, :] * stride_xc
            + xh * stride_xh
            + xw * stride_xw
        )

        mask_x = (
            (n[:, None] < BATCH)
            & (ic[None, :] < GROUP_IN_C)
            & (xh >= 0)
            & (xh < IN_H)
            & (xw >= 0)
            & (xw < IN_W)
        )
        matrix_x = tl.load(x_ptrs, mask=mask_x, other=0.0)
        matrix_x = tl.trans(matrix_x)

        dy_ptrs = (
            dY_base
            + n[:, None] * stride_dyn
            + oc[None, :] * stride_dyc
            + yh[:, None] * stride_dyh
            + yw[:, None] * stride_dyw
        )

        mask_dy = (
            (n[:, None] < BATCH)
            & (oc[None, :] < GROUP_OUT_C)
            & (yh[:, None] < OUT_H)
            & (yw[:, None] < OUT_W)
        )
        matrix_dy = tl.load(dy_ptrs, mask=mask_dy, other=0.0)

        acc += tl.dot(matrix_x, matrix_dy, allow_tf32=ALLOW_TF32)

    m = m0 + tl.arange(0, BLOCK_M)
    ic = m // (KERNEL_H * KERNEL_W)
    ij = m % (KERNEL_H * KERNEL_W)
    i = ij // KERNEL_W
    j = ij % KERNEL_W

    out_ic = ic
    out_oc = oc + group * GROUP_OUT_C

    mask = (out_ic[:, None] < GROUP_IN_C) & (oc[None, :] < GROUP_OUT_C)

    tl.store(
        out_ptr0
        + (
            tl.broadcast_to(
                (out_ic[:, None]) + 3 * (out_oc[None, :]), [BLOCK_M, BLOCK_N]
            )
        ),
        acc,
        mask,
    )


def main():
    torch.manual_seed(0)
    dev = "cuda"
    X = torch.randn(2, 3, 16, 16, device=dev)  # input
    dY = torch.randn(2, 4, 18, 18, device=dev)  # grad_output
    out = torch.zeros(4, 3, 1, 1, device=dev)  # grad_weight (OUT_C, IN_C, 1, 1)

    grid = (1, 1, 1)  # ceil(3/512), ceil(4/16), GROUPS
    triton_convolution2d_bwd_weight[grid](X, dY, out, num_warps=4, num_stages=2)
    torch.cuda.synchronize()
    print("OK", out.flatten().tolist())


if __name__ == "__main__":
    main()
