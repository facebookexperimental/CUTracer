#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType

import torch
import triton
from triton.tools.tensor_descriptor import TensorDescriptor

SHAPE = (4, 8, 1024, 128)
DEFAULT_KERNEL_SOURCE = (
    Path(__file__).resolve().parent
    / "kernel"
    / "blackwell_fa_ws_pipelined_persistent.py"
)


def load_kernel(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("tlx_tmem_war_kernel", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load kernel source from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def descriptor(tensor: torch.Tensor) -> TensorDescriptor:
    z, h, n_ctx, head_dim = tensor.shape
    return TensorDescriptor(
        tensor,
        shape=[z, h, n_ctx, head_dim],
        strides=[h * n_ctx * head_dim, n_ctx * head_dim, head_dim, 1],
        block_shape=[1, 1, 1, 1],
    )


def make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(20)
    tensors = [
        torch.empty(SHAPE, device="cuda", dtype=torch.float16)
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
        for _ in range(3)
    ]
    return tensors[0], tensors[1], tensors[2]


def prepare_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    output = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, scale=scale, is_causal=False
    )
    torch.manual_seed(21)
    grad_output = torch.randn_like(output)
    output.backward(grad_output)
    return grad_output, q.grad.clone(), k.grad.clone(), v.grad.clone()


def run_forward(
    kernel: ModuleType,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    z, h, n_ctx, head_dim = SHAPE
    output = torch.empty_like(q)
    softmax_lse = torch.empty((z, h, n_ctx), device="cuda", dtype=torch.float32)
    flat_rows = z * h * n_ctx
    block_shape = [1, 1]
    desc_q = TensorDescriptor(q, [flat_rows, head_dim], [head_dim, 1], block_shape)
    desc_k = TensorDescriptor(k, [flat_rows, head_dim], [head_dim, 1], block_shape)
    desc_v = TensorDescriptor(v, [flat_rows, head_dim], [head_dim, 1], block_shape)
    desc_o = TensorDescriptor(output, [flat_rows, head_dim], [head_dim, 1], block_shape)
    config = {
        "BLOCK_M": 256,
        "BLOCK_N": 128,
        "NUM_BUFFERS_Q": 1,
        "NUM_BUFFERS_KV": 3,
        "NUM_BUFFERS_QK": 1,
        "NUM_MMA_GROUPS": 2,
        "NUM_MMA_SLICES": 2,
        "GROUP_SIZE_N": 1,
        "USE_WARP_BARRIER": True,
        "RESCALE_OPT": False,
        "USE_WHERE": False,
    }
    kernel._host_descriptor_pre_hook(
        {
            **config,
            "HEAD_DIM": head_dim,
            "desc_q": desc_q,
            "desc_k": desc_k,
            "desc_v": desc_v,
            "desc_o": desc_o,
        }
    )
    grid = (triton.cdiv(n_ctx, config["BLOCK_M"]) * z * h, 1, 1)
    kernel._attn_fwd_ws.fn[grid](
        scale,
        softmax_lse,
        z,
        h,
        desc_q,
        desc_k,
        desc_v,
        desc_o,
        N_CTX=n_ctx,
        HEAD_DIM=head_dim,
        STAGE=1,
        num_stages=1,
        **config,
    )
    return output, softmax_lse


def run_backward(
    kernel: ModuleType,
    q: torch.Tensor,
    scaled_k: torch.Tensor,
    v: torch.Tensor,
    grad_output: torch.Tensor,
    softmax_lse: torch.Tensor,
    delta: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z, h, n_ctx, head_dim = SHAPE
    dq = torch.zeros(SHAPE, device="cuda", dtype=torch.float32)
    dk = torch.empty_like(scaled_k)
    dv = torch.empty_like(v)
    desc_q = descriptor(q)
    desc_k = descriptor(scaled_k)
    desc_v = descriptor(v)
    desc_do = descriptor(grad_output)
    desc_dq = descriptor(dq)
    desc_dk = descriptor(dk)
    desc_dv = descriptor(dv)
    desc_kt = descriptor(scaled_k)
    desc_qt = descriptor(q)
    desc_dot = descriptor(grad_output)
    desc_m = TensorDescriptor(
        softmax_lse,
        shape=[z * h * n_ctx],
        strides=[1],
        block_shape=[1],
    )
    desc_delta = TensorDescriptor(
        delta,
        shape=[z * h * n_ctx],
        strides=[1],
        block_shape=[1],
    )
    config = kernel.configs_bwd_2cta[0]
    launch_args = {
        **config.kwargs,
        "HEAD_DIM": head_dim,
        "desc_q": desc_q,
        "desc_k": desc_k,
        "desc_v": desc_v,
        "desc_do": desc_do,
        "desc_dq": desc_dq,
        "desc_dk": desc_dk,
        "desc_dv": desc_dv,
        "desc_m": desc_m,
        "desc_delta": desc_delta,
        "desc_kt": desc_kt,
        "desc_qt": desc_qt,
        "desc_dot": desc_dot,
    }
    kernel._bwd_host_descriptor_pre_hook_tlx(launch_args)
    grid = (triton.cdiv(n_ctx, config.kwargs["BLOCK_N1"]), h, z)
    kernel._attn_bwd_ws.fn[grid](
        desc_q,
        desc_k,
        desc_v,
        scale,
        desc_do,
        desc_dq,
        desc_dk,
        desc_dv,
        desc_m,
        desc_delta,
        softmax_lse,
        delta,
        h,
        z,
        n_ctx,
        desc_kt,
        desc_qt,
        desc_dot,
        BLK_SLICE_FACTOR=2,
        HEAD_DIM=head_dim,
        STAGE=1,
        num_warps=8,
        num_stages=1,
        ctas_per_cga=(2, 1, 1),
        **config.kwargs,
    )
    torch.cuda.synchronize()
    return dq, dk, dv


def mismatch(actual: torch.Tensor, expected: torch.Tensor) -> tuple[int, float]:
    expected = expected.to(actual.dtype)
    close = torch.isclose(actual, expected, atol=1e-2, rtol=0)
    count = int((~close).sum().item())
    maximum = float((actual - expected).abs().max().item())
    return count, maximum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the TLX 2-CTA attention-backward TMEM WAR."
    )
    parser.add_argument(
        "kernel_source",
        nargs="?",
        type=Path,
        default=DEFAULT_KERNEL_SOURCE,
        help=(
            "Kernel source to load. Defaults to the affected d4e09c9e source "
            "bundled with this example."
        ),
    )
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be greater than zero")
    return args


def main() -> int:
    args = parse_args()
    kernel_source = args.kernel_source.resolve()
    if not kernel_source.is_file():
        raise FileNotFoundError(f"kernel source does not exist: {kernel_source}")
    kernel = load_kernel(kernel_source)
    triton.set_allocator(
        lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda")
    )
    scale = 0.5
    q, k, v = make_inputs()
    grad_output, ref_dq, ref_dk, ref_dv = prepare_reference(q, k, v, scale)
    output, softmax_lse = run_forward(kernel, q, k, v, scale)
    delta = torch.empty_like(softmax_lse)
    kernel._attn_bwd_preprocess[(SHAPE[2] // 128, SHAPE[0] * SHAPE[1])](
        output, grad_output, delta, SHAPE[2], BLOCK_M=128, HEAD_DIM=SHAPE[3]
    )
    scaled_k = k * (scale * 1.4426950408889634)

    for iteration in range(1, args.iterations + 1):
        dq, dk, dv = run_backward(
            kernel,
            q,
            scaled_k,
            v,
            grad_output,
            softmax_lse,
            delta,
            scale,
        )
        errors = {
            "dq": mismatch(dq, ref_dq),
            "dk": mismatch(dk, ref_dk),
            "dv": mismatch(dv, ref_dv),
        }
        if any(count for count, _ in errors.values()):
            details = ", ".join(
                f"{name}: mismatches={count}, max_abs={maximum:.6g}"
                for name, (count, maximum) in errors.items()
            )
            print(f"RACE REPRODUCED on iteration {iteration}: {details}")
            return 1
        print(f"iteration {iteration}: pass")

    print(f"RACE NOT OBSERVED in {args.iterations} iterations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
