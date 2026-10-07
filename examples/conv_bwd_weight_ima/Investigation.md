# CUTracer investigates illegal global-memory reads in `triton_convolution2d_bwd_weight` on NVIDIA B200: malformed high-word address arithmetic

**Status:** Confirmed · **Example:** [`mini_repro_bwd_weight.py`](mini_repro_bwd_weight.py)

## TL;DR

- **Symptom:** `triton_convolution2d_bwd_weight` fails during synchronization
  on NVIDIA B200 with `CUDA error: an illegal memory access was encountered`.
- **Conclusion:** With `num_stages=2`, generated code turns a `-4` byte
  contribution in the 32-bit `X` address into high word `3`. Carry propagation
  raises the pointer high word by four. Threads 0–2 issue an
  `X + 16 GiB + low_offset` load. The blind investigation localized the defect
  to stage-2-sensitive lowering or code generation of this address form.
- **Outcome:** Two matched controls avoid the crash and produce the expected
  result. One uses `num_stages=1`. The other keeps `num_stages=2` and casts the
  complete `X` element offset to `tl.int64`. Both match PyTorch with maximum
  absolute error `1.1444091796875e-05`. Compute Sanitizer reports zero errors
  for both. These are controls, not production fixes.
- **CUTracer contribution:** CUTracer identified the exact launch, preserved
  incomplete-capture status, dumped the failing cubin, mapped PC `0x2120` to
  the source `X` load, and captured the registers that explain the 16-GiB
  displacement.

## Case identity

| Field | Value |
|---|---|
| Reproducer | [`mini_repro_bwd_weight.py`](mini_repro_bwd_weight.py) |
| Kernel | `triton_convolution2d_bwd_weight` |
| Platform | NVIDIA B200, compute capability `(10, 0)` / `sm_100` |
| Software | Python 3.13.13; Torch `2.13.0+cu130`; Triton `3.7.1+git5d6048aa`; CUDA 13.0 |
| Recognition signature | Three issued 4-byte reads at PC `0x2120`; addresses are `X + 16 GiB + low_offset`; failing address SASS contains `SHF.L.U64.HI` |
| Discriminating test | Keep `num_stages=2` and cast only the complete `X` element offset to `tl.int64` |
| Validated action | The complete-offset cast and `num_stages=1` pass numerical comparison and Compute Sanitizer; both remain controls |
| Blind conclusion | The immediate mechanism is confirmed; compiler-internal ownership was unresolved before historical evidence was revealed |

## Symptom and reproducer

The failure depends on the Triton build bundled with the pinned PyTorch
nightly. Released Triton 3.7.1 generates different PTX and does not reproduce
this case.

```bash
git clone https://github.com/facebookexperimental/CUTracer.git
cd CUTracer/examples/conv_bwd_weight_ima

uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python \
    torch==2.13.0 numpy \
    --index-url https://download.pytorch.org/whl/cu130
uv pip uninstall --python .venv/bin/python triton
uv pip install --python .venv/bin/python --no-deps \
    'https://github.com/facebookexperimental/CUTracer/releases/download/deps%2Ftriton-3.7.1-git5d6048aa/triton-3.7.1+git5d6048aa-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl'

# Run on a B200 or GB200 GPU.
.venv/bin/python mini_repro_bwd_weight.py
```

Set `CUDA_VISIBLE_DEVICES` only when the default GPU is not a B200 or GB200.

The explicit `torch.cuda.synchronize()` fails with:

```text
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

Matched variants produced these results:

| Variant | Changed variable | Result | Interpretation |
|---|---|---|---|
| Original | None | Illegal memory access | Establishes the symptom |
| `K_TOTAL=640` | Removes the partial eight-lane K tail | Still fails | The partial tail is not necessary |
| `num_stages=1` | Launch parameter only | Pass; max absolute error `1.1444091796875e-05`; memcheck 0 | The failing generated form is not unavoidable |
| `num_stages=3` | Launch parameter only | Pass with the same printed output | The failure is stage-count sensitive; this variant was not separately checked with memcheck |
| Complete `X` offset cast to `tl.int64` | Keeps stage 2; changes only the address-width expression | Pass; max absolute error `1.1444091796875e-05`; memcheck 0 | Bounds the trigger to the original 32-bit complete-offset lowering |
| `INDEX_DTYPE=tl.int64` | Changes the program-ID-related constant only | Still fails | Does not widen the explicitly 32-bit address path |

## Evidence summary

| Claim | State | Producing tool | Evidence | Boundary |
|---|---|---|---|---|
| The original process fails with a CUDA illegal-memory-access error. | observed | PyTorch CUDA runtime | Failure at explicit synchronization | Asynchronous failure alone does not identify the instruction |
| Threads 0–2 issue invalid 4-byte global reads at PC `0x2120`. | observed | Compute Sanitizer | Three invalid-read reports at the source `X` load | Applies to this B200 and software environment |
| The bad addresses contain a valid-looking low offset plus exactly 16 GiB. | observed | Pointer correlation and Compute Sanitizer | First displacement `0x400001bc`; [`example_mem_addr_excerpt.txt`](example_mem_addr_excerpt.txt) | Allocation bases vary; the invariant is the displacement |
| PC `0x2120` is the generated `LDGSTS.E` for the masked `X` load. | observed | CUTracer cubin dump and SASS mapping | [`example_ptx_vs_sass.txt`](example_ptx_vs_sass.txt) | Source mapping does not identify the responsible compiler pass |
| Pre-fault registers contain the malformed high-word inputs and an enabled load predicate. | observed | CUTracer `reg_trace` and SASS | [`example_reg_trace_excerpt.txt`](example_reg_trace_excerpt.txt) | The trace is a crash prefix; the claim uses records captured before the fault |
| The register arithmetic explains the 16-GiB displacement. | causal | Arithmetic over CUTracer registers, corroborated by Compute Sanitizer | Low-word carry plus corrupted high word | Confirms the immediate mechanism, not compiler ownership |
| The original complete 32-bit `X` address form is necessary for this reproduced failure. | causal | Matched `tl.int64` complete-offset control | Stage 2 passes numerics and memcheck after only this change | Applies to the tested case and toolchain |

### Decisive evidence chain

Compute Sanitizer reported an issued read at source line 88 and PC `0x2120`.
The first bad address was exactly `0x400001bc` bytes above the `X` base. This
changed the leading hypothesis from an ordinary bounds error to high-word
address corruption.

CUTracer mapped the sanitizer PC to the source `X` load and this generated
sequence:

```text
/*2020*/ IMAD.SHL.U32 R24, R25.reuse, 0x4, RZ ;
/*2030*/ SHF.L.U64.HI R25, R25, 0x2, RZ ;
/*20e0*/ IADD3 R20, P6, PT, R16, R24, RZ ;
/*20f0*/ IMAD.X R21, R17, 0x1, R25, P6 ;
/*2100*/ ISETP.NE.U32.AND P6, PT, R26, RZ, PT ;
/*2120*/ LDGSTS.E [R73+0x400], desc[UR6][R20.64], P6 ;
```

The final warp-0 register prefix contained `R24=0xfffffffc`, `R25=3`, the
base high word, the carry, and a true predicate source for lanes 0–2. The low
addition produced a carry. The high addition raised the pointer high word by
four. This arithmetic reconstructs the observed 16-GiB displacement.

`mem_addr_trace` can record operands for predicated instructions. Therefore,
the address trace alone does not prove that a load was issued. Compute
Sanitizer confirmed the issued illegal reads. CUTracer register and SASS
evidence confirmed that the load predicate was true for lanes 0–2.

The failing trace was a readable crash prefix with 4,675 records. It had no
EXIT record, and all four warps remained in progress. The conclusions use only
pre-fault records. Missing later records do not prove absence.

## How CUTracer changed the investigation

CUTracer connected the sanitizer fault to one generated instruction and then
to its input registers. The address trace exposed the high displacement. The
register trace explained how the generated arithmetic produced it. This
evidence changed the next test from general bounds checks to a matched
address-width intervention.

## Resolution and validation

The blind investigation validated two discriminating controls:

| Gate | Original | `num_stages=1` | Stage 2 with complete `X` offset cast to `tl.int64` |
|---|---:|---:|---:|
| Program completion | Illegal access | Pass | Pass |
| PyTorch comparison | Not available | Max error `1.1444091796875e-05` | Max error `1.1444091796875e-05` |
| Compute Sanitizer | Invalid reads | 0 errors | 0 errors |

These controls establish the address-lowering boundary for the reproduced
case. They are not production fixes.

## Investigation timeline

Elapsed times are rounded from recorded artifact timestamps. `T+0` is the
initial evidence record. The times identify artifact creation, not GPU runtime.

| Elapsed | Action and evidence | Decision or next action |
|---|---|---|
| T+0 min | Inspected the sanitized reproducer, then reproduced the illegal access. Compute Sanitizer localized three issued reads to PC `0x2120`. | Correlate the addresses with allocation bases. |
| T+1 min | CUTracer identified the launch and reported incomplete-capture status. | Preserve later captures as incomplete crash prefixes. |
| T+2 min | Pointer correlation showed a 16-GiB displacement. CUTracer mapped PC `0x2120` to the `X` load. | Test high-word corruption rather than an ordinary bounds error. |
| T+4 min | `K_TOTAL=640` still failed. Stages 1 and 3 passed. | Reject the partial-tail hypothesis. Focus on a stage-2-sensitive form. |
| **T+6 min — root cause localized** | CUTracer `reg_trace` reconstructed the malformed high word and true load predicate. | Test the complete address width directly. |
| **T+8 min — defect boundary confirmed** | The complete-offset `tl.int64` control passed numerics and Compute Sanitizer while retaining stage 2. | Bound the blind conclusion to lowering of the original 32-bit address form. |
| T+10 min | Trace validation confirmed a structurally valid but incomplete crash prefix. | Retain all trace conclusions as bounded pre-fault claims. |

## Blind investigation and post-investigation validation

The investigation was blind. The investigator received the sanitized kernel
without the known diagnosis or fix. Initial hypotheses were recorded before
execution. Historical conclusions were not imported into the blind phase.

After the blind phase, the checked-in case materials supplied the historical
compiler A/B. The PTX correctly uses the `cp.async` source-size operand to
suppress the masked global read. `ptxas` lowers the operation to a predicated
`LDGSTS` that dereferences the malformed address. Disabling `ptxas`
optimization for the same PTX gives correct numerics and zero sanitizer
errors. [`example_ptx_vs_sass.txt`](example_ptx_vs_sass.txt) preserves the
PTX-to-SASS comparison.

This historical evidence assigns the defect to `ptxas` optimization. It does
not change the narrower claim established by CUTracer during the blind phase:
CUTracer localized the faulting instruction and reconstructed the malformed
runtime address.
