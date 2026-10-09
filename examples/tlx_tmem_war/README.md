# TLX 2-CTA attention-backward TMEM WAR reproducer

See [`Investigation.md`](Investigation.md) for the complete blind CUTracer case
report and matched validation.

This example reproduces numeric corruption in a TLX 2-CTA attention-backward
kernel. The kernel lets eight compute warps reuse aliased TMEM without a
task-wide barrier. Warps `N` and `N+4` share a TMEM lane group. A store can
therefore overwrite a value before its paired warp completes the read.

The affected kernel is pinned to
[`facebookexperimental/triton@d4e09c9e`](https://github.com/facebookexperimental/triton/commit/d4e09c9e8c9c1ba3b1dbc974756b32d613e94942).
The complete upstream fix is
[`facebookexperimental/triton#1992`](https://github.com/facebookexperimental/triton/pull/1992).
The included `fix.patch` contains only its two task-wide barrier changes. It is
a matched control, not a complete backport of the upstream fix.

## Requirements

- NVIDIA B200 GPU
- The `fbtriton` PyPI wheel, which includes TLX
- PyTorch with CUDA support

The defect is in the bundled kernel source. It is not specific to one
`fbtriton` release. The setup below installs the current wheel. Verification
also succeeded with `fbtriton==3.7.4`.

## Reproduce

Create an isolated environment. Install `fbtriton` after PyTorch because the
PyTorch wheel installs the upstream `triton` distribution:

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python \
  torch numpy \
  --index-url https://download.pytorch.org/whl/cu130
uv pip uninstall --python .venv/bin/python triton
uv pip install --python .venv/bin/python fbtriton
```

Run from this example directory:

```bash
.venv/bin/python repro.py --iterations 30
```

The affected source usually fails quickly. The first verification run failed
on iteration 2:

```text
iteration 1: pass
RACE REPRODUCED on iteration 2: dq: mismatches=1393, max_abs=0.135401, dk: mismatches=1453, max_abs=0.159302, dv: mismatches=1038, max_abs=0.0756836
```

Exit code `1` means that corruption was detected. Exit code `0` means that all
requested iterations passed. More iterations increase confidence but do not
prove that a probabilistic race is absent.

## Run the matched barrier control

Apply the included patch to the bundled source:

```bash
patch -p1 < fix.patch
.venv/bin/python repro.py --iterations 100
```

The patch changes only the synchronization under test. It inserts task-wide
barriers before the two stores that reuse aliased TMEM. It omits the other
lifetime-order changes from the complete upstream fix. The confirmed control
completed 100 iterations:

```text
RACE NOT OBSERVED in 100 iterations
```
