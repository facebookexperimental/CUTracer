# TLX CLC GEMM: missing async-proxy fence after the CLC response read

See [`Investigation.md`](Investigation.md) for the case report. It covers three
blind agent audits with the CUTracer `audit-memory-model` skill, and the
matched validation.

The TLX function `clc_consumer` reads the cluster launch control (CLC) response
from shared memory with a generic-proxy load, then releases the buffer to the
producer. The producer's next `try_cancel` writes the same buffer through the
async proxy. Without a `fence.proxy.async.shared::cta` between the read and the
release, that write is not ordered after the read.

The fix is
[`facebookexperimental/triton#1723`](https://github.com/facebookexperimental/triton/pull/1723).
`fbtriton` 3.7.4 includes it. [`affected.patch`](affected.patch) removes the
fence from the installed wheel to restore the affected source.

No runtime failure was observed in this case. Both states pass `repro.py`. The
difference is visible in the generated SASS and in a CUTracer trace.

## Requirements

- NVIDIA B200 GPU
- `fbtriton==3.7.4` from PyPI (provides the `triton` import, including TLX)
- PyTorch with CUDA support
- CUTracer, for the trace step

## Set up

Install `fbtriton` after PyTorch, because the PyTorch wheel installs the
upstream `triton` distribution:

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python torch numpy \
  --index-url https://download.pytorch.org/whl/cu130
uv pip uninstall --python .venv/bin/python triton
uv pip install --python .venv/bin/python fbtriton==3.7.4
SITE=$(.venv/bin/python -c "import triton, os; print(os.path.dirname(os.path.dirname(triton.__file__)))")
```

## Run the affected and fixed states

The patch edits one Python file, so nothing is rebuilt. Use a separate Triton
cache for each state, because the cache key may not change with this edit.

```bash
patch -d "$SITE" -p1 < affected.patch
TRITON_CACHE_DIR=$PWD/cache-affected .venv/bin/python repro.py --iterations 100

patch -d "$SITE" -p1 -R < affected.patch
TRITON_CACHE_DIR=$PWD/cache-fixed .venv/bin/python repro.py --iterations 100
```

Both runs printed `0/100 iterations mismatched` on every tested shape. Exit code
`1` means that a mismatch was detected.

## Inspect the SASS

```bash
for s in affected fixed; do
  cubin=$(find cache-$s -name 'matmul_kernel_tma_ws_blackwell_clc.cubin')
  echo "== $s"
  nvdisasm -c "$cubin" | grep -E 'LDS\.128 .*0x200e0|MEMBAR\.ALL\.CTA|FENCE\.VIEW\.ASYNC\.S|SYNCS\.ARRIVE.*0x20100'
done
```

In the affected state, each `LDS.128 [..0x200e0]` (the response read) goes
straight to `SYNCS.ARRIVE [..0x20100]` (the release). In the fixed state,
`MEMBAR.ALL.CTA ; FENCE.VIEW.ASYNC.S` sits between them. The recorded output is
in [`evidence/sass_excerpt.txt`](evidence/sass_excerpt.txt). Offsets can differ
across driver and compiler versions.

## Trace the edge

The persistent kernel reuses the response buffer only when a CTA processes more
than one tile, so trace a launch with more tiles than SMs. With
`BLOCK_SIZE_M=128` and `BLOCK_SIZE_N=256`, M=4096 and N=2048 give 256 tiles.
Then run the `audit-memory-model` skill's script on the trace. Repeat with
`cache-fixed` for the fixed state.

```bash
TRITON_CACHE_DIR=$PWD/cache-affected cutracer trace -i reg_trace,mem_addr_trace \
  -k matmul_kernel_tma_ws_blackwell_clc --output-dir trace-affected -- \
  .venv/bin/python repro.py --m 4096 --n 2048 --k 128 --iterations 1
python3 ../../agentic/skills/audit-memory-model/scripts/trace_edges.py \
  "$(ls trace-affected/*.ndjson | head -1)" | grep -- '-> 0x920'
```

In the affected state, the `generic_read->async_write` rows from the MMA-warp
and TMA-warp reads to the `try_cancel` at `0x920` have no `fence_seen`. In the
fixed state, every occurrence has it. These rows also show
`sync_lanes_unresolved` in both states, because one elected lane arrives for the
warp. The PTX joins the lanes with `bar.warp.sync`, which compiles to no SASS
instruction, so the trace cannot show the join. Recorded rows from the blind
runs are in [`evidence/runs/`](evidence/runs/).
