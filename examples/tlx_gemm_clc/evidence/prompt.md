# Prompt given to all three agents

Each agent got this prompt in its own workspace and on its own GPU. `<workspace>` and `<n>` replace the per-agent path and GPU index. Nothing else differs between the three prompts.

```text
Investigate the correctness of the GPU kernel launched by <workspace>/repro.py
on this NVIDIA B200 host. The kernel is the matmul from
triton.language.extra.tlx.tutorials.blackwell_gemm_clc, provided by the fbtriton
package installed in <workspace>/.venv.

Observed so far: <workspace>/repro.py compares the kernel against
torch.matmul (atol=rtol=1e-2). On one GPU it reported 0 mismatched iterations out
of 50 for each of these shapes: M=N=4096 K=512; M=4096 N=1536 K=64;
M=N=8192 K=512; M=N=2048 K=64.

Tools:
- CUTracer: use <workspace>/.venv/bin/cutracer (run `cutracer --help`). Run it
  from <workspace> so it picks up ./lib/cutracer.so. Do not build or run any
  other CUTracer build.
- CUTracer skills: CUTracer's agent skills are installed in this session; use
  them as you see fit.
- Standard NVIDIA tools on the host (nvdisasm, cuobjdump, compute-sanitizer).

Environment: use GPU <n> (CUDA_VISIBLE_DEVICES=<n>). Run Python as
<workspace>/.venv/bin/python with TRITON_CACHE_DIR=<workspace>/cache.

Work only from the reproducer, the installed package source, the tools, the
CUTracer guides above, public documentation (PTX ISA, CUDA, CUTLASS docs) and
what you measure. Do not read other directories on this host. Do not search for
existing issues, PRs, commits, tests, or reports about this kernel or library,
and do not download other versions of any package to compare against.

Keep everything under <workspace>/out/:
- timeline.md: a timestamped entry (`date -Is`) for each step, with the
  hypothesis, the action, and the decision it led to;
- commands.sh: every command you ran, in order;
- the traces, logs and excerpts your claims rely on (large raw traces may be
  deleted once you have extracted what you need; record that you did).
If you test a change to any source, record the exact patch and restore the
original afterwards.

Finish with <workspace>/out/conclusion.md: your verdict (the defect you
found, the narrowest boundary you could establish, or no defect found), each
claim marked observed, inferred or causal with the artifact that supports it,
any fix you propose, and how you verified it.
```
