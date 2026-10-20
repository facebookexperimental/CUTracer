# Prompt given to all three agents

Each agent got this prompt in its own workspace and on its own GPU. `<workspace>` and `<n>` replace the per-agent directory and GPU index. Nothing else differs between the prompts.

```text
Investigate the correctness of the GPU kernel launched by this command on
this AMD Instinct MI350 (gfx950) host:

  <workspace>/build/bin/tile_example_fmha_fwd -prec=bf16 -b=4096 -h=1 -s=16 -s_k=200 -d=96 -lse=1

It is the bf16 FMHA forward example from Composable Kernel (CK). The source
is in <workspace>/ck (a snapshot without version-control history). It is built in
<workspace>/build for gfx950, with the FMHA code generator restricted to head
dimension 96 (`--optdim 96` in example/ck_tile/01_fmha/CMakeLists.txt).
Rebuild with `ninja -C <workspace>/build tile_example_fmha_fwd`.

Observed so far: with GPU validation (-v=2), 500 of 500 runs at this shape
reported valid:y.

Tools:
- ROCm 7.0 in /opt/rocm (hipcc, llvm-objdump, roc-obj, rocgdb, rocprofv3).
- CUTracer skills: CUTracer's agent skills are installed in this session; use
  them as you see fit.

Environment: use GPU <n> (HIP_VISIBLE_DEVICES=<n>). Other users may share
this host.

Work only from the source and build in <workspace>, the tools, the CUTracer guides
above, public documentation (AMD ISA guides, LLVM AMDGPU documentation, ROCm
documentation) and what you measure. Do not read other directories on this
host. Do not search for existing issues, PRs, commits, tests, or reports about
this kernel or library, and do not download other versions of CK or any other
package to compare against.

Keep everything under <workspace>/out/:
- timeline.md: a timestamped entry (`date -Is`) for each step, with the
  hypothesis, the action, and the decision it led to;
- commands.sh: every command you ran, in order;
- the logs, disassembly excerpts and other artifacts your claims rely on.
If you test a change to any source, record the exact patch and restore the
original afterwards.

Finish with <workspace>/out/conclusion.md: your verdict (the defect you found, the
narrowest boundary you could establish, or no defect found), each claim marked
observed, inferred or causal with the artifact that supports it, any fix you
propose, and how you verified it.
```
