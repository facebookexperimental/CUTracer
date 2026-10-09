# TLX named-barrier kernel hang

This example reproduces a kernel hang in an affected TLX compiler. The public
fix changes named-barrier lowering, but the handed-off investigation artifacts
contain only the affected runtime capture. See [Investigation.md](Investigation.md)
for the bounded conclusion and the separate source-level fix comparison.

## Build the affected compiler

The regression is present at Triton commit
`d077424a67a74634e64ce35f644b2ac36eccd229`, the parent of the public fix.
Build that revision in an isolated environment:

```bash
git clone https://github.com/facebookexperimental/triton.git
cd triton
git checkout d077424a67a74634e64ce35f644b2ac36eccd229
python3 -m venv .venv
.venv/bin/pip install -r python/requirements.txt torch
.venv/bin/pip install -e . --no-build-isolation
```

The Triton build downloads its matching LLVM toolchain during installation.
The test requires an NVIDIA Hopper or Blackwell GPU.

## Run the reproducer

From the CUTracer repository root, run:

```bash
/path/to/triton/.venv/bin/python \
  examples/tlx_named_barrier_deadlock/repro.py --timeout 15
```

The observed affected run printed this message and exited with status 124:

```text
HANG REPRODUCED: kernel exceeded 15 seconds
```

The child process isolates the hung CUDA context. The timeout prevents the
reproducer from blocking the calling shell.

## Compare with the known fix

Triton commit
[`a6ffe3ce`](https://github.com/facebookexperimental/triton/commit/a6ffe3ce5ee1ade3596332c85ce3a1e6ea377630)
replaces PTX inline assembly for named barriers with convergent NVVM
intrinsics. To create a matched treatment artifact, rebuild that exact revision
in a separate environment and rerun the same command:

```bash
git clone https://github.com/facebookexperimental/triton.git triton-fixed
cd triton-fixed
git checkout a6ffe3ce5ee1ade3596332c85ce3a1e6ea377630
python3 -m venv .venv
.venv/bin/pip install -r python/requirements.txt torch
.venv/bin/pip install -e . --no-build-isolation
```

Then, from the CUTracer repository root:

```bash
/path/to/triton-fixed/.venv/bin/python \
  examples/tlx_named_barrier_deadlock/repro.py --timeout 15
```

The acceptance criterion for a fixed build is exit status 0 and:

```text
PASS: kernel completed with output [5, 99]
```

This command is the required follow-up, not a result claimed by the preserved
blind artifacts.
