# TLX named-barrier kernel hang

This example reproduces a kernel hang in an affected TLX compiler. The same
kernel completes with a compiler that contains the fix.

## Build the affected compiler

The regression is present at Triton commit
`d077424a67a74634e64ce35f644b2ac36eccd229`. Build that public revision in an
isolated environment:

```bash
git clone https://github.com/facebookexperimental/triton.git
cd triton
git checkout d077424a67a74634e64ce35f644b2ac36eccd229
python3 -m venv .venv
.venv/bin/pip install -r python/requirements.txt torch
.venv/bin/pip install -e . --no-build-isolation
```

The Triton build downloads its matching LLVM toolchain during installation.
The test also requires an NVIDIA Hopper or Blackwell GPU.

## Run the reproducer

```bash
/path/to/triton/.venv/bin/python \
  examples/tlx_named_barrier_deadlock/repro.py --timeout 15
```

An affected compiler prints this message after the timeout and exits with
status 124:

```text
HANG REPRODUCED: kernel exceeded 15 seconds
```

A fixed compiler prints this message and exits with status 0:

```text
PASS: kernel completed with output [5, 99]
```

The child process isolates the hung CUDA context. The timeout prevents the
reproducer from blocking the calling shell.

## Check a current release

The PyPI package is named `fbtriton`, but it provides the `triton` Python
import. Install it in a separate environment to check the fixed behavior:

```bash
python3 -m venv .venv-fixed
.venv-fixed/bin/pip install torch fbtriton
.venv-fixed/bin/python \
  examples/tlx_named_barrier_deadlock/repro.py --timeout 15
```
