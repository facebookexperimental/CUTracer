---
name: investigate-kernel
description: Use when a CUDA or Triton GPU kernel misbehaves or its correctness is in doubt and no CUTracer workflow has been chosen yet, for example when asked to debug or investigate a kernel without a named symptom, when it is unclear whether the problem is in the kernel at all, or when the symptom is a hang, a timeout, CUDA error 700, an illegal or misaligned address, or a test that passes while the kernel's correctness is still in doubt.
license: MIT
compatibility: Trace-based workflows need an NVIDIA GPU and the cutracer CLI with cutracer.so (see the CUTracer readme, Quickstart). audit-memory-model needs the generated code and ISA for a supported target.
---

# Investigate a GPU kernel

This skill places the issue.
It decides whether the problem is in the kernel at all, which skill investigates it, and when to switch skills.
The skills it routes to carry the procedures.

## Check the setup

1. Check that `cutracer --help` runs.
   If it is not installed, install the CLI from PyPI, in a virtual environment: `pip install --pre cutracer`.
   `cutracer --help` lists the subcommands; run `cutracer <subcommand> --help` before using a flag.
   The installed CLI is the reference for commands and flags, not these skills.
2. Make sure `cutracer.so` is available for the subcommands that need it:

   | Subcommand | Needs `cutracer.so` | Why |
   |---|---|---|
   | `trace`, `stress`, `reduce` | Yes | They inject it into the program |
   | `query`, `sass`, `analyze warp-summary` | No | They read files that already exist |
   | `compute-sanitizer` | No | It runs NVIDIA's sanitizer, from a CUDA toolkit |

   The PyPI package does not include it; build it from the CUTracer source as the Installation section of the repository's [readme](https://github.com/facebookexperimental/CUTracer/blob/main/readme.md) describes.
   The result is `lib/cutracer.so`: pass `--cutracer-so <path>/lib/cutracer.so`, or run from the source root, where `./lib/cutracer.so` is found.
   Some installs bundle it; `cutracer trace` reports when it cannot find it.
   The commands in these skills omit `--cutracer-so`; add it to every `trace`, `stress`, and `reduce` command.
3. Identify the platform.
   `nvidia-smi` succeeds on NVIDIA; `rocminfo` succeeds on AMD.
   On AMD, CUTracer cannot capture traces.
   For wrong output or NaN that comes and goes, or appears only under load, use `audit-memory-model`; it checks whether the kernel's mechanisms are ones it covers.
   No skill here covers an AMD fault (illegal or misaligned address) or an AMD failure that repeats the same way on every run; use the ROCm debugging tools.

## Establish the context

Collect the facts that decide the route:

1. The health of the GPU.
   `nvidia-smi -q` must not show a pending `GPU Recovery Action`, and a trivial CUDA program must run.
2. The GPU model and compute capability:

   ```bash
   nvidia-smi --query-gpu=index,name,compute_cap,driver_version --format=csv
   ```

3. Whether a kernel runs, and which one.
   Check the compiler or runtime error first.
   `cutracer trace --kernel-events nostack -o <dir> -- <command>` writes a kernel-events file that lists every launch in order; an empty file means no kernel ran.
   If the program may hang, run it and this command under `timeout -s INT`.
   The last launch before a hang or a fault is the suspect.
4. How it fails.
   Run the reproducer 5 to 10 times without CUTracer and record the exit code, the output, and the run time. A hang that reproduces on each of the first two runs needs no more.
   Compare the error with the test's tolerance and with the precision of its reference.
   For NaN, Inf, or garbage output, check that the kernel's inputs are finite and initialized.
   For a timeout, check whether the program finishes when given several times longer.
5. Who reads the output.
   Note the stream or event between the kernel and the host step that reads or reuses its output.
6. The software versions.
   A failure that follows a version is a compiler, driver, or library question, and the report needs them either way:

   | Component | How to read it |
   |---|---|
   | Driver, and the newest CUDA version it supports | The `nvidia-smi` header |
   | CUDA toolkit | `nvcc --version` |
   | `ptxas` that compiled the kernel | Triton uses its own copy under `triton/backends/nvidia/bin/` unless `TRITON_PTXAS_PATH` names another; run that one with `--version` |
   | Triton | `python -c "import triton; print(triton.__version__)"` |
   | PyTorch, and the CUDA version it was built with | `python -c "import torch; print(torch.__version__, torch.version.cuda)"` |
   | CUTracer | `cutracer --version`, and the commit `cutracer.so` was built from |
   | AMD: ROCm and the compiler | `/opt/rocm/.info/version`, `hipcc --version` |

Also record, for the report and for the skill you pick: which inputs fail and the smallest one that does, the last version that passed, and any sanitizer results that already exist.

## Is the kernel the problem?

A CUTracer skill investigates one thing: a defect in the device code of a kernel that ran.
Many failures look like that and are not, so decide this before picking a workflow.
Use the facts from "Establish the context" to answer:

- Does a trivial CUDA program fail too? Then the GPU or the host is at fault.
- Did the kernel run at all? A compile error, an allocation failure, or a launch the runtime rejects never reaches device code.
- Is the output wrong, or only slower, or different only by what a change in floating-point summation order allows?
- Is the output wrong when it leaves the kernel? Bad input, a host step that reads or reuses the output too early, another rank, or another job on the same GPU can each make a correct kernel look wrong.
- Is the expectation right? An error that shrinks against a reference at the kernel's own precision is a tolerance question.

If the facts place the failure outside the kernel's device code, say where it belongs and stop; do not start a CUTracer workflow.
Otherwise, pick the workflow below.

## Pick the workflow

| Symptom | Use |
|---|---|
| A fault in device code: CUDA error 700, "unspecified launch failure", an illegal or misaligned address | `debug-illegal-access` |
| A device kernel stops making progress | `debug-hanging-kernel` |
| Tests pass, but a fence, wait, or barrier may be missing, on a target and mechanism `audit-memory-model` lists | `audit-memory-model` |

Take the first row that matches; an earlier symptom can hide a later one.

## End the investigation with evidence

- Use the chosen skill's evidence bar.
  An `audit-memory-model` finding needs no reproduced failure.
- "Tests pass" or "no reproduction" bounds only the conditions tested; report it as such.
- If the source is right and the compiled code is wrong, for example when only one compiler version fails, the conclusion goes to the compiler.
- When the user asks for a write-up and the investigation has reached a conclusion, use `report-case-study`.

## Common mistakes

| Mistake | Instead |
|---|---|
| Setting `CUDA_INJECTION64_PATH` yourself | Leave it unset; `cutracer` sets it, and refuses to run when it is already set. |
