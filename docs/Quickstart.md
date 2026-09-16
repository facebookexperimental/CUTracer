Minimal steps to build CUTracer, attach it to an app, and collect traces.

## Prerequisites 📦
- CUDA toolkit installed and `nvcc` in PATH
- A C++ compiler (like g++)
- Git (for cloning dependencies)

> 💡 **No CUDA toolkit?** You can install one locally without `sudo`:
> ```bash
> mkdir -p ~/opt
> CUDA_INSTALL_PREFIX=~/opt ./CUTracer/scripts/install_cuda.sh 13.0
> export PATH=~/opt/cuda/bin:$PATH
> export LD_LIBRARY_PATH=~/opt/cuda/lib64:$LD_LIBRARY_PATH
> ```
> This installs CUDA 13.0 (plus cuDNN, NCCL, cuSparseLt, nvSHMEM) under `~/opt/cuda`.
> Supported versions: 12.6, 12.8, 12.9, 13.0, 13.2, 13.4. Requires ~15 GB disk space.
>
> **aarch64/GB200 support:** The script auto-detects architecture via `uname -m`. For cross-compilation or container builds targeting ARM64 (e.g., NVIDIA GB200, Grace Hopper), set `TARGETARCH=aarch64`:
> ```bash
> TARGETARCH=aarch64 CUDA_INSTALL_PREFIX=~/opt ./CUTracer/scripts/install_cuda.sh 12.8
> ```

## 1. Install Dependencies 🛠️
First, run the script to download and set up NVBit.
```bash
cd ~/CUTracer
./install_third_party.sh
```

## 2. Build CUTracer 🧱
```bash
make -j$(nproc)
ls lib/cutracer.so
```

**Note:** The `make` command will build for all GPU architectures (`-arch=all`) by default. For a faster build, you can target a specific architecture, e.g., `make ARCH=sm_90`.

## 3. Run a CUDA app with CUTracer ▶️

> 💡 **Two ways to invoke CUTracer.** The `cutracer trace …` wrapper
> (recommended) is a thin Python CLI installed via `pip install -e ./python`
> that resolves `lib/cutracer.so`, sets `CUDA_INJECTION64_PATH`, and
> translates flags into `CUTRACER_*` environment variables. The raw
> `CUDA_INJECTION64_PATH=… CUTRACER_*=…` form below is the equivalent
> "advanced" alternative — useful in CI scripts or when you want to be
> explicit about every variable.

### Instruction histogram (lightweight)

Attach CUTracer to your application. This example collects a lightweight
instruction histogram.

**Recommended (wrapper):**
```bash
cutracer trace --analysis=proton_instr_histogram --kernel-filters=add_kernel \
  -- ./your_app
```

**Equivalent (raw env vars):**
```bash
CUDA_INJECTION64_PATH=~/CUTracer/lib/cutracer.so \
CUTRACER_ANALYSIS=proton_instr_histogram \
KERNEL_FILTERS=add_kernel \
./your_app
```

**Outputs** (in your current working directory):
- `cutracer_main_YYYYMMDD_HHMMSS.log` (main tool log)
- `kernel_<hash>_iter<idx>_<name>_hist.csv` (per-kernel instruction histogram)

### Multi-mode tracing with cubin dump (advanced)

This example enables register and memory value tracing and uses uncompressed NDJSON for easy inspection.
Note: cubin dump is now **auto-enabled** when any instrumentation is active, so `CUTRACER_DUMP_CUBIN=1` is no longer needed (but still accepted).

**Recommended (wrapper):**
```bash
cutracer trace --instrument=reg_trace,mem_value_trace \
  --kernel-filters=triton_poi_fused --trace-format=ndjson \
  -- python test_add.py
```

**Equivalent (raw env vars):**
```bash
CUDA_INJECTION64_PATH=~/CUTracer/lib/cutracer.so \
CUTRACER_TRACE_FORMAT=ndjson \
KERNEL_FILTERS=triton_poi_fused \
CUTRACER_INSTRUMENT=reg_trace,mem_value_trace \
python test_add.py
```

Explanation of environment variables:
- `CUTRACER_TRACE_FORMAT=ndjson` — uncompressed NDJSON output for easy debugging (also accepts numeric `2`)
- `KERNEL_FILTERS=triton_poi_fused` — only instrument kernels matching this substring
- `CUTRACER_INSTRUMENT=reg_trace,mem_value_trace` — collect register values and memory access with values

**Outputs**:
- `cutracer_main_*.log` (main tool log)
- `kernel_*_triton_poi_fused*.ndjson` (per-kernel NDJSON trace with register and memory data)
- Cubin files for instrumented kernels (auto-dumped)

See [Configuration](Configuration.md) for all available environment variables and [Instrumentation Modes](Instrumentation-Modes.md) for mode details.

## 4. End-to-end Example (Triton Proton Test) 🔁
This demonstrates the full two-pass workflow for calculating IPC. See also: [Post-processing: IPC Merge](Post-processing-IPC-Merge.md).
```bash
cd ~/CUTracer/tests/proton_tests

# 1) Collect instruction histogram using CUTracer (filtered to add_kernel)
CUDA_INJECTION64_PATH=~/CUTracer/lib/cutracer.so \
CUTRACER_ANALYSIS=proton_instr_histogram \
KERNEL_FILTERS=add_kernel \
python ./vector-add-instrumented.py

# 2) Generate a clean Chrome trace without CUTracer for accurate timing
python ./vector-add-instrumented.py

# 3) Parse and join traces into an IPC CSV
python ~/CUTracer/scripts/parse_instr_hist_trace.py \
  --chrome-trace ./vector.chrome_trace \
  --cutracer-trace ./kernel_*_add_kernel_hist.csv \
  --cutracer-log ./cutracer_main_*.log \
  --output vectoradd_ipc.csv
```

Next: [Analyses](Analyses.md) and [Post-processing: IPC Merge](Post-processing-IPC-Merge.md).
