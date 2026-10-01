## Build 🔨
- **1. Install Dependencies:** Before the first build, run `./install_third_party.sh` to download NVBit.
- **2. Compile:** `make -j` builds `lib/cutracer.so` for all supported architectures by default.
- For a faster, targeted build, you can specify an architecture, e.g., `make ARCH=sm_90`.
- `inject_funcs.cu` is compiled with special flags; `ptxas` version gates may alter `-maxrregcount`.

Standalone builds require NVBit 1.8.1 or newer, which includes the upstream channel synchronization fix. Run `./install_third_party.sh` when upgrading an existing checkout. `make` tracks NVBit headers and the library, and rebuilds objects and embedded fatbins after changes or dependency reinstallation, including archives that preserve older timestamps.

### Build knobs (Makefile)

These are the variables the top-level `Makefile` recognizes. They can be set
on the command line (`make DEBUG=1 …`) or as environment variables.

| Variable | Effect |
|----------|--------|
| `ARCH` | GPU architecture passed to `nvcc -arch=`. Default `all`. Examples: `sm_80`, `sm_90`, `sm_100` |
| `DEBUG` | `1` enables `-g -O0`; otherwise `-O3 -g`. Default off |
| `STATIC_ZSTD` | Force static linking of `libzstd.a`. Errors out if the static lib cannot be located |
| `DYNAMIC_ZSTD` | Force dynamic linking (`-lzstd`). Useful on Ubuntu/Debian where the system `libzstd.a` is not built with `-fPIC` |
| `CXX` | Host C++ compiler used by `nvcc -ccbin` |

The Makefile auto-detects RHEL-like distros (RHEL/CentOS/Fedora/Rocky/AlmaLinux)
via `/etc/os-release` and defaults to **static** zstd there; everything else
defaults to **dynamic** zstd. Use `STATIC_ZSTD=1` / `DYNAMIC_ZSTD=1` to override.

## Local tests 🧪
- C++ baseline and injected run: `tests/vectoradd`
- Python Triton/Proton example: `tests/proton_tests`
- Hang detection example: `tests/hang_test`

Example commands ▶️:
```bash
# Build tool
cd ~/CUTracer && make -j

# VectorAdd (no CUTracer)
cd ~/CUTracer/tests/vectoradd && make && ./vectoradd

# Triton/Proton histogram collection
cd ~/CUTracer/tests/proton_tests
cutracer trace --analysis=proton_instr_histogram --kernel-filters=add_kernel \
  -- python ./vector-add-instrumented.py

# Clean Chrome trace (no CUTracer)
python ./vector-add-instrumented.py

# Merge for IPC
python ~/CUTracer/scripts/parse_instr_hist_trace.py \
  --chrome-trace ./vector.chrome_trace \
  --cutracer-trace ./kernel_*_add_kernel_hist.csv \
  --cutracer-log ./cutracer_main_*.log \
  --output vectoradd_ipc.csv

# Hang detection (intentional loop kernel)
cd ~/CUTracer/tests/hang_test
cutracer trace --analysis=deadlock_detection -- python ./test_hang.py
```

Key validations in tests:
- CUTracer run creates kernel log and matches CTA/warp EXIT lines
- Histogram CSV header: `warp_id,region_id,instruction,count`
- Generated IPC CSV has more than a minimal number of lines

## CI 🤖

The H100 and B200 CI workflows run on push to `main`/`develop` and on PRs to `main`. The `paths-ignore` list excludes `*.md`, `.gitignore`, and `docs/**`, so documentation-only changes do not trigger a full build/test cycle.

| GPU workflow | Runner label | CUDA toolkit |
|--------------|--------------|--------------|
| [H100](../.github/workflows/test-h100.yml) | `linux-gcp-h100` | 13.0 |
| [B200](../.github/workflows/test-b200.yml) | `nvidia-dgx-b200` | 13.4 |

The `workflow_dispatch` trigger also exposes `test-type` (`all` / `build-only` / `vectoradd`) and `debug` (boolean) inputs for ad-hoc runs.

Both workflows use the runner-provided `meta-triton` and `triton-main` uv environments and run `.ci/run_tests.sh`. The `meta-triton` lane skips Proton; the `triton-nightly` lane uses `triton-main` and includes Proton. The B200 runner label is also used by [facebookexperimental/triton](https://github.com/facebookexperimental/triton/blob/main/.github/workflows/b200.yml).

The shared [GPU setup script](../.ci/setup-gpu.sh) installs the selected CUDA toolkit and test dependencies. B200 sets `CUDA_VERSION=13.4` and `INSTALL_CUDA_COMPAT=1` to install `cuda-toolkit-13-4` and `cuda-compat-13-4`, with `/usr/local/cuda-13.4/compat` first in `LD_LIBRARY_PATH`. These [forward-compatibility libraries](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html) provide CUDA 13.4 PTX JIT support with the runner's R580 host driver. The H100 entrypoint keeps its CUDA 13.0 default.

Before building CUTracer, the B200 workflow verifies the toolkit version, CUDA initialization, B200 compute capability `(10, 0)`, and GPU arithmetic. It records the loaded CUDA driver library and Driver API version, plus the PyTorch and Triton versions. The runner-provided PyTorch currently uses a CUDA 13.0 build; its build version is recorded separately from the CUDA 13.4 toolkit used to compile CUTracer. Setup logs, environment details, test logs, and traces are uploaded as artifacts with a 30-day retention period.
