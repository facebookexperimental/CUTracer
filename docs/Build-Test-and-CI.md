## Build 🔨
- **1. Install Dependencies:** Before the first build, run `./install_third_party.sh` to download NVBit.
- **2. Compile:** `make -j` builds `lib/cutracer.so` for all supported architectures by default.
- For a faster, targeted build, you can specify an architecture, e.g., `make ARCH=sm_90`.
- `inject_funcs.cu` is compiled with special flags; `ptxas` version gates may alter `-maxrregcount`.

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

CI is defined in [`.github/workflows/test.yml`](../.github/workflows/test.yml)
and runs on push to `main`/`develop` and on PRs to `main`. The `paths-ignore`
list excludes `*.md`, `.gitignore`, and `docs/**`, so documentation-only
changes do not trigger a full build/test cycle.

| Job | Runner | What it does |
|-----|--------|--------------|
| `format-check` | `ubuntu-latest` | Installs `clang-format==21.1.2` and the Python dev extras, then runs `./format.sh check` |
| `build-and-test (default)` / `build-and-test (nightly)` | `4-core-ubuntu-gpu-t4` | Matrix job over `triton: [default, nightly]`. Both legs build CUTracer via `bash .ci/setup.sh` (CUDA 12.8) and run the full suite via `bash .ci/run_tests.sh` (`TEST_TYPE=all`, `TIMEOUT=60`, `INSTALL_THIRD_PARTY=1`). The `nightly` leg first replaces PyTorch nightly's bundled `pytorch-triton` with the upstream `Triton-Nightly` wheel from OpenAI's Azure DevOps feed; it carries `continue-on-error: true` and is treated as a canary, since upstream Triton main can ABI-drift from the commit PyTorch nightly pins. Artifacts (per leg): vectoradd, py_add, proton_tests logs/traces |
| `check-status` | `ubuntu-latest` | Aggregates the matrix `build-and-test` results. Fails the workflow if the required `default` leg failed; the `nightly` leg is non-blocking (canary) |

The `workflow_dispatch` trigger also exposes `test-type` (`all` /
`build-only` / `vectoradd`) and `debug` (boolean) inputs for ad-hoc runs.
