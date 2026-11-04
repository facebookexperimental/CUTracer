## Instruction Histogram (proton_instr_histogram) 📈

- Purpose: Count instruction mnemonics per warp within regions delimited by clock reads.
- Enable with: `CUTRACER_ANALYSIS=proton_instr_histogram` (auto-enables `opcode_only`).
- Region model: first clock starts, next clock ends; alternating start/stop. Nested regions are not supported.
- Output: per-kernel CSV `kernel_<hash>_iter<idx>_<name>_hist.csv` with columns `warp_id,region_id,instruction,count`.
- Typical workflow: Collect histogram with CUTracer; collect a clean Chrome trace separately; merge to compute IPC (see "Post-processing: IPC Merge").

Caveats:
- Ensure your kernel emits clock reads (e.g., Triton `pl.scope`). Without clocks, regions remain empty.
- Match kernels using `KERNEL_FILTERS` to avoid unnecessary instrumentation.

## Deadlock / Hang Detection (deadlock_detection) ⛔

- Purpose: Detect sustained kernel hangs by identifying warps stuck in stable loops.
- Enable with: `CUTRACER_ANALYSIS=deadlock_detection` (auto-enables `reg_trace`).
- Host logic summary:
  - Maintains a ring of recent PCs per warp, derives a canonical loop signature and period.
  - When all active warps are in stable loops for consecutive checks, the tool logs and signals termination (SIGTERM, then SIGKILL if needed).
- Output: Messages in the main log (e.g., "Possible kernel hang", "Deadlock sustained...").

Caveats:
- `reg_trace` increases overhead; narrow `KERNEL_FILTERS` and instruction intervals to reduce impact.
- EXIT opcode detection helps prune exiting warps to avoid false positives.

## Data Race Detection (random_delay) 🔀

- Purpose: Expose hidden data races by injecting delays before synchronization instructions.
- Enable with: `CUTRACER_ANALYSIS=random_delay` (auto-enables `RANDOM_DELAY` instrumentation).
- Requires: `CUTRACER_DELAY_NS` environment variable to specify the delay value in nanoseconds.

### How It Works

Data races depend on timing and often pass by luck. This analysis disrupts timing by:
1. Identifying synchronization and memory instructions that can be involved in race conditions
2. Randomly enabling/disabling delay injection for each instruction (50% probability)
3. Injecting a fixed delay (specified by `CUTRACER_DELAY_NS`) before enabled instructions

#### Targeted Instruction Patterns

Delays are injected before the following SASS instruction patterns:

| Pattern | Category |
|---------|----------|
| `SYNCS.PHASECHK.TRANS64.TRYWAIT` | mbarrier try_wait |
| `SYNCS.ARRIVE.TRANS64.RED.A1T0` | mbarrier arrive |
| `UTCBAR` | mbarrier arrive |
| `UTMALDG` | TMA load |
| `UTMASTG` | TMA store |
| `UTMAREDG` | TMA store with reduction |
| `WARPGROUP.DEPBAR.LE` | MMA wait |
| `UTCQMMA`, `UTCHMMA`, `UTCIMMA`, `UTCOMMA` | MMA operations |
| `LDTM`, `LDT` | Tensor memory load (Blackwell sm_100+) |
| `STT`, `STTM` | Tensor memory store (Blackwell sm_100+) |
| `LD`, `ST` | Generic load/store |

### Basic Usage

```bash
CUTRACER_DELAY_NS=10000 \
CUTRACER_ANALYSIS=random_delay \
CUDA_INJECTION64_PATH=~/CUTracer/lib/cutracer.so \
python3 your_kernel.py
```

### Delay Dump and Replay

CUTracer supports dumping delay configurations to JSON for **deterministic reproduction** of data races. This is essential for debugging because once a race condition triggers a failure, you need to reproduce the exact same timing to debug it.

#### Environment Variables

| Variable | Purpose |
|----------|---------|
| `CUTRACER_DELAY_NS` | Fixed delay value in nanoseconds (required) |
| `CUTRACER_DELAY_DUMP_PATH` | Output path for delay config JSON (dump mode) |
| `CUTRACER_DELAY_LOAD_PATH` | Input path for delay config JSON (replay mode) |
| `CUTRACER_DELAY_SEED` | Seed of the per-point on/off choices (dump mode; random if unset) |

**Note**: You cannot use `CUTRACER_DELAY_DUMP_PATH` and `CUTRACER_DELAY_LOAD_PATH` at the same time.

**Multi-process runs**: every process of a run (for example torchrun ranks) writes the same `CUTRACER_DELAY_DUMP_PATH`. A point's on/off choice depends only on `CUTRACER_DELAY_SEED`, the kernel checksum and the PC, so processes that share the seed make the same choices, and their writes merge into one config that every process can replay. `cutracer trace --delay-dump-path` and `cutracer stress` set one seed per run. Each process rewrites the file after it instruments a new kernel, so the config exists even when a process exits without ending its CUDA contexts.

#### Workflow

1. **Run with dump mode** to record the random delay pattern:
   ```bash
   CUTRACER_DELAY_NS=10000 \
   CUTRACER_DELAY_DUMP_PATH=/tmp/delay_config.json \
   CUTRACER_ANALYSIS=random_delay \
   CUDA_INJECTION64_PATH=~/CUTracer/lib/cutracer.so \
   python3 your_kernel.py
   ```

2. **When a failure occurs**, save the config file. The JSON contains:
   - Per-kernel instrumentation points
   - PC offset and SASS instruction for each point
   - Delay value and enabled/disabled state

3. **Replay with the saved config** to reproduce deterministically:
   ```bash
   CUTRACER_DELAY_NS=10000 \
   CUTRACER_DELAY_LOAD_PATH=/tmp/delay_config.json \
   CUTRACER_ANALYSIS=random_delay \
   CUDA_INJECTION64_PATH=~/CUTracer/lib/cutracer.so \
   python3 your_kernel.py
   ```

#### JSON Config Format

The delay config file has the following structure:

```json
{
  "version": "1.0",
  "delay_ns": 10000,
  "kernels": {
    "kernel_name_2026-02-03T21:15:21.567": {
      "kernel_name": "matmul_kernel",
      "timestamp": "2026-02-03T21:15:21.567",
      "instrumentation_points": {
        "10192": {
          "pc": 10192,
          "sass": "SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [UR15+0x38110], R4 ;",
          "delay": 10000,
          "on": true
        },
        "10864": {
          "pc": 10864,
          "sass": "WARPGROUP.DEPBAR.LE gsb0, 0x0 ;",
          "delay": 10000,
          "on": false
        }
      }
    }
  }
}
```

---

## SASS Extraction (cutracer sass) 🔍

- Purpose: Extract disassembled SASS from cubin files for inspection and analysis.
- CLI command: `cutracer sass`
- Uses `nvdisasm` under the hood with source-level debug info enabled by default.

### Usage

```bash
cutracer sass kernel.cubin              # writes kernel.sass to disk
cutracer sass kernel.cubin -o out.sass  # explicit output path
cutracer sass kernel.cubin --stdout     # print to stdout
cutracer sass kernel.cubin -G -C        # minimal output (no debug/source info)
cutracer sass kernel.cubin --timeout 120
```

### Options

| Option | Description |
|--------|-------------|
| `--output / -o` | Output `.sass` file path (default: replaces `.cubin` with `.sass`) |
| `--no-source-info / -G` | Omit `-g` flag (source-level debug info) |
| `--no-line-info / -C` | Omit `-c` flag (`//##` source line comments) |
| `--timeout` | nvdisasm timeout in seconds (default: 60) |
| `--stdout` | Print to stdout instead of writing a file |

### Programmatic API

```python
from cutracer.query.sass import dump_sass, dump_sass_to_file

# Returns SassOutput with .raw_text, .cubin_path, .flags_used, .line_count
result = dump_sass(Path("kernel.cubin"))
print(result.raw_text)
result.save(Path("kernel.sass"))

# Convenience: write directly to disk
dump_sass_to_file(Path("kernel.cubin"), output_path=Path("out.sass"))
```

Note: Cubin files are auto-dumped when any instrumentation is active (see [Configuration](Configuration.md#debug-configuration)), so you can use this command on the cubin files generated during a CUTracer run.

---

## CLI Subcommands 🛠️

The `cutracer` Python CLI (installed via `pip install -e ./python`) bundles
several subcommands. Run any of them with `--help` for full details.

| Subcommand | Purpose |
|------------|---------|
| `cutracer trace` | Wrapper that sets `CUDA_INJECTION64_PATH` and `CUTRACER_*` env vars and runs your CUDA app. See [Quickstart](Quickstart.md) and [Configuration](Configuration.md) |
| `cutracer validate` | Validate the structure / schema of an NDJSON, Zstd-NDJSON, or text trace file |
| `cutracer query` | Inspect a trace file with row-level filters (`--filter "warp=24;pc=0x43d0"`), `--head` / `--tail` / `--all-lines`, and `--group-by … --count` aggregation |
| `cutracer sass` | Disassemble a `.cubin` to SASS via `nvdisasm` (see "SASS Extraction" above) |
| `cutracer reduce` | Post-process a trace into a smaller summary report (see `cutracer reduce --help`) |
| `cutracer compare` | Diff two traces / validation reports and surface deltas |
| `cutracer analyze` | Analysis subgroup. Includes `cutracer analyze warp-summary <trace>`; internal builds expose additional analyses (e.g. data-race, deadlock, MMA/TMA dataflow) under the same group |

A few quick examples (from `cutracer --help`):

```bash
cutracer trace -i tma_trace -- ./vectoradd
cutracer trace -i tma_trace --instr-categories=tma -- python my_test.py
cutracer validate kernel_trace.ndjson
cutracer validate kernel_trace.ndjson.zst --verbose
cutracer query trace.ndjson --filter "warp=24"
cutracer query trace.ndjson -f "pc=0x43d0;warp=24"
cutracer query trace.ndjson --group-by warp --count
cutracer analyze warp-summary trace.ndjson
```
