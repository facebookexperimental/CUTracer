## Instruction Histogram (proton_instr_histogram) 📈

- Purpose: Count instruction mnemonics per warp within regions delimited by clock reads.
- Enable with: `CUTRACER_ANALYSIS=proton_instr_histogram` (auto-enables `opcode_only`).
- Region model: first clock starts, next clock ends; alternating start/stop. Nested regions are not supported.
- Output: per-kernel CSV `kernel_<hash>_iter<idx>_pid<pid>_<name>_hist.csv` with columns `warp_id,region_id,instruction,count`.
- Typical workflow: Collect histogram with CUTracer; collect a clean Chrome trace separately; merge to compute IPC (see "Post-processing: IPC Merge").

Caveats:
- Ensure your kernel emits clock reads (e.g., Triton `pl.scope`). Without clocks, regions remain empty.
- Match kernels using `KERNEL_FILTERS` to avoid unnecessary instrumentation.

## Deadlock / Hang Detection (deadlock_detection) ⛔

- Purpose: Detect sustained kernel hangs by identifying warps stuck in stable loops.
- Enable with: `CUTRACER_ANALYSIS=deadlock_detection` (auto-enables `reg_trace`).
- Host logic summary:
  - Maintains a ring of recent PCs per warp, derives a canonical loop signature and period.
  - At most one check per second (monotonic clock). When no active warp is progressing (each is at a blocking barrier or in a stable loop) for 3 consecutive checks, the tool logs and signals termination (SIGTERM, then SIGKILL if needed). The count starts over whenever the record stream switches to another launch and whenever no warp is active.
- Output: Messages in the main log: "Possible kernel hang" on each check without progress, and "Deadlock sustained for N checks at least M ms apart" with the launch and warp counts when it terminates. `cutracer.watchdog` counts that line as a watchdog-ended hang.

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

### Searching for a Triggering Config (`cutracer stress`)

`cutracer stress` runs an oracle (exit 0 = bug reproduced) many times under delay injection, after a matched control arm (`--baseline-runs`: the same instrumentation with every delay off), and saves the first delay config that reproduces the bug for replay and `cutracer reduce`.

| Mode | What one attempt delays | Defaults | Finds |
|------|-------------------------|----------|-------|
| `lds` | Random per-thread delays before shared-memory loads only (`LDS*`), every point on | 100 and 10 us, 2 attempts each | A warp that reads a value other warps keep updating, read late (a running-max race) |
| `random` | Random per-thread delays at a random tenth of CUTracer's built-in points (memory accesses and synchronization), or in one warpgroup (`--warpgroup-ids`) | 10 and 100 us, 2 attempts each, `--enable-prob 0.1` | Races that a perturbed schedule exposes elsewhere |
| `isolate` | One CTA-local warp on one cluster rank: a fixed delay before each of its mbarrier waits (`SYNCS.PHASECHK`), while every other warp runs at full speed | 1 ms, 1 attempt per target, 60 s kernel timeout, a hang counts | A warp the synchronization protocol does not wait for, such as a consumer that reads a ring slot it never releases |

The defaults come from three races reproduced on GB200: two softmax running-max races (a warp reads the running max from shared memory after a faster warp's next update) and a DistMoE consumer that never releases its ring slot. Instrumentation alone slows such kernels 40 to 200 times, so useful delays are tens of microseconds and up. With every built-in point on, random delays almost every warp at almost every step and keeps their order (0 of 18 attempts on the running-max races); delaying only shared loads reproduced both (6 of 6 at 100 us), and a tenth of the built-in points still reaches other points. Only isolate reproduced the DistMoE hang. The modes cover only what these races needed: a clean verdict says nothing about a race between instructions none of them delays (for example TMEM accesses, named barriers or distributed shared memory), which needs a new mode calibrated on such a race.

By default (`--strategy all`) a search runs every mode, in the order above; `--strategy lds`, `--strategy random`, `--strategy isolate` or a comma-separated list picks modes. With several modes:

- Each control arm also records the instrumented launches and their delay points. A mode whose delay patterns match no instruction of the launched kernels (lds on a kernel without shared-memory loads, isolate on one without mbarrier waits) is skipped and does not make the search incomplete; a control arm that launched nothing leaves its modes not run, and the search incomplete.

- Each mode keeps its own defaults (`cutracer/stress/modes.py`) and writes its own `stress_report.json`, `stress.log` and delay configs under `<output-dir>/<mode>/`. `--mode-option MODE.KEY=VALUE` sets one option of one mode (for example `random.delay_ladder_ns=10000,100000` or `isolate.isolate_warps=0-7`); `--delay-ladder-ns`, `--attempts-per-delay`, `--delay-patterns` and `--enable-prob` apply only to a search with one mode, while `--kernel-timeout-s` and `--hang-is-bug` apply to every mode.
- An isolate search tries one target (a warp of a cluster rank) per attempt, which can be dozens per kernel. `--mode-option isolate.isolate_max_targets=N` limits it to a window of N targets, and `isolate.isolate_rotation=K` moves the window to start at target K x N (wrapping around), so a nightly run that passes an increasing K (a build number) covers every target over ceil(targets / N) runs. A clean windowed search says how many of the targets it covered, and its report has `targets_total`.
- Modes that instrument the same points the same way (delay patterns, kernel filter, timeouts) share one control arm, which runs once; each mode counts it under its own hang rule. Random and isolate delay at different points, so each has its own.
- The search stops at the first mode that reproduces (`--no-stop-on-first` runs them all). A mode that does not apply is skipped: isolate on kernels whose launches have no mbarrier wait to delay.
- `--budget-s` bounds the whole search, control arms included; no run starts with less than 30 s left.

`<output-dir>/stress_report.json` then holds the verdict, each mode's own report under `modes`, and the same top-level keys as a single-mode report (`reproduced`, `triggering_config`, the summed trial counts), plus `triggering_mode`.

Every report carries a `verdict`, with `reasons`:

| `verdict.status` | Meaning |
|------------------|---------|
| `finding` | An attempt reproduced the bug, and neither the control arm nor an attempt that injected nothing did, so the injected delay made the difference |
| `suspect` | The bug reproduced, but not attributably to a delay: it also reproduced with injection disabled, in an attempt that injected nothing, or without a complete control arm. In a search over several modes, one suspect mode makes the whole search `suspect`, even when another mode has a finding |
| `clean` | Every planned attempt ran and measured something (no infra error, timeout or unexplained hang), after a complete control arm, and nothing reproduced. It covers only the attempts made |
| `incomplete` | Anything else, including a mode that did not run and a search the budget cut short |

`--confirm` checks a finding before the search reports it: `cutracer reduce` replays the triggering config with the search's targeting, narrows it to the delay points that matter (`bisect`, majority of 3 runs per candidate), and the minimal config is replayed once more. The report then carries a `confirmation` (`confirmed`, `not_reproduced` or `unconfirmed`, with the essential points and `minimal_config.json` next to the mode's report), and the verdict stays `finding` only when the confirmation reproduced: a config that stops reproducing makes it `suspect`, and a replay that fails or runs out of `--budget-s` makes it `incomplete`.

The exit code stays 0 whenever the search itself ran; read the report for the outcome.

#### A pytest node as the oracle (`cutracer oracle pytest`)

Stress and reduce read the oracle's exit code: 0 reproduced, `--not-interesting-exit-codes` (usually 1) passed, anything else measured nothing. `cutracer oracle pytest` turns one pytest node into that contract without a wrapper script:

```bash
cutracer stress -k my_kernel --not-interesting-exit-codes 1 --confirm -o /tmp/s -- \
    cutracer oracle pytest --python .venv/bin/python --rootdir . \
    --nodeid tests/test_attn.py::test_bitwise --assertion-site tests/test_attn.py::check_bitwise \
    -k my_kernel --signature-file /tmp/s/signature -o /tmp/s/oracle
```

- Exit 0 only for an `AssertionError` whose innermost frame under `--rootdir` (installed packages excluded) is one of the `--assertion-site FILE::FUNCTION`s (any, if none is given); 1 when the node passes; 2 for a setup or collection error, a skip or xfail, another exception, a wrong selection, a `--timeout`, or a crash. A run that measured nothing never reads as a pass. A torchelastic test is judged by its first failed worker's own traceback, which needs torchelastic's `@record` on the worker's entry point.
- An injected run must also have launched a kernel matching `-k` with delay points in its delay config (it turns on kernel events and always keeps the dump), so a filter that matches nothing exits 2 instead of passing every attempt. `--plain` runs without CUTracer, for a preflight.
- `--signature-file` pins the first reproduction's signature (node and assertion site); later runs, such as the replays of `--confirm`, count only that failure. An empty file pins nothing yet.
- The node runs in a child of `--python`, the project's interpreter, which inherits CUTracer's injection; the oracle itself never loads CUDA. `-o` keeps each run's report, log, delay config and `outcome.json` in an `attempt-*` directory; every run also prints its exit code and the reason on one stderr line. `--pytest-arg=ARG` (repeatable) adds pytest arguments after the defaults `-p no:timeout -o addopts=`.
- The oracle is one standalone script (standard library only, plus pytest in the child). A search runs its oracle for every attempt and control run, and starting CUTracer each time costs seconds, so CI should write the script out once with `cutracer oracle export /tmp/s/oracle.py` and use `.venv/bin/python /tmp/s/oracle.py --rootdir . --nodeid ...` as the oracle (`--python` then defaults to that interpreter).

#### Any command as the oracle (`cutracer oracle cmd`)

For a workload that is not a pytest node — a TritonBench or tutorial script, a shell command, a Buck `python_unittest` target — `cutracer oracle cmd` maps its exit codes to the same contract without a wrapper script:

```bash
cutracer stress -k my_kernel --not-interesting-exit-codes 1 --confirm -o /tmp/s -- \
    cutracer oracle cmd --fail-codes 1 --pass-codes 0 --fail-pattern 'AssertionError|mismatch' \
    -k my_kernel --signature-file /tmp/s/signature -o /tmp/s/oracle -- \
    python /path/to/workload.py --args
```

- Exit 0 when the command exits with a `--fail-codes` code (1 by default), 1 when it exits with a `--pass-codes` code (0 by default); any other code, a signal, a `--timeout` or an interrupt exits 2, so a crash never reads as a pass or a reproduction.
- `--fail-pattern REGEX` requires the regex in the command's output (stdout and stderr) for a fail-code run to count, and pins the signature on the matched text; without it the signature is the exit code alone, so two different failures that share a fail code count as the same failure. Prefer a pattern whenever the command can fail in more than one way, especially with `--confirm`, whose replays should reproduce the same failure and not just any fail-code exit.
- Like the pytest oracle, an injected run must have launched a kernel matching `-k` with delay points in its delay config, `--signature-file` pins the first reproduction for the replays of `--confirm` (an empty file pins nothing yet), `-o` keeps each run's log, delay config and `outcome.json` in an `attempt-*` directory, every run prints its exit code and the reason on one stderr line, and `--plain` runs without CUTracer, for a preflight.
- The same standalone script serves both modes: `cutracer oracle export /tmp/s/oracle.py` once, then `/usr/bin/python3 /tmp/s/oracle.py cmd --fail-codes ... -- <command>` as the oracle.

Random delays slow a producer down along with its consumers, so they rarely let the producer lap a consumer that is missing from the release protocol. `--strategy isolate` tries one target per attempt (per instrumented kernel in first-launch order, cluster ranks high to low and then warps within each kernel, taken from every instrumented launch shape of the control arm, or from `--isolate-ranks` and `--isolate-warps`) and stops at the first that fails. A consumer that reaches its wait two phases late waits on the parity of a phase that may never complete, so isolate counts a hang ended by CUTracer's kernel timeout as a reproduction (`--hang-is-bug`, with `--kernel-timeout-s 60`) and reports it as `"symptom": "hang"`. By default each target gets one 1 ms delay: a clean kernel with W warps per CTA and clusters of C CTAs costs W × C attempts.

```bash
cutracer stress --strategy isolate -k my_kernel -o /tmp/iso -- ./oracle.sh
# Reproduced (hang): delay_ns=1000000 target=cluster rank 1, warp 4 rate=0.20 -> /tmp/iso/triggering_config.json
# Replay: cutracer trace ... --delay-mode cluster_fixed --delay-cluster-cta-id 1 --delay-warp-mask 0x10 ... --delay-load-path /tmp/iso/triggering_config.json -- ./oracle.sh
# Reduce: cutracer reduce -c /tmp/iso/triggering_config.json -t ./oracle.sh ... --delay-warp-mask 0x10 ... --hang-is-bug
```

A delay config records which points were on, not how the delay was targeted, so `triggering_config` in `stress_report.json` also carries the `target`, the `symptom` and the `replay_args` that restore the targeting. The CLI also prints the `cutracer reduce` command (`--delay-mode`, `--delay-cluster-cta-id`, `--delay-warp-mask`, `--delay-patterns`, `--kernel-timeout-s`, `--timeout`, `--hang-is-bug`) that narrows the config down to the waits that matter with the same targeting and the search's per-replay timeout. Both strategies take `--kernel-timeout-s` and `--hang-is-bug`. With a kernel timeout, the 15 s no-data watchdog is off, so a kernel slowed down by its delays is not mistaken for a hung one.

For a CUDA graph execution, the kernel timeout and the no-data watchdog are scoped to the whole graph, not to each kernel in it: timing starts at the graph's first node and is cleared only when the graph completes, and the timeout message names that first node with a `(graph)` suffix. A graph whose kernels collectively run past the limit can therefore be ended even if no individual kernel would be, and under `isolate` such a run is counted as a hang finding like any other kernel timeout.

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
| `cutracer stress` | Search for a delay config that triggers a bug, with every delay mode by default (random delays, then stalling one warp of one cluster rank at a time), and report a verdict (see "Searching for a Triggering Config" above) |
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
