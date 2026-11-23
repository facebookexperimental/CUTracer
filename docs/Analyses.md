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
  - At most one check per interval (default 1000 ms, monotonic clock). When no active warp is progressing (each is at a blocking barrier or in a stable loop) for 3 consecutive checks (`--hang-check-hits`), and the stall has lasted the minimum stall time (`--hang-min-stall-s`, default 0, counted from the first check without progress), the tool logs and signals termination (SIGTERM, then SIGKILL if needed). The count starts over whenever the record stream switches to another launch and whenever no warp is active.
  - With the defaults a correct kernel whose warps legitimately spin longer than 2 to 3 s, for example on another rank, is ended as a hang. Set `--hang-min-stall-s` above the longest such wait (`CUTRACER_HANG_CHECK_INTERVAL_MS`, `CUTRACER_HANG_CHECK_HITS` and `CUTRACER_HANG_MIN_STALL_S` without the CLI). Warps that block without spinning send no trace data, so the no-data watchdog (`--no-data-timeout-s`, default 15 s) ends such a stall first; raise it too if warps can block that long.
- Output: Messages in the main log: "Possible kernel hang" on each check without progress, with the per-warp report on the first such check and when the run is ended, and "Deadlock sustained for N checks at least M ms apart over T ms (minimum stall S ms)" with the launch and warp counts when it terminates. `cutracer.watchdog` counts that line as a watchdog-ended hang. With a non-text trace format, the first hang check without progress, the check that ends the run, and a kernel or no-data timeout that ends it also append one JSON snapshot of every active warp's status, last instruction and loop body, plus the finished and never-executed warp ids and a `trigger` record saying which check wrote it (see [Warp Status NDJSON](Outputs-and-File-Formats.md#warp-status-ndjson-)), to `<kernel log basename>_warp_status_summary.ndjson` (nothing is written before the first trace data arrives). The file is appended to across runs. `cutracer analyze warp-summary` prints the never-executed count of the traced launch.

Caveats:
- `reg_trace` increases overhead; narrow `KERNEL_FILTERS` and instruction intervals to reduce impact.
- EXIT opcode detection helps prune exiting warps to avoid false positives.
- The check needs CUTracer's default blocking launches (it is rejected with `--nonblocking-launches`). A program in which a running kernel waits for a later launch, such as a cross-rank cohort in eager mode, hangs under blocking launches, so the online check cannot be used on it; trace it with `--nonblocking-launches` and analyze offline instead.

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
| `pdl` | A fixed delay on every warp before each global-memory store (`STG`, `ST.E`, `REDG`, `ATOMG`, `ATOM.E`, `UBLKCP.G`, `UBLKRED.G`, `SUST`, `SURED`; `STG` and `REDG` also match the TMA stores `UTMASTG` and `UTMAREDG`), every point on, with non-blocking launches in every run, control arm included. Skipped when no launch uses PDL | 300 us and 1 ms, 2 attempts each | A kernel launched with programmatic dependent launch (PDL) that reads an earlier kernel's output before that kernel writes it: no `griddepcontrol.wait`, or a read before it |
| `random` | Random per-thread delays at a random tenth of CUTracer's built-in points (memory accesses and synchronization), or in one warpgroup (`--warpgroup-ids`) | 10 and 100 us, 2 attempts each, `--enable-prob 0.1` | Races that a perturbed schedule exposes elsewhere |
| `isolate` | One CTA-local warp on one cluster rank: a fixed delay before each of its mbarrier waits (`SYNCS.PHASECHK`), while every other warp runs at full speed | 1 ms, 1 attempt per target, 60 s kernel timeout, a hang counts | A warp the synchronization protocol does not wait for, such as a consumer that reads a ring slot it never releases |
| `bar` | One named barrier: a fixed delay on every warp before its syncs and reductions (`BAR.SYNC`, `BAR.RED`), every instruction that names one immediate barrier id or one instruction whose barrier id is in a register; arrives (`BAR.ARV`) are not delayed | 10 us, 1 attempt per target, 60 s kernel timeout, a hang counts | Warps that arrive at a barrier a phase ahead of a late sync, so the sync completes on the wrong arrivals and a later one never does (a named-barrier token imbalance) |

The defaults of lds, random and isolate come from three races reproduced on GB200: two softmax running-max races (a warp reads the running max from shared memory after a faster warp's next update) and a DistMoE consumer that never releases its ring slot. Instrumentation alone slows such kernels 40 to 200 times, so useful delays are tens of microseconds and up. With every built-in point on, random delays almost every warp at almost every step and keeps their order (0 of 18 attempts on the running-max races); delaying only shared loads reproduced both (6 of 6 at 100 us), and a tenth of the built-in points still reaches other points. Only isolate reproduced the DistMoE hang.

`bar` comes from a named-barrier deadlock in an FP8-KV decode attention kernel: one warp arrives at barrier 0x5 twice before the other reaches its first sync. Only a fixed delay before that barrier's syncs, on every warp, reproduced it (30 of 30 runs hang before the fix at 1, 10 and 100 us alike, 0 of 30 after); delaying its arrive as well slows the arriving warp just as much and removed the hang (0 of 20), so `bar` does not delay arrives. A delay before every sync also slows a correct kernel: a launch takes about the delay times the delayed syncs it runs one after another (along a warp's loop and over the CTAs an SM runs in turn) longer. `bar` therefore tries one 10 us rung, at which a launch needs about 6,000,000 delayed syncs to pass the 60 s kernel timeout and be reported as a hang.

`pdl` looks for races between kernels, which the other modes cannot reach: by default CUTracer waits for the device around every launch, so a PDL consumer never overlaps its producer. One without `griddepcontrol.wait` read stale data in 10 of 10 native runs on GB200 when its producer spun 200 us after triggering its dependents, and in none of 307 runs under CUTracer. Every pdl run, its control arm included, therefore uses non-blocking launches (`--nonblocking-launches`, see "Launch Handling" in [Configuration](Configuration.md)), captures no CPU call stacks (`--cpu-callstack 0`) and records kernel events, and so do its replays: the consumer can launch while the producer's stores are still pending, and every run does the same host work per launch. Every warp waits the full delay before each global store; delaying loads would move the consumer's read later, and random per-thread delays would let some stores land on time.

The delay has to outlast the host work between the producer's launch and the consumer's, which CUTracer adds to. On GB200, with a producer that stores right after triggering its dependents (never stale natively, 0 of 20 runs), 10 and 100 us reproduced none of the four PDL races of the microbenchmark (no wait, a late wait, a wait in only some warps, and a consumer that inherits an early trigger from a kernel in between): the consumer started 0.1 to 0.8 ms after the producer ended. 300 us, 1 ms and 3 ms reproduced each race in 3 of 3 attempts, and the two correct variants stayed clean in all 15 attempts each. The 1 ms rung is only about 1.25 times the largest gap measured on one host and one benchmark, so when pdl comes back clean but a PDL race is still suspected, rerun it with a longer delay: `--mode-option pdl.delay_ladder_ns=3000000` (or `--strategy pdl --delay-ladder-ns 3000000`).

pdl applies when a control run that completed records a direct launch that uses PDL and can overlap the kernel before it: its `launch_attributes` (see [Outputs and File Formats](Outputs-and-File-Formats.md)) set `programmatic_stream_serialization` and it directly follows another direct launch on its stream, without being the first launch of an instrumented kernel (which waits for the running kernels while CUTracer instruments it), or they set a `programmatic_event`. Otherwise it is skipped, with a reason that says which of these held. A search with several modes decides from the control arms that ran before it with the same oracle and kernel filter (lds's, by default) and skips pdl without running its own, so a program without PDL launches pays nothing for it; if a later control arm records a PDL launch after all, the search counts pdl as one it could not test (see below). pdl needs every kernel such a launch may read early from instrumented: the launch before it on its stream and, while each of those used stream serialization itself, the launch before that (or, for a programmatic event, the launch itself). When the kernel filter leaves one of them out for every launch, the search cannot test pdl, and the reason names the kernel: widen `-k`. If `--strategy` lists pdl with other modes, pdl does not run and the search is incomplete; under `--strategy all`, which names no mode, pdl is skipped, and a clean verdict lists it (`pdl: skipped: ...`, with `coverage_gap: true` in its entry of `modes`). When no completed control run says whether any launch uses PDL (`--baseline-runs 0`, control runs that never completed, or a CUTracer that does not record launch attributes), a search with several modes does not search blind: it cannot test pdl, so pdl does not run and the search is incomplete if `--strategy` lists pdl with other modes, and under `--strategy all` pdl is skipped and a clean verdict lists it. `--strategy pdl` alone decides from its own control arm, exits with an error when pdl does not apply or cannot test the launches, and searches anyway when nothing tells. Limits:

- PDL inside a CUDA graph does not overlap under CUTracer, so a program whose PDL launches all come from graph replay (`captured_launch_attributes`), apart from an eager warm-up launch, is skipped, and a PDL race that only graph replay exposes is out of reach.
- The first launch of each instrumented kernel does not overlap the kernel before it, so the oracle has to launch the PDL pair more than once.
- Launches pair up by the stream handle CUTracer records, so the per-thread default streams of different threads, or the NULL streams of different contexts, look like one stream, and a launch may pair up with another thread's or context's launch.
- A delay-only kernel sends no trace records, so with non-blocking launches neither the kernel timeout nor the no-data watchdog ends a hung kernel; only `--timeout` bounds each run. Delays add up in a kernel that stores in a loop.
- A run in which CUTracer could not overlap launches (it logs that `--nonblocking-launches` has no effect) counts as an infra error.
- Non-blocking launches also expose a missing stream or event wait, or a host read without synchronization, so a pdl finding names the stores that had to be late, not that PDL is the cause; `--confirm` reduces it to those stores. It corroborates no per-kernel finding of `analyze data-race --stress-report`.

The modes cover only what the races behind them needed: a clean verdict says nothing about a race between instructions none of them delays (for example distributed shared memory), which needs a new mode calibrated on such a race.

By default (`--strategy all`) a search runs the modes in the order of the table, except those that run only when named; `cutracer stress --help` lists every mode and its defaults under "Mode defaults". `--strategy` with one mode, or a comma-separated list, picks modes. With several modes:

- Each control arm also records the instrumented launches and their delay points. A mode that does not apply to them is skipped and does not make the search incomplete: one whose delay patterns match no instruction of the launched kernels (lds on a kernel without shared-memory loads, isolate on one without mbarrier waits), or as its own paragraph says. A control arm that launched nothing leaves its modes not run, and the search incomplete.

- Each mode keeps its own defaults (`cutracer/stress/modes.py`) and writes its own `stress_report.json`, `stress.log` and delay configs under `<output-dir>/<mode>/`. `--mode-option MODE.KEY=VALUE` sets one option of one mode (for example `random.delay_ladder_ns=10000,100000` or `isolate.isolate_warps=0-7`); `--delay-ladder-ns`, `--attempts-per-delay`, `--delay-patterns` and `--enable-prob` apply only to a search with one mode, while `--kernel-timeout-s` and `--hang-is-bug` apply to every mode.
- A mode that isolates targets tries one target per attempt: for isolate a warp of a cluster rank, which can be dozens per kernel. `--mode-option isolate.isolate_max_targets=N` (likewise for any mode that isolates targets) limits it to a window of N targets, and `isolate.isolate_rotation=K` moves the window to start at target K x N (wrapping around), so a nightly run that passes an increasing K (a build number) covers every target over ceil(targets / N) runs. A clean windowed search says how many of the targets it covered, and its report has `targets_total`.
- Modes whose control runs would be the same (delay patterns, kernel filter, timeouts, launch handling) share one control arm, which runs once; each mode counts it under its own hang rule.
- The search stops at the first mode that reproduces (`--no-stop-on-first` runs them all).
- `--budget-s` bounds the whole search, control arms included; no run starts with less than 30 s left.

`<output-dir>/stress_report.json` then holds the verdict, each mode's own report under `modes`, and the same top-level keys as a single-mode report (`reproduced`, `triggering_config`, the summed trial counts), plus `triggering_mode`.

Every report carries a `verdict`, with `reasons`:

| `verdict.status` | Meaning |
|------------------|---------|
| `finding` | An attempt reproduced the bug, and neither the plain oracle (`--plain-oracle`), the control arm nor an attempt that injected nothing did, so the injected delay made the difference |
| `suspect` | The bug reproduced, but not attributably to a delay: it also reproduced without CUTracer (the plain oracle), with injection disabled, in an attempt that injected nothing, or without a complete control arm or plain run. In a search over several modes, one suspect mode makes the whole search `suspect`, even when another mode has a finding |
| `clean` | Every planned attempt ran and measured something (no infra error, timeout or unexplained hang), after a complete control arm, and nothing reproduced. It covers only the attempts made; a search with several modes lists a mode it could not test and that `--strategy` did not name, as `<mode>: skipped: <reason>` |
| `incomplete` | Anything else, including a mode that did not run and a search the budget cut short |

`--confirm` checks a finding before the search reports it: `cutracer reduce` replays the triggering config with the search's targeting, narrows it to the delay points that matter (`bisect`, majority of 3 runs per candidate), and the minimal config is replayed once more. The report then carries a `confirmation` (`confirmed`, `not_reproduced` or `unconfirmed`, with the essential points and `minimal_config.json` next to the mode's report), and the verdict stays `finding` only when the confirmation reproduced: a config that stops reproducing makes it `suspect`, and a replay that fails or runs out of `--budget-s` makes it `incomplete`.

The exit code stays 0 whenever the search itself ran; read the report for the outcome.

Three oracle rules:
- An oracle that compares floating-point outputs bitwise fails on its own when the kernel's result varies from launch to launch (fp32 atomics or bulk reduce-adds that land in no fixed order), often at a rate of about 1% that a few control runs miss, and the search then credits those failures to the delay. Give such an oracle `--plain-oracle '<command>'`: a command that runs the workload without CUTracer, many times in one process, and exits like the oracle (the oracle with an iteration count, or `cutracer oracle cmd --plain ...`). It runs once before the control arm (once per search with several modes); if it reproduces, the attribution is `natural` and the verdict `suspect`. Reports record `plain_requested`, `plain_outcome` (`reproduced`, `completed`, `timed_out` or `infra_error`) and `plain_returncode`, and its output goes to `stress.log`, or `plain.log` for a search with several modes. Then compare with a tolerance or make the kernel deterministic.
- Let CUTracer's kernel timeout end a hang (`--kernel-timeout-s`, `--hang-is-bug`). An oracle that exits 0 when its own timeout fires turns any slowdown into a reproduction in every mode, and must not be used with the default plan.
- CUTracer prints its launch lines to the program's stdout, where they can interleave mid-line with the program's own output. Do not grep stdout for a line that starts with a marker (`^RESULT`); write the program's verdict to a file or match it anywhere in a line.

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

#### Comparing numerics with a golden (`--numerics`)

A race can change a kernel's results by less than the test's tolerance, or in a tensor the test does not check, so the test passes under every delay. `--numerics` compares every tensor the test compares with a golden recorded from plain runs, instead of trusting the test's verdict alone. Both oracles take it:

```bash
# Once, without CUTracer: exits 0 when the golden is usable.
cutracer oracle pytest --numerics record --golden /tmp/s/golden --rootdir . --nodeid tests/test_attn.py::test_bwd
cutracer stress -k my_kernel --not-interesting-exit-codes 1 --confirm -o /tmp/s \
    --plain-oracle 'cutracer oracle pytest --numerics check --golden /tmp/s/golden --plain --rootdir . --nodeid tests/test_attn.py::test_bwd' -- \
    cutracer oracle pytest --numerics check --golden /tmp/s/golden --rootdir . \
    --nodeid tests/test_attn.py::test_bwd -k my_kernel --signature-file /tmp/s/signature
```

- The child imports `cutracer_numerics.py` (the capture), which records both operands of every `torch.testing.assert_close`, `torch.allclose`, `torch.equal`, `Tensor.allclose`, `Tensor.equal` and numpy `assert_allclose`, `assert_array_equal` and `assert_array_almost_equal` call the test makes. `--numerics-wrap MODULE:FUNCTION` (repeatable; `MODULE:*` for every public function of the module) also records a function's tensor results, for a test that checks nothing itself. The pytest oracle loads the capture in its worker; `oracle cmd` puts it on `PYTHONPATH` and `PYTEST_PLUGINS`, and a driver that is not pytest imports `cutracer_numerics` and can call `record_tensor(name, value)`.
- `--numerics record --golden DIR` (a new or empty directory) runs the test `--numerics-runs` times without CUTracer (3 by default): run 1 records, the others are checked against it, and each golden tensor keeps how much it moved between them. It is not an oracle: it exits 0 for a usable golden, 3 when its own plain runs already changed by a large amount or failed (the numerics are not reproducible without CUTracer, so a stress finding could not be credited to the delay; checks refuse that golden with exit 2), and 2 when it wrote none, keeping the runs' evidence (`-o`, or a temporary directory it names). `DIR/stability.json` holds each run, and `DIR/manifest.json` the test, its rootdir (or the command and working directory) and the capture's settings, which every check reuses. Each distinct tensor is stored once, in `DIR/blobs`, compressed (zstd when the project's interpreter has it, else zlib) unless a sample shows that compression does not pay; `CUTRACER_NUMERICS_MAX_TOTAL_MB` (4096 by default) bounds the bytes each process writes, and the record's last line and `stability.json` give the golden's size.
- A check classifies each tensor that differs bitwise as `small` (the size of reduction-order noise) or `large`: an element changed between finite, +inf, -inf and nan; any change of an integer, bool or complex tensor; relative L2 above 0.02; a row (the last dimension) above 0.05; or one element that moved by more than 0.25 times the larger of its golden magnitude and its row's RMS. An integer or bool tensor that already changed between the plain runs (an argmax with ties) is only `small`; a complex one is not. Each threshold is at least the dtype's eps; `CUTRACER_NUMERICS_LARGE_REL_L2`, `CUTRACER_NUMERICS_LARGE_ROW_REL_L2` and `CUTRACER_NUMERICS_LARGE_ELEM_REL` set them when recording. With `--numerics-strict`, any change of a tensor that was bitwise identical in every plain run is large too, for kernels meant to be deterministic; it needs `--numerics-runs 2` or more, and a few plain runs are weak evidence of determinism.
- A passing test with a large change exits 0, and a test that fails its correctness check exits 0 as before, whatever its numbers. A passing run that cannot be compared with the golden exits 2: a different torch, Triton or GPU, a process the golden does not know or one that wrote no result, a capture error, an extra call or a different operand structure, a missing call, or a changed tensor that could not be measured (too big to save; `CUTRACER_NUMERICS_MAX_MB`, 512 by default). A large change before the first call that does not line up still exits 0, since a race can derail the rest of a test. Skips, xfails and other exceptions exit 2 as before.
- With `--numerics`, every reproduction of one test has one signature, so `--signature-file` pins "this test reproduced" whether its assertion failed or its numbers changed. `outcome.json` gets a `numerics` object: each process's verdict, calls, worst changes, and the rule that decided.
- Calls are matched with the golden by pytest node, function, call site and ordinal; calls from more than one thread at one call site cannot be matched (the run exits 2). Only processes that import `cutracer_numerics` are compared: the pytest oracle's worker, and in `oracle cmd` any child that loads it through `PYTHONPATH` or `PYTEST_PLUGINS`. Such a child needs a `RANK` or `CUTRACER_NUMERICS_PROC` label, or it leaves a note that makes every run exit 2 (pytest-xdist workers under `oracle cmd` do; do not combine `-n` with `--numerics`). Comparisons made inside torch, numpy or Triton are not recorded. Hypothesis draws the same examples in every run.
- An autotuned Triton kernel benchmarks its configs under CUTracer in an injected run and can pick another one than in the plain runs, which changes its numbers for that reason alone.
- `cutracer oracle export PATH` also writes `cutracer_numerics.py` beside `PATH`, where the exported oracle finds it.

Random delays slow a producer down along with its consumers, so they rarely let the producer lap a consumer that is missing from the release protocol. `--strategy isolate` tries one target per attempt (per instrumented kernel in first-launch order, cluster ranks high to low and then warps within each kernel, taken from every instrumented launch shape of the control arm, or from `--isolate-ranks` and `--isolate-warps`) and stops at the first that fails. A consumer that reaches its wait two phases late waits on the parity of a phase that may never complete, so isolate counts a hang ended by CUTracer's kernel timeout as a reproduction (`--hang-is-bug`, with `--kernel-timeout-s 60`) and reports it as `"symptom": "hang"`. By default each target gets one 1 ms delay: a clean kernel with W warps per CTA and clusters of C CTAs costs W × C attempts.

```bash
cutracer stress --strategy isolate -k my_kernel -o /tmp/iso -- ./oracle.sh
# Reproduced (hang): delay_ns=1000000 target=cluster rank 1, warp 4 rate=0.20 -> /tmp/iso/triggering_config.json
# Replay: cutracer trace ... --delay-mode cluster_fixed --delay-cluster-cta-id 1 --delay-warp-mask 0x10 ... --delay-load-path /tmp/iso/triggering_config.json -- ./oracle.sh
# Reduce: cutracer reduce -c /tmp/iso/triggering_config.json -t ./oracle.sh ... --delay-warp-mask 0x10 ... --hang-is-bug
```

`--strategy bar` isolates named barriers instead of warps (any mode that isolates targets does with `StressConfig.isolate_by="barrier"` and delay patterns that match `BAR.SYNC`, `BAR.ARV` and `BAR.RED`). Its control arm lists the named-barrier instructions its delay patterns match in the launched kernels, and each attempt delays every warp, with a fixed delay, before the points of one barrier only: every instruction that names one immediate barrier id, in every instrumented kernel, or a single instruction whose barrier id is in a register (two sites that read the same register need not name the same barrier). A target delays its barrier's syncs and reductions; its arrives (`BAR.ARV`) only with `MODE.barrier_arrives=true`, because a late arrive slows down the warps that run ahead of the syncing ones. Barriers that some warps only arrive at come first, then register-id sites, then barriers every participant syncs on. A clean kernel costs one attempt per target, which the CLI prints once the control arm has found the targets; barrier 0x0 (`__syncthreads`) counts, so `bar` applies to most kernels that use shared memory, and it is skipped on kernels without a named barrier to delay. `bar.isolate_max_targets` windows a kernel with many register-id sites (each its own target); there is no default cap, so no barrier is skipped unless asked. An immediate id is selected with SASS patterns (opcode and id, for example `BAR.SYNC.DEFER_BLOCKING 0x5`), which the replay and reduce commands carry as `--delay-patterns`. A register-id site, or an id whose patterns could also select another instruction, is replayed from a delay config that turns on only its sites (`--delay-load-path`), which is then the triggering config.

A delay config records which points were on, not how the delay was targeted or how kernels were launched, so `triggering_config` in `stress_report.json` also carries the `target` (for a mode that isolates targets, the target's record, whose fields depend on its kind), the `symptom` and the `replay_args` that restore them. The CLI also prints the `cutracer reduce` command that narrows the config down to the points that matter with the same targeting, launch handling and per-replay timeout as the search (the `trace` options that set them, plus `--timeout` and `--hang-is-bug`). Every mode takes `--kernel-timeout-s` and `--hang-is-bug`. With a kernel timeout, the 15 s no-data watchdog is off, so a kernel that its delays slow down past 15 s is not mistaken for a hung one; one slowed down past the kernel timeout still is (see `bar` above).

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
| `cutracer stress` | Search for a delay config that triggers a bug with the delay modes that `cutracer stress --help` lists (by default every mode but those that run only when named), and report a verdict (see "Searching for a Triggering Config" above) |
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
