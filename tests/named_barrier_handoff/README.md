**Named-barrier alignment under divergent control flow**

This standalone CUDA program is a semantic counterexample: the paired kernels differ only in whether their named barriers use `barrier.cta.sync.aligned` or `barrier.cta.sync`. Each thread writes its own shared-memory slot, waits at barrier 1, and checks the value written by its peer in the other role. Barrier 2 prevents the next iteration from overwriting data before peers finish reading it.

`lane` mode deliberately splits a single warp into lanes 0–15 and 16–31, with each half entering a different branch. Intra-warp divergence itself is legal; using aligned barriers at different instructions for the same meeting violates the PTX execution contract. On the tested GB200, the aligned variant hangs during ordinary execution, while the unaligned variant completes with correct values. This is a constructed counterexample, not a minimized reproduction of the original FA4 whole-warp role split and not evidence of a silent numerical failure in FA4.

`warp` mode instead assigns each role a whole warp. Both variants completed numerically on the tested configuration, although synccheck reported the aligned variant's contract violation. Numerical completion does not establish PTX legality. Undefined behavior does not guarantee a hang on every compiler or GPU.

Run the program manually with a host timeout, because the aligned variant deliberately violates the synchronization contract. The observed configuration is GB200 (`sm_100a`), CUDA `nvcc 13.0.88` with `-O3`, and CUDA 13.4 Compute Sanitizer `2026.3.0.0`. Adjust the tool paths and selected GPU for your machine.

Build from this directory:

```bash
/usr/local/cuda-13.0/bin/nvcc -std=c++17 -arch=sm_100a -lineinfo -O3 barrier_handoff.cu -o barrier_handoff
```

The command-line interface is `barrier_handoff warp|lane aligned|unaligned [delay_ns] [iterations] [blocks]`. Defaults are `10000`, `32`, and `16`, respectively. The optional delay alternates between roles on successive iterations; it is deterministic, not randomized.

The smallest tested invocation uses one CTA, one iteration, and zero delay. Run the two commands independently; the aligned command is expected to return 124 from `timeout` on the recorded configuration.

```bash
CUDA_VISIBLE_DEVICES=0 timeout --kill-after=3s 10s ./barrier_handoff lane unaligned 0 1 1
CUDA_VISIBLE_DEVICES=0 timeout --kill-after=3s 10s ./barrier_handoff lane aligned 0 1 1
```

| Mode | Alignment | Ordinary execution on the recorded configuration |
| --- | --- | --- |
| `lane` | `unaligned` | Exit 0; 32 peer-value checks pass |
| `lane` | `aligned` | Launch succeeds; `cudaDeviceSynchronize()` does not return within 10 seconds; timeout exits 124 |
| `warp` | `unaligned` | Exit 0; 64 peer-value checks pass |
| `warp` | `aligned` | Exit 0; 64 peer-value checks pass, despite the aligned-contract violation |

`LAUNCHED` is printed and flushed after the CUDA launch error check and immediately before `cudaDeviceSynchronize()`. A completed run prints `RESULT ... bad_threads=0 mismatches=0`; the observed aligned lane timeout has only the `LAUNCHED` marker. Exit 1 means a value mismatch, and exit 2 means an argument or CUDA API error. A timeout is a failure to complete, not a measured numerical mismatch.

Use the CUDA 13.4 sanitizer explicitly. The unaligned lane case passes synccheck; the numerically completing aligned warp case reports barrier errors and exits 99 with this error-exitcode setting.

```bash
barrier_sanitizer="$HOME/opt/cuda-13.4/bin/compute-sanitizer"
CUDA_VISIBLE_DEVICES=0 timeout --kill-after=3s 30s "$barrier_sanitizer" --tool synccheck --error-exitcode 99 ./barrier_handoff lane unaligned 0 1 1
CUDA_VISIBLE_DEVICES=0 timeout --kill-after=3s 30s "$barrier_sanitizer" --tool synccheck --error-exitcode 99 ./barrier_handoff warp aligned 0 1 1
```

For a larger shared-memory check, use 32 CTAs and 64 iterations with alternating 100,000 ns delays. The unaligned lane variant performs 65,536 peer-value checks and passes racecheck with 0 errors and 0 warnings on the recorded configuration.

```bash
CUDA_VISIBLE_DEVICES=0 timeout --kill-after=3s 30s "$barrier_sanitizer" --tool racecheck --error-exitcode 99 ./barrier_handoff lane unaligned 100000 64 32
```

The paired PTX bodies match after normalizing template-specific symbols, basic-block labels, debug locations, and the `.aligned` qualifier. Their generated SASS differs: the aligned lane specialization includes reconvergent control flow and `WARPSYNC.ALL` inside the branches, while the unaligned specialization uses different collective lowering. No stalled instruction PC was sampled, so this observation does not identify a directly measured stall location. The compiler can use the alignment contract beyond the encoding of the named BAR instruction itself.
