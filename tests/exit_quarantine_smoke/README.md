# Exit-quarantine GPU smoke checks

This workload requires SM90+ to run and validates the exit-quarantine runtime with bounded kernels. Its cluster-only device code is architecture-gated, and unsupported GPUs return skip code 77 rather than a false pass. Runtime cluster launch attributes follow the existing `cluster_delay_test` workload pattern, so the target also builds in the default SM80 sanity configuration. Run it through the source-built `cutracer` CLI, not through a manually injected shared library. Use the architecture flag for the GPU (`b200a` on GB200).

```sh
buck2 run -c fbcode.nvcc_arch=b200a fbcode//triton/tools/CUTracer:cutracer -- trace -i random_delay --delay-enable-prob 0 --exit-quarantine-ns 100000 --kernel-timeout-s 5 --no-data-timeout-s 0 --output-dir /tmp/quarantine-basic -- buck2 run -c fbcode.nvcc_arch=b200a fbcode//triton/tools/CUTracer/tests/exit_quarantine_smoke:exit_quarantine_smoke -- basic
```

`basic` runs three direct launches with predicated warp exits and checks that live threads do not read poisoned shared memory. Repeating it with `INSTR_END=1` verifies that quarantine EXIT coverage is independent of the trace/delay instruction interval. Each launch should report one quarantined CTA and no changed words.

`partial-warp` repeats the direct-launch check with 48 threads, and `multi-cta` uses four independent CTAs. `dynamic` checks a dynamic-only shared-memory window; `four-byte` checks an ordinary 4-byte static shared-memory window. These cases should report complete quarantine coverage without changed words. `callee`, selected with `--kernel-filters exit_quarantine_calls_helper`, exercises a kernel containing a device helper call. CUDA may retain the helper inside the root kernel's instruction image, so this run does not establish independently named callee-only filtering.

`mixed` deliberately combines static and dynamic shared memory. It must be rejected with a `FATAL: exit quarantine:` diagnostic because the available extent fields do not establish padding between those regions. An unsupported run is infrastructure/incomplete evidence, never a clean stress sample.

`ordered` uses two-CTA clusters and an acquire-cluster wait for the remote store completion before the closing cluster barrier. It should complete without changed-word reports.

`late-write` deliberately omits the receiving CTA's completion wait, and its peer writes after a short delay. Run this mode only with quarantine enabled: without it the access to an exited CTA is undefined. With quarantine, at least one changed-word report is expected; a clean run alone is not a proof that the bug is absent.

`fault` raises a controlled kernel trap in its own CUDA context. The original CUDA failure must remain visible, with a nonfatal not-quarantined diagnostic rather than `FATAL: exit quarantine:`. This is a negative control and must not be counted as a passing workload.

`graph` first warms a direct launch, then constructs independent graph nodes using the same kernel. Nonzero quarantine must reject graph construction or execution before unsafe hooks run. With quarantine disabled, the graph remains compatible.
