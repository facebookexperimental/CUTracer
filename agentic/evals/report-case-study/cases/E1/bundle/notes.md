# Notes

1. `tests/test_attn_bwd.py::test_dk` passes natively (100 of 100 runs, `runs/natural.txt`). The kernel is `attn_bwd_ws` in `kernels/attn_bwd.py`, a warp-specialized tcgen05 kernel. Eight compute warps split each score tile by column.
2. Recorded a numerics golden from 3 plain runs and ran `cutracer stress --strategy isolate` with the golden check as the oracle (`runs/unfixed_stress.txt`). 16 of 16 attempts gave a large change in dK and dV (relative L2 0.31 to 0.37); the test's own tolerance passes most of them.
3. `--confirm` reduced the triggering config to one point: the `SYNCS.PHASECHK` wait before the compute warps load the scores.
4. In a `reg_trace` capture, warps 0-7 of each CTA run `LDTM.x32` at 0x2110 and then `STTM.x16` at 0x3100 into the same TMEM columns (the score and probability buffers share storage). There is no barrier between them (`sass/unfixed.txt`). A warp can store over columns that another warp has not loaded yet.
5. Fix: a named barrier over the 256 compute threads before the store (`fix.patch`).
6. Re-ran the same stress command on the fixed tree: 0 of 16 (`runs/fixed_stress.txt`). The fixed SASS has a `BAR.SYNC` between the load and the store (`sass/fixed.txt`). Fix validated: 16/16 -> 0/16.
