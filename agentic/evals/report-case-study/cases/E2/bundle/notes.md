# Notes

1. `tests/test_gemm.py::test_epilogue_bias` fails about 1 run in 300 natively on H100 (`runs/natural_3.7.txt`, `runs/natural_3.9.txt`). The kernel is `gemm_persistent_ws` in `kernels/gemm.py`. Each tile's epilogue goes through one shared-memory staging buffer.
2. `cutracer stress` with the default plan and the test as the oracle reproduces it on both compiler builds we have (`runs/unfixed_stress_3.7.txt`, `runs/unfixed_stress_3.9.txt`).
3. `--confirm` reduced both to one point: the `LDS.128` that reads the staging buffer for the TMA store.
4. The SASS of both builds stores the next tile into the staging buffer (`STS.128`) with no `BAR.SYNC` after the previous tile's `LDS.128` (`sass/unfixed.txt`; the two builds give the same instructions here). A fast warp overwrites the buffer while a slow warp still reads the previous tile.
5. Fix: `tl.debug_barrier()` between the read for the store and the next write (`fix.patch`).
6. Re-ran the same stress on the fixed tree under both builds: 0 of 16 each (`runs/fixed_stress_3.7.txt`, `runs/fixed_stress_3.9.txt`).
