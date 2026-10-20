# CK FMHA forward: missing workgroup barrier before the V tile reuses the K buffer in LDS

See [`Investigation.md`](Investigation.md) for the case report. It covers three
blind agent audits with the CUTracer `audit-memory-model` skill, and the
matched validation.

In Composable Kernel's `BlockFmhaPipelineQRKSVSAsync` (FMHA forward), the
last K stage and the first V stage can use the same LDS buffer. Every wave
reads the K tile from that buffer, then the waves store the V tile into it.
Each wave waits for its own reads (`s_waitcnt lgkmcnt(0)`), but nothing joins
the waves between the K reads and the V stores. A fast wave can overwrite K
bytes that a slow wave has not read yet.

The fix is
[ROCm/composable_kernel@fe8b7d0c](https://github.com/ROCm/composable_kernel/commit/fe8b7d0c277f309d2a0f497c35aec3cb1e87e0dd).
It adds `__builtin_amdgcn_s_barrier()` before the V store when the two stages
share a buffer. Its parent, `683865895e`, is the affected state.

The kernel passes its validation at the default schedule: 0 of 500 runs were
invalid in the affected state. [`delay_wave0.patch`](delay_wave0.patch)
delays wave 0 before its K-tail reads, which makes the race happen. With the
delay, the affected state fails validation and the fixed state passes.

## Requirements

- AMD Instinct MI350 (gfx950)
- ROCm 7.0 (`hipcc`, `roc-obj`)
- CMake 3.21 or later, Ninja, Python 3, git

## Set up both states

The barrier only matters when the K tail and the V tile share an LDS buffer,
which depends on the head dimension. The example's default code generation
does not build head dimension 96, so [`optdim96.patch`](optdim96.patch)
restricts it to 96. Apply it in both states.

```bash
EX=$PWD
git clone https://github.com/ROCm/composable_kernel.git ck
git -C ck worktree add ../ck-affected 683865895e
git -C ck worktree add ../ck-fixed fe8b7d0c277f309d2a0f497c35aec3cb1e87e0dd
for s in affected fixed; do
  git -C ck-$s apply "$EX/optdim96.patch"
  cmake -S ck-$s -B build-$s -G Ninja -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc -DGPU_TARGETS=gfx950 \
    -DFMHA_FWD_ENABLE_APIS=fwd -DCMAKE_BUILD_TYPE=Release
  ninja -C build-$s tile_example_fmha_fwd
done
```

## Check the ISA

```bash
for s in affected fixed; do
  echo "== $s"; python3 "$EX/count_barriers.py" build-$s/bin/tile_example_fmha_fwd
done
```

Every bf16 qr_async kernel with the 128x128x32x128x32x96 tile has one more
`s_barrier` in the fixed state. The recorded counts are in
[`evidence/isa_barriers.txt`](evidence/isa_barriers.txt).

## Run without and with the delay

```bash
RUN="bin/tile_example_fmha_fwd -prec=bf16 -b=4096 -h=1 -s=16 -s_k=200 -d=96 -lse=1 -v=2 -warmup=0 -repeat=1"
for s in affected fixed; do
  for i in $(seq 20); do build-$s/$RUN 2>&1 | tail -1 | grep -o 'valid:[yn]'; done | sort | uniq -c
done

for s in affected fixed; do
  git -C ck-$s apply "$EX/delay_wave0.patch"
  ninja -C build-$s tile_example_fmha_fwd
  for i in $(seq 20); do build-$s/$RUN 2>&1 | tail -1 | grep -o 'valid:[yn]'; done | sort | uniq -c
  git -C ck-$s apply -R "$EX/delay_wave0.patch"
done
```

`-v=2` checks the output against a reference computed on the GPU. Without the
delay, both states print `valid:y` in every run. With the delay, the affected
state prints `valid:n` and the fixed state prints `valid:y`. The recorded
results are in [`evidence/run_results.txt`](evidence/run_results.txt).
