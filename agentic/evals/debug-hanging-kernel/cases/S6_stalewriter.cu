// BUG: each step runs update_active, then handoff. update_active decrements each CTA's
// remaining budget and clears active[b] when it reaches 0. remaining[5] starts at 4, so in
// step 3 (the 4th step, counting from 0) update_active writes active[5] = 0. In handoff the
// producer warp skips its bar.arrive for inactive CTAs, but the consumer warp still waits at
// BAR.SYNC 1, 64, so CTA 5 of the 4th handoff launch hangs. The handoff code is the same in
// every step; the value that breaks it was written by the earlier update_active launch.
// Expected cause: handoff's producer skips the arrive when active[b]==0 while the consumer
// waits unconditionally (active[5] set to 0 by update_active in step 3); fix: arrive on every
// path. The output check needs the inactive CTAs' zeros, so skipping both warps is not a fix.
#include <cstdio>
#include <vector>

__global__ void update_active(int* remaining, int* active) {
  auto b = threadIdx.x;
  if (b < 8) {
    remaining[b] -= 1;
    active[b] = remaining[b] > 0;
  }
}

__global__ void handoff(const int* active, const float* in, float* out, int step) {
  __shared__ float buf[32];
  auto lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int on = active[blockIdx.x];
  if (warp == 0) {
    buf[lane] = on ? in[blockIdx.x * 32 + lane] + step : 0.f;
#ifdef FIXED
    asm volatile("bar.arrive 1, 64;" ::: "memory");
#else
    if (on) asm volatile("bar.arrive 1, 64;" ::: "memory");
#endif
  } else {
    asm volatile("bar.sync 1, 64;" ::: "memory");
    out[(step * 8 + blockIdx.x) * 32 + lane] = buf[lane];
  }
}

int main() {
  const int blocks = 8, steps = 6;
  int *remaining, *active;
  float *in, *out;
  cudaMalloc(&remaining, blocks * 4);
  cudaMalloc(&active, blocks * 4);
  cudaMalloc(&in, blocks * 32 * 4);
  cudaMalloc(&out, steps * blocks * 32 * 4);
  std::vector<int> rem(blocks);
  for (int b = 0; b < blocks; b++) rem[b] = b == 5 ? 4 : 10 + b;
  std::vector<float> hin(blocks * 32);
  for (int i = 0; i < blocks * 32; i++) hin[i] = (float)(i % 11);
  cudaMemcpy(remaining, rem.data(), blocks * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(in, hin.data(), blocks * 32 * 4, cudaMemcpyHostToDevice);
  cudaError_t e = cudaSuccess;
  for (int s = 0; s < steps && e == cudaSuccess; s++) {
    update_active<<<1, 32>>>(remaining, active);
    handoff<<<blocks, 64>>>(active, in, out, s);
    e = cudaDeviceSynchronize();
    printf("step %d: %s\n", s, cudaGetErrorString(e));
  }
  std::vector<float> o(steps * blocks * 32);
  cudaMemcpy(o.data(), out, o.size() * 4, cudaMemcpyDeviceToHost);
  int wrong = 0;
  for (int s = 0; s < steps; s++)
    for (int b = 0; b < blocks; b++)
      for (int l = 0; l < 32; l++) {
        bool on = rem[b] - (s + 1) > 0;
        wrong += o[(s * 8 + b) * 32 + l] != (on ? hin[b * 32 + l] + s : 0.f);
      }
  printf("wrong outputs: %d\n", wrong);
  return e != cudaSuccess || wrong;
}
