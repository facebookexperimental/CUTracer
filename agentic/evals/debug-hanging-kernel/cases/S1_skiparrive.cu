// BUG: in handoff, the producer warp arrives on named barrier 1 only when its CTA's flag is
// nonzero, but the consumer warp always waits there for 64 threads. flags[5] is 0, so CTA 5's
// consumer waits forever at BAR.SYNC 1, 64.
// Expected cause: the conditional bar.arrive; fix: arrive on every path. The output
// check needs CTA 5's zeros, so skipping both sides is not a fix.
#include <cstdio>

__global__ void handoff(const int* flags, float* out) {
  __shared__ float buf[32];
  auto lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (warp == 0) {
    buf[lane] = flags[blockIdx.x] ? lane * 1.5f : 0.f;
#ifdef FIXED
    asm volatile("bar.arrive 1, 64;" ::: "memory");
#else
    if (flags[blockIdx.x]) asm volatile("bar.arrive 1, 64;" ::: "memory");
#endif
  } else {
    asm volatile("bar.sync 1, 64;" ::: "memory");
    out[blockIdx.x * 32 + lane] = buf[lane];
  }
}

int main() {
  const int blocks = 8;
  int* flags;
  float* out;
  cudaMallocManaged(&flags, blocks * sizeof(int));
  cudaMallocManaged(&out, blocks * 32 * sizeof(float));
  for (int b = 0; b < blocks; b++) flags[b] = (b != 5);
  handoff<<<blocks, 64>>>(flags, out);
  cudaError_t e = cudaDeviceSynchronize();
  int wrong = 0;
  for (int b = 0; b < blocks; b++)
    for (int l = 0; l < 32; l++) wrong += out[b * 32 + l] != (flags[b] ? l * 1.5f : 0.f);
  printf("handoff: %s, wrong outputs: %d\n", cudaGetErrorString(e), wrong);
  return e != cudaSuccess || wrong;
}
