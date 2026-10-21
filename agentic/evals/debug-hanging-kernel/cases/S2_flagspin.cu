// BUG: in stage_and_consume, the consumer warp spins on a shared ready flag that the producer
// warp sets only when the tile has elements (counts[b] > 0). counts[0] is 0, so CTA 0's
// consumer spins forever (LDS, ISETP, BRA loop).
// Expected cause: ready is set only inside `if (c > 0)`; fix: set ready on every path.
#include <cstdio>

__global__ void stage_and_consume(const int* counts, const float* in, float* out) {
  __shared__ float buf[32];
  __shared__ volatile int ready;
  auto lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (threadIdx.x == 0) ready = 0;
  __syncthreads();
  if (warp == 0) {
    int c = counts[blockIdx.x];
    buf[lane] = 0.f;
    if (c > 0) {
      buf[lane] = lane < c ? in[blockIdx.x * 32 + lane] : 0.f;
      __threadfence_block();
#ifndef FIXED
      if (lane == 0) ready = 1;
#endif
    }
#ifdef FIXED
    __threadfence_block();
    if (lane == 0) ready = 1;
#endif
  } else {
    while (ready == 0) {
    }
    out[blockIdx.x * 32 + lane] = buf[lane] * 2.f;
  }
}

int main() {
  const int blocks = 16;
  int* counts;
  float *in, *out;
  cudaMallocManaged(&counts, blocks * sizeof(int));
  cudaMallocManaged(&in, blocks * 32 * sizeof(float));
  cudaMallocManaged(&out, blocks * 32 * sizeof(float));
  for (int b = 0; b < blocks; b++) counts[b] = (b * 7) % 33;
  for (int i = 0; i < blocks * 32; i++) in[i] = i * 0.5f;
  stage_and_consume<<<blocks, 64>>>(counts, in, out);
  cudaError_t e = cudaDeviceSynchronize();
  int wrong = 0;
  for (int b = 0; b < blocks; b++)
    for (int l = 0; l < 32; l++) wrong += out[b * 32 + l] != (l < counts[b] ? in[b * 32 + l] * 2.f : 0.f);
  printf("status: %s, wrong outputs: %d\n", cudaGetErrorString(e), wrong);
  return e != cudaSuccess || wrong;
}
