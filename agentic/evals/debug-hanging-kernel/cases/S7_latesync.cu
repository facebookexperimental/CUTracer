// BUG: stream_sums hands `steps` chunks from warp 0 (producer) to warp 1 (consumer) through
// named barrier 1: the producer writes a chunk into a two-slot buffer and arrives
// (`bar.arrive 1, 64`), the consumer syncs (`bar.sync 1, 64`) and reads it. Nothing makes the
// producer wait for the consumer. Its per-chunk work (about 8000 cycles) normally keeps it
// behind, so the program passes natively. If the consumer reaches a sync
// late (preempted or slowed down; `cutracer stress --strategy bar` delays every warp before
// barrier 1's syncs), the producer arrives twice before that sync: barrier 1 completes on
// the producer's two arrivals alone, every later sync of the consumer pairs with the wrong
// arrive, and its last sync waits for an arrival that never comes, so the CTAs hang.
// Expected cause: no back-pressure from the consumer to the producer; fix: the consumer
// arrives at barrier 2 once it has read a chunk, and the producer syncs on barrier 2 before
// it writes the next one.
#include <cstdio>
#include <vector>

__global__ void stream_sums(const float* in, float* out, int steps) {
  __shared__ float slot[2][32];
  auto lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const float* chunks = in + blockIdx.x * steps * 32;
  if (warp == 0) {
    for (int s = 0; s < steps; s++) {
      float v = chunks[s * 32 + lane];
      for (long long t0 = clock64(); clock64() - t0 < 8000;) {
      }
      slot[s & 1][lane] = v;
      asm volatile("bar.arrive 1, 64;" ::: "memory");
#ifdef FIXED
      asm volatile("bar.sync 2, 64;" ::: "memory");
#endif
    }
  } else if (warp == 1) {
    float sum = 0.f;
    for (int s = 0; s < steps; s++) {
      asm volatile("bar.sync 1, 64;" ::: "memory");
      sum += slot[s & 1][lane];
#ifdef FIXED
      asm volatile("bar.arrive 2, 64;" ::: "memory");
#endif
    }
    out[blockIdx.x * 32 + lane] = sum;
  }
}

int main() {
  const int ctas = 16, steps = 64;
  std::vector<float> in(ctas * steps * 32);
  for (size_t i = 0; i < in.size(); i++) in[i] = (float)(i % 5);
  float *din, *dout;
  cudaMalloc(&din, in.size() * 4);
  cudaMalloc(&dout, ctas * 32 * 4);
  cudaMemcpy(din, in.data(), in.size() * 4, cudaMemcpyHostToDevice);
  stream_sums<<<ctas, 64>>>(din, dout, steps);
  cudaError_t e = cudaDeviceSynchronize();
  std::vector<float> o(ctas * 32);
  cudaMemcpy(o.data(), dout, o.size() * 4, cudaMemcpyDeviceToHost);
  int wrong = 0;
  for (int b = 0; b < ctas; b++)
    for (int l = 0; l < 32; l++) {
      float s = 0.f;
      for (int c = 0; c < steps; c++) s += in[(b * steps + c) * 32 + l];
      wrong += o[b * 32 + l] != s;
    }
  printf("stream_sums: %s, wrong outputs: %d\n", cudaGetErrorString(e), wrong);
  return e != cudaSuccess || wrong;
}
