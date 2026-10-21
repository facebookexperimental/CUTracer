// BUG: two_phase is a hand-written grid barrier: every CTA increments a global counter, then
// spins until the counter reaches gridDim.x. The grid (200000 CTAs of 1024 threads) is far
// larger than the GPU holds at once, so the resident CTAs spin forever and the rest never start.
// Expected cause: a grid-wide spin wait on a grid larger than the resident limit; fix: split
// into two kernels (phase 1, phase 2), or a cooperative launch with the grid capped.
#include <cstdio>
#include <vector>

__global__ void phase1(float* data, int n) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] = data[i] * 2.f;
}

__global__ void phase2(float* data, int n) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] = data[i] + 1.f;
}

__global__ void two_phase(int* counter, float* data, int n) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) data[i] *= 2.f;
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    atomicAdd(counter, 1);
  }
  while (atomicAdd(counter, 0) < gridDim.x) {
  }
  if (i < n) data[i] += 1.f;
}

int main() {
  const int threads = 1024, blocks = 200000, n = threads * blocks;
  int* counter;
  float* data;
  cudaMalloc(&counter, sizeof(int));
  cudaMemset(counter, 0, sizeof(int));
  std::vector<float> h(n);
  for (int i = 0; i < n; i++) h[i] = (float)(i % 1000);
  cudaMalloc(&data, (size_t)n * sizeof(float));
  cudaMemcpy(data, h.data(), (size_t)n * sizeof(float), cudaMemcpyHostToDevice);
#ifdef FIXED
  phase1<<<blocks, threads>>>(data, n);
  phase2<<<blocks, threads>>>(data, n);
#else
  two_phase<<<blocks, threads>>>(counter, data, n);
#endif
  cudaError_t e = cudaDeviceSynchronize();
  std::vector<float> o(n);
  cudaMemcpy(o.data(), data, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);
  long wrong = 0;
  for (int i = 0; i < n; i++) wrong += o[i] != h[i] * 2.f + 1.f;
  printf("status: %s, wrong elements: %ld\n", cudaGetErrorString(e), wrong);
  return e != cudaSuccess || wrong;
}
