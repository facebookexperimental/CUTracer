// BUG: build_offsets stores byte offsets (prefix * sizeof(float)) but scatter
// uses them as element indices; buckets past ~1/4 of the output write out of bounds.
// Expected cause: build_offsets (the producer kernel) scales by 4; scatter is correct.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#define NB 4096
#define PER 4096

__global__ void count_keys(const int* keys, int n, int* counts) {
  // patternlint-disable-next-line cuda-kernel-indices-are-unsigned
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(&counts[keys[i]], 1);
}

__global__ void build_offsets(const int* counts, int* offs) {
  __shared__ int s[NB];
  // patternlint-disable-next-line cuda-kernel-indices-are-unsigned
  for (int b = threadIdx.x; b < NB; b += blockDim.x) s[b] = counts[b];
  __syncthreads();
  if (threadIdx.x == 0) {
    int run = 0;
    for (int b = 0; b < NB; ++b) {
      int c = s[b];
      s[b] = run * (int)sizeof(float);
      run += c;
    }
  }
  __syncthreads();
  // patternlint-disable-next-line cuda-kernel-indices-are-unsigned
  for (int b = threadIdx.x; b < NB; b += blockDim.x) offs[b] = s[b];
}

__global__ void scatter(const int* keys, const float* vals, int n, const int* offs, int* cursor, float* out) {
  // patternlint-disable-next-line cuda-kernel-indices-are-unsigned
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int b = keys[i];
  int r = atomicAdd(&cursor[b], 1);
  out[offs[b] + r] = vals[i];
}

int main() {
  int n = NB * PER;
  int* hk = (int*)malloc(n * sizeof(int));
  float* hv = (float*)malloc(n * sizeof(float));
  for (int i = 0; i < n; ++i) {
    hk[i] = (i * 2654435761u) % NB;
    hv[i] = (float)i;
  }
  int *k, *counts, *offs, *cursor;
  float *v, *out;
  cudaMalloc(&k, n * 4);
  cudaMalloc(&v, n * 4);
  cudaMalloc(&out, n * 4);
  cudaMalloc(&counts, NB * 4);
  cudaMalloc(&offs, NB * 4);
  cudaMalloc(&cursor, NB * 4);
  cudaMemcpy(k, hk, n * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(v, hv, n * 4, cudaMemcpyHostToDevice);
  cudaMemset(counts, 0, NB * 4);
  cudaMemset(cursor, 0, NB * 4);
  count_keys<<<(n + 255) / 256, 256>>>(k, n, counts);
  build_offsets<<<1, 1024>>>(counts, offs);
  scatter<<<(n + 255) / 256, 256>>>(k, v, n, offs, cursor, out);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    printf("bucket sort failed: %s\n", cudaGetErrorString(e));
    return 1;
  }
  printf("bucket sort ok\n");
  return 0;
}
