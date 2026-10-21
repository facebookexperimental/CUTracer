// BUG: row offset computed in 32-bit int (row * ld) overflows once row * ld > 2^31;
// the wrapped negative offset points ~8 GB below the buffer. Only rows >= 2^31/ld fault.
// Expected cause: int32 overflow in `int base = row * ld`; fix: 64-bit index (size_t).
#include <cuda_runtime.h>

#include <cstdio>

__global__ void row_scale(float* x, const float* s, int rows, int ld, int cols) {
  // patternlint-disable-next-line cuda-kernel-indices-are-unsigned
  int row = blockIdx.x;
  if (row >= rows) return;
  int base = row * ld;
  // patternlint-disable-next-line cuda-kernel-indices-are-unsigned
  for (int c = threadIdx.x; c < cols; c += blockDim.x) x[base + c] *= s[row];
}

int main() {
  int rows = 40000, ld = 65536, cols = 64;  // 2.6e9 elements, 10.5 GB
  size_t n = (size_t)rows * ld;
  float *x, *s;
  if (cudaMalloc(&x, n * sizeof(float)) != cudaSuccess) {
    printf("alloc failed\n");
    return 2;
  }
  cudaMalloc(&s, rows * sizeof(float));
  cudaMemset(x, 0, n * sizeof(float));
  cudaMemset(s, 0, rows * sizeof(float));
  row_scale<<<rows, 64>>>(x, s, rows, ld, cols);
  cudaError_t e = cudaDeviceSynchronize();
  printf("row_scale: %s\n", cudaGetErrorString(e));
  return e == cudaSuccess ? 0 : 1;
}
