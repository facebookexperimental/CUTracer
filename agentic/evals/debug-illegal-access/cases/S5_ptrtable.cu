// BUG: the host copies only nbatch-1 entries of the pointer table, so the last
// batch's pointer is whatever cudaMalloc left there (0 after the memset) -> null deref.
// Expected cause: host cudaMemcpy of the pointer table copies (nbatch - 1) entries.
// Built without -lineinfo.
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

__global__ void batched_axpy(float* const* xs, float* const* ys, float a, int n) {
  const float* x = xs[blockIdx.y];
  float* y = ys[blockIdx.y];
  // patternlint-disable-next-line cuda-kernel-indices-are-unsigned
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] += a * x[i];
}

int main() {
  const int nb = 24, n = 1 << 16;
  std::vector<float*> hx(nb), hy(nb);
  for (int b = 0; b < nb; ++b) {
    cudaMalloc(&hx[b], n * 4);
    cudaMalloc(&hy[b], n * 4);
    cudaMemset(hx[b], 0, n * 4);
    cudaMemset(hy[b], 0, n * 4);
  }
  float **dx, **dy;
  cudaMalloc(&dx, nb * sizeof(float*));
  cudaMalloc(&dy, nb * sizeof(float*));
  cudaMemset(dx, 0, nb * sizeof(float*));
  cudaMemset(dy, 0, nb * sizeof(float*));
  cudaMemcpy(dx, hx.data(), nb * sizeof(float*), cudaMemcpyHostToDevice);
  cudaMemcpy(dy, hy.data(), (nb - 1) * sizeof(float*), cudaMemcpyHostToDevice);
  batched_axpy<<<dim3(n / 256, nb), 256>>>(dx, dy, 2.0f, n);
  cudaError_t e = cudaDeviceSynchronize();
  printf("batched axpy: %s\n", cudaGetErrorString(e));
  return e == cudaSuccess ? 0 : 1;
}
