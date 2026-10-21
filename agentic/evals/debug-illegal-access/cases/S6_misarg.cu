// BUG: the host passes x + 1 (a float* one element in) to a kernel that loads float4;
// every warp's first 16-byte load is misaligned, so no warp gets past it.
// Expected cause: misaligned base pointer passed by the host (kernel argument), not the
// kernel's index math.
#include <cuda_runtime.h>

#include <cstdio>

__global__ void sum4(const float* __restrict__ x, float* out, int n4) {
  const float4* x4 = reinterpret_cast<const float4*>(x);
  // patternlint-disable-next-line cuda-kernel-indices-are-unsigned
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) {
    float4 v = x4[i];
    out[i] = v.x + v.y + v.z + v.w;
  }
}

int main() {
  const int n = 1 << 22;
  float *x, *out;
  cudaMalloc(&x, (n + 4) * 4);
  cudaMalloc(&out, n / 4 * 4);
  cudaMemset(x, 0, (n + 4) * 4);
  int skip = 1;  // drop the header element
  sum4<<<(n / 4 + 255) / 256, 256>>>(x + skip, out, n / 4);
  cudaError_t e = cudaDeviceSynchronize();
  printf("sum4: %s\n", cudaGetErrorString(e));
  return e == cudaSuccess ? 0 : 1;
}
