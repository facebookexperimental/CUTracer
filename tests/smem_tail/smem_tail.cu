// Copyright (c) Meta Platforms, Inc. and affiliates.
#include <stdio.h>
#include <stdlib.h>

#define WARP_SIZE 32

// Each lane stores and reloads a 16-bit value. Lane 31's two bytes are the last two bytes of the CTA's
// dynamic shared memory, so mem_value_trace must not read past the bytes an access touched.
__global__ void smemTail(unsigned short* out, int smem_bytes) {
  extern __shared__ unsigned short s[];
  const auto idx = smem_bytes / 2 - WARP_SIZE + threadIdx.x;
  s[idx] = static_cast<unsigned short>(threadIdx.x + 1);
  __syncthreads();
  out[threadIdx.x] = s[idx];
}

int main(int argc, char* argv[]) {
  const int smem_bytes = argc > 1 ? atoi(argv[1]) : 1024;
  unsigned short* d_out;
  cudaMalloc(&d_out, WARP_SIZE * sizeof(unsigned short));
  smemTail<<<1, WARP_SIZE, smem_bytes>>>(d_out, smem_bytes);
  const cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "smemTail failed: %s\n", cudaGetErrorString(err));
    return 1;
  }
  unsigned short h_out[WARP_SIZE];
  cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost);
  cudaFree(d_out);
  printf("out[31] = %u (should be %d)\n", h_out[WARP_SIZE - 1], WARP_SIZE);
  return h_out[WARP_SIZE - 1] == WARP_SIZE ? 0 : 1;
}
