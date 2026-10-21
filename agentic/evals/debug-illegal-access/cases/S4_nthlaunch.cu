// BUG: layer 7 of 12 is launched with the wrong buffer length (hidden * 4 instead of
// hidden), so its gather reads past the end of the 'emb' table. The error is reported
// only at the final sync, after 12 launches of the same kernel and other kernels.
// Expected cause: launch 7 of gather_rows receives a table row count that is too large
// (the host passes the wrong size for that layer); index i*ld is out of the table.
#include <cuda_runtime.h>

#include <cstdio>

__global__ void gather_rows(const float* table, const int* idx, int nidx, int ld, float* out) {
  // patternlint-disable-next-line cuda-kernel-indices-are-unsigned
  int r = blockIdx.x, c = threadIdx.x;
  if (r < nidx) out[r * ld + c] = table[(size_t)idx[r] * ld + c];
}

__global__ void make_idx(int* idx, int n, int rows) {
  // patternlint-disable-next-line cuda-kernel-indices-are-unsigned
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) idx[i] = (int)((i * 7919u) % rows);
}

__global__ void relu(float* x, int n) {
  // patternlint-disable-next-line cuda-kernel-indices-are-unsigned
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = fmaxf(x[i], 0.f);
}

int main() {
  const int L = 12, ld = 256, nidx = 4096;
  int rows[L];
  float* table[L];
  int* idx;
  float* out;
  cudaMalloc(&idx, nidx * 4);
  cudaMalloc(&out, (size_t)nidx * ld * 4);
  for (int l = 0; l < L; ++l) {
    rows[l] = 1024 + 512 * l;
    cudaMalloc(&table[l], (size_t)rows[l] * ld * 4);
    cudaMemset(table[l], 0, (size_t)rows[l] * ld * 4);
  }
  for (int l = 0; l < L; ++l) {
    int r = (l == 7) ? rows[l] * 64 : rows[l];
    make_idx<<<(nidx + 255) / 256, 256>>>(idx, nidx, r);
    gather_rows<<<nidx, ld>>>(table[l], idx, nidx, ld, out);
    relu<<<(nidx * ld + 255) / 256, 256>>>(out, nidx * ld);
  }
  float* h = new float[16];
  cudaMemcpy(h, out, 64, cudaMemcpyDeviceToHost);
  cudaError_t e = cudaGetLastError();
  printf("forward done: %s\n", cudaGetErrorString(e));
  return e == cudaSuccess ? 0 : 1;
}
