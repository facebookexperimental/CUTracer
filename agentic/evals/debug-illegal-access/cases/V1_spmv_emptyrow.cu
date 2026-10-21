// BUG: write_rowptr stores the legacy empty-row marker 0xFFFFFFFF as row_ptr[r+1] when a
// row has no nonzeros, instead of the running offset. spmv reads it as an end offset, so the
// empty row's loop (and the next row's start) runs past the end of vals/col.
// Only one row is empty (a late, data-dependent row), so only large inputs fail.
// Expected cause: write_rowptr's count==0 branch; fix: store base + inclusive sum for every row.
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>
#define RB 1024

__global__ void block_sums(const uint32_t* cnt, uint32_t n, uint32_t* bsum) {
  __shared__ uint32_t s[RB];
  uint32_t r = blockIdx.x * RB + threadIdx.x;
  s[threadIdx.x] = r < n ? cnt[r] : 0;
  __syncthreads();
  for (int o = RB / 2; o > 0; o >>= 1) {
    if (threadIdx.x < o) s[threadIdx.x] += s[threadIdx.x + o];
    __syncthreads();
  }
  if (threadIdx.x == 0) bsum[blockIdx.x] = s[0];
}

__global__ void scan_blocks(const uint32_t* bsum, uint32_t nb, uint32_t* boff) {
  if (threadIdx.x == 0) {
    uint32_t run = 0;
    for (uint32_t b = 0; b < nb; ++b) {
      boff[b] = run;
      run += bsum[b];
    }
  }
}

__global__ void write_rowptr(const uint32_t* cnt, uint32_t n, const uint32_t* boff, uint32_t* row_ptr) {
  __shared__ uint32_t s[RB];
  uint32_t r = blockIdx.x * RB + threadIdx.x;
  uint32_t c = r < n ? cnt[r] : 0;
  s[threadIdx.x] = c;
  __syncthreads();
  for (int o = 1; o < RB; o <<= 1) {
    uint32_t v = threadIdx.x >= o ? s[threadIdx.x - o] : 0;
    __syncthreads();
    s[threadIdx.x] += v;
    __syncthreads();
  }
  if (r < n) row_ptr[r + 1] = c == 0 ? 0xFFFFFFFFu : boff[blockIdx.x] + s[threadIdx.x];
  if (r == 0) row_ptr[0] = 0;
}

__global__ void spmv(const uint32_t* row_ptr, const uint32_t* col, const float* val, const float* x, float* y,
                     uint32_t n) {
  uint32_t r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= n) return;
  float acc = 0.f;
  for (uint32_t j = row_ptr[r]; j < row_ptr[r + 1]; ++j) acc += val[j] * x[col[j]];
  y[r] = acc;
}

int main() {
  const uint32_t n = 4u << 20;
  std::vector<uint32_t> cnt(n);
  uint64_t s = 0x9E3779B97F4A7C15ull, nnz = 0;
  for (uint32_t r = 0; r < n; ++r) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    cnt[r] = 1 + (uint32_t)(s % 40);
  }
  uint32_t empty = n - 1 - (uint32_t)(s % 200000);  // one late empty row
  cnt[empty] = 0;
  for (uint32_t r = 0; r < n; ++r) nnz += cnt[r];
  uint32_t nb = (n + RB - 1) / RB;
  uint32_t *d_cnt, *d_bsum, *d_boff, *d_rp, *d_col;
  float *d_val, *d_x, *d_y;
  cudaMalloc(&d_cnt, n * 4);
  cudaMalloc(&d_bsum, nb * 4);
  cudaMalloc(&d_boff, nb * 4);
  cudaMalloc(&d_rp, (n + 1) * 4);
  cudaMalloc(&d_col, nnz * 4);
  cudaMalloc(&d_val, nnz * 4);
  cudaMalloc(&d_x, n * 4);
  cudaMalloc(&d_y, n * 4);
  cudaMemcpy(d_cnt, cnt.data(), n * 4, cudaMemcpyHostToDevice);
  cudaMemset(d_col, 0, nnz * 4);
  cudaMemset(d_val, 0, nnz * 4);
  cudaMemset(d_x, 0, n * 4);
  block_sums<<<nb, RB>>>(d_cnt, n, d_bsum);
  scan_blocks<<<1, 32>>>(d_bsum, nb, d_boff);
  write_rowptr<<<nb, RB>>>(d_cnt, n, d_boff, d_rp);
  spmv<<<(n + 255) / 256, 256>>>(d_rp, d_col, d_val, d_x, d_y, n);
  cudaError_t e = cudaDeviceSynchronize();
  printf("spmv: %s\n", cudaGetErrorString(e));
  return e != cudaSuccess;
}
