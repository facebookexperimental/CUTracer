// BUG: in tile_sum, thread 0 tells the mbarrier to expect the tile's full byte count
// (len*4), but the copy loop issues only whole 256-byte cp.async.bulk chunks. Tile 37 has
// 1040 floats = 4160 bytes, so its last 64 bytes are never copied, the transaction count
// never completes, and all of CTA 37's warps spin forever on the mbarrier try_wait.
// Expected cause: the copy loop drops the tail that expect_tx counts; fix: copy the tail
// (a final chunk of bytes % 256, which is a multiple of 16), or pad tiles to 256 bytes.
#include <cstdint>
#include <cstdio>
#include <vector>

__device__ __forceinline__ uint32_t smem(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}

__global__ void tile_sum(const float* in, const int* off, const int* len, float* out) {
  __shared__ alignas(128) float buf[2048];
  __shared__ alignas(8) uint64_t bar;
  auto t = blockIdx.x;
  int bytes = len[t] * 4;
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared.b64 [%0], 1;" ::"r"(smem(&bar)));
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(smem(&bar)), "r"(bytes));
    const char* src = (const char*)(in + off[t]);
    int c = 0;
    for (; c + 256 <= bytes; c += 256)
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 256, [%2];" ::"r"(
                       smem((char*)buf + c)),
                   "l"(src + c), "r"(smem(&bar))
                   : "memory");
#ifdef FIXED
    if (c < bytes)
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %3, [%2];" ::"r"(
                       smem((char*)buf + c)),
                   "l"(src + c), "r"(smem(&bar)), "r"(bytes - c)
                   : "memory");
#endif
  }
  uint32_t done = 0;
  while (!done)
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared.b64 p, [%1], 0; selp.u32 %0, 1, 0, p; }"
                 : "=r"(done)
                 : "r"(smem(&bar))
                 : "memory");
  float s = 0.f;
  for (auto i = threadIdx.x; i < len[t]; i += blockDim.x) s += buf[i];
  for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
  __shared__ float part[4];
  if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = s;
  __syncthreads();
  if (threadIdx.x == 0) out[t] = part[0] + part[1] + part[2] + part[3];
}

int main() {
  const int tiles = 96;
  std::vector<int> off(tiles), len(tiles);
  int total = 0;
  for (int t = 0; t < tiles; t++) {
    len[t] = 1024 + 128 * (t % 8);
    if (t == 37) len[t] = 1040;
    off[t] = total;
    total += len[t];
  }
  float *in, *out;
  int *doff, *dlen;
  cudaMalloc(&in, total * 4);
  cudaMalloc(&out, tiles * 4);
  cudaMalloc(&doff, tiles * 4);
  cudaMalloc(&dlen, tiles * 4);
  std::vector<float> h(total);
  for (int i = 0; i < total; i++) h[i] = (float)(i % 5);
  cudaMemcpy(in, h.data(), total * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(doff, off.data(), tiles * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(dlen, len.data(), tiles * 4, cudaMemcpyHostToDevice);
  tile_sum<<<tiles, 128>>>(in, doff, dlen, out);
  cudaError_t e = cudaDeviceSynchronize();
  std::vector<float> ho(tiles);
  cudaMemcpy(ho.data(), out, tiles * 4, cudaMemcpyDeviceToHost);
  int bad = 0;
  for (int t = 0; t < tiles; t++) {
    float r = 0.f;
    for (int i = 0; i < len[t]; i++) r += h[off[t] + i];
    bad += ho[t] != r;
  }
  printf("status: %s, wrong tiles: %d\n", cudaGetErrorString(e), bad);
  return e != cudaSuccess || bad;
}
