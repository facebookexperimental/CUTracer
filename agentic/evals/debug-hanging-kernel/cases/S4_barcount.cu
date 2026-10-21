// BUG: plan_tiles (host) sets each tile's named-barrier count to 32 threads per 16 rows,
// rounded up: 64 for tiles of up to 32 rows. Tile 7 has 40 rows, so its count is 96. The
// kernel always uses exactly two warps (64 threads) for the handoff, so CTA 7's consumer
// waits forever at BAR.SYNC 1, R (R = 96) while only 64 threads ever arrive.
// Expected cause: plan_tiles' bar_count is not capped at the 64 threads that participate;
// fix: use 64 (the participating threads) for every tile, independent of rows.
#include <cstdio>
#include <vector>

struct Tile {
  int rows;
  int bar_count;
};

__global__ void row_sums(const Tile* tiles, const float* in, float* out) {
  __shared__ float part[32];
  auto lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  Tile t = tiles[blockIdx.x];
  if (warp == 0) {
    float s = 0.f;
    for (int r = 0; r < t.rows; r++) s += in[(blockIdx.x * 64 + r) * 32 + lane];
    part[lane] = s;
    asm volatile("bar.arrive 1, %0;" ::"r"(t.bar_count) : "memory");
  } else if (warp == 1) {
    asm volatile("bar.sync 1, %0;" ::"r"(t.bar_count) : "memory");
    out[blockIdx.x * 32 + lane] = part[lane] / t.rows;
  }
}

static void plan_tiles(std::vector<Tile>& tiles) {
  for (auto& t : tiles) {
#ifdef FIXED
    t.bar_count = 64;
#else
    int c = 32 * ((t.rows + 15) / 16);
    t.bar_count = c < 64 ? 64 : c;
#endif
  }
}

int main() {
  const int ntiles = 24;
  std::vector<Tile> h(ntiles);
  for (int b = 0; b < ntiles; b++) h[b].rows = 8 + (b * 5) % 25;
  h[7].rows = 40;
  plan_tiles(h);
  std::vector<float> in(ntiles * 64 * 32);
  for (size_t i = 0; i < in.size(); i++) in[i] = (float)(i % 7);
  Tile* dt;
  float *din, *dout;
  cudaMalloc(&dt, ntiles * sizeof(Tile));
  cudaMalloc(&din, in.size() * 4);
  cudaMalloc(&dout, ntiles * 32 * 4);
  cudaMemcpy(dt, h.data(), ntiles * sizeof(Tile), cudaMemcpyHostToDevice);
  cudaMemcpy(din, in.data(), in.size() * 4, cudaMemcpyHostToDevice);
  row_sums<<<ntiles, 128>>>(dt, din, dout);
  cudaError_t e = cudaDeviceSynchronize();
  std::vector<float> o(ntiles * 32);
  cudaMemcpy(o.data(), dout, o.size() * 4, cudaMemcpyDeviceToHost);
  int wrong = 0;
  for (int b = 0; b < ntiles; b++)
    for (int l = 0; l < 32; l++) {
      float s = 0.f;
      for (int r = 0; r < h[b].rows; r++) s += in[(b * 64 + r) * 32 + l];
      wrong += o[b * 32 + l] != s / h[b].rows;
    }
  printf("row_sums: %s, wrong outputs: %d\n", cudaGetErrorString(e), wrong);
  return e != cudaSuccess || wrong;
}
