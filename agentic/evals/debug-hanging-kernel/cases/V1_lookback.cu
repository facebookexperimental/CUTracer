// BUG: the single-pass scan packs each tile's status into one 32-bit word: flag in bits
// 31..30 (1 = aggregate, 2 = inclusive prefix), value in bits 29..0. The inclusive prefix
// is packed without masking, so once it reaches 2^30 its bit 30 lands in the flag field.
// The first tile whose inclusive prefix is >= 2^30 publishes flag 3, which the look-back
// treats as "not ready", so the next tile spins on that slot forever; every later tile's
// look-back walks back to the same slot and spins there too. 1M-element inputs stay below
// 2^30 and pass.
// With this input, tile 3986 is the first: its inclusive prefix is 1074002055, published as
// 0xc003f887. Which later status-3 slots the waiters stop at depends on timing.
// Expected cause: pack() overflows the 30-bit value field (inclusive prefix >= 2^30 sets
// flag bits to 3); fix: a 64-bit status word (flag in the high half), or mask and widen.
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>
#define TILE 1024
#define THREADS 256
#ifdef FIXED
typedef unsigned long long word_t;

__device__ __forceinline__ word_t pack(unsigned flag, long long v) {
  return ((word_t)flag << 62) | (word_t)v;
}

__device__ __forceinline__ unsigned flag_of(word_t w) {
  return (unsigned)(w >> 62);
}

__device__ __forceinline__ long long value_of(word_t w) {
  return (long long)(w & ((1ull << 62) - 1));
}
#else
typedef unsigned word_t;

__device__ __forceinline__ word_t pack(unsigned flag, long long v) {
  return (flag << 30) | (unsigned)v;
}

__device__ __forceinline__ unsigned flag_of(word_t w) {
  return w >> 30;
}

__device__ __forceinline__ long long value_of(word_t w) {
  return w & 0x3fffffff;
}
#endif
__global__ void scan_tiles(const int* in, long long* out, volatile word_t* status, unsigned* next_tile, int n) {
  __shared__ int tile_id;
  __shared__ long long warp_sums[THREADS / 32];
  __shared__ long long exclusive;
  if (threadIdx.x == 0) tile_id = atomicAdd(next_tile, 1);
  __syncthreads();
  int t = tile_id;
  auto base = t * TILE + threadIdx.x * 4;
  long long v[4], s = 0;
  for (int k = 0; k < 4; k++) {
    v[k] = base + k < n ? in[base + k] : 0;
    s += v[k];
    v[k] = s;
  }
  long long x = s;
  auto lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int o = 1; o < 32; o <<= 1) {
    long long y = __shfl_up_sync(0xffffffff, x, o);
    if (lane >= o) x += y;
  }
  if (lane == 31) warp_sums[warp] = x;
  __syncthreads();
  if (threadIdx.x == 0) {
    long long run = 0;
    for (int w = 0; w < THREADS / 32; w++) {
      long long y = warp_sums[w];
      warp_sums[w] = run;
      run += y;
    }
    long long agg = run, excl = 0;
    if (t == 0) {
      status[0] = pack(2, agg);
    } else {
      status[t] = pack(1, agg);
      __threadfence();
      int p = t - 1;
      while (true) {
        word_t w = status[p];
        unsigned f = flag_of(w);
        if (f == 2) {
          excl += value_of(w);
          break;
        }
        if (f == 1) {
          excl += value_of(w);
          p--;
          continue;
        }
      }
      __threadfence();
      status[t] = pack(2, excl + agg);
    }
    exclusive = excl;
  }
  __syncthreads();
  long long pre = exclusive + warp_sums[warp] + x - s;
  for (int k = 0; k < 4; k++)
    if (base + k < n) out[base + k] = pre + v[k];
}

int main() {
  const int n = 4 << 20, tiles = (n + TILE - 1) / TILE;
  std::vector<int> h(n);
  for (int i = 0; i < n; i++) h[i] = (int)((i * 2654435761u) >> 22) % 600;
  int* din;
  long long* dout;
  word_t* status;
  unsigned* next;
  cudaMalloc(&din, n * 4);
  cudaMalloc(&dout, n * 8);
  cudaMalloc(&status, tiles * sizeof(word_t));
  cudaMalloc(&next, 4);
  cudaMemset(status, 0, tiles * sizeof(word_t));
  cudaMemset(next, 0, 4);
  cudaMemcpy(din, h.data(), n * 4, cudaMemcpyHostToDevice);
  scan_tiles<<<tiles, THREADS>>>(din, dout, status, next, n);
  cudaError_t e = cudaDeviceSynchronize();
  std::vector<long long> o(n);
  cudaMemcpy(o.data(), dout, n * 8, cudaMemcpyDeviceToHost);
  long long r = 0;
  long wrong = 0;
  for (int i = 0; i < n; i++) {
    r += h[i];
    wrong += o[i] != r;
  }
  printf("scan: %s, wrong elements: %ld, total %lld\n", cudaGetErrorString(e), wrong, r);
  return e != cudaSuccess || wrong;
}
