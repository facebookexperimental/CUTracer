// Bug: route 173 uses a 17-bit mask for a 10-bit payload index.
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define RC(x)                                              \
  do {                                                     \
    cudaError_t e = (x);                                   \
    if (e != cudaSuccess) {                                \
      std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); \
      std::exit(2);                                        \
    }                                                      \
  } while (0)
#define DC(x)                          \
  do {                                 \
    CUresult e = (x);                  \
    if (e != CUDA_SUCCESS) {           \
      const char* s = nullptr;         \
      cuGetErrorString(e, &s);         \
      std::fprintf(stderr, "%s\n", s); \
      std::exit(2);                    \
    }                                  \
  } while (0)

struct Region {
  uint64_t base;
};

Region region() {
  RC(cudaFree(nullptr));
  CUdevice dev;
  DC(cuCtxGetDevice(&dev));
  CUmemAllocationProp p{};
  p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  p.location.id = dev;
  size_t g;
  DC(cuMemGetAllocationGranularity(&g, &p, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  CUdeviceptr reserve;
  DC(cuMemAddressReserve(&reserve, 1ULL << 30, g, 0, 0));
  CUmemGenericAllocationHandle backing;
  DC(cuMemCreate(&backing, g, &p, 0));
  CUdeviceptr mapped = reserve + g;
  DC(cuMemMap(mapped, g, 0, backing, 0));
  CUmemAccessDesc access{};
  access.location = p.location;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  DC(cuMemSetAccess(mapped, g, &access, 1));
  DC(cuMemsetD32(mapped, 0, g / 4));
  Region r{mapped + g - 4096};
  std::printf("payload=%p bytes=4096\n", (void*)r.base);
  std::fflush(stdout);
  return r;
}

template <class T>
T* alloc(size_t n) {
  T* p;
  RC(cudaMalloc(&p, n * sizeof(T)));
  return p;
}

__device__ __forceinline__ uint32_t load32(uint64_t address) {
  uint32_t v;
  asm volatile("ld.global.u32 %0, [%1];" : "=r"(v) : "l"(address) : "memory");
  return v;
}

int finish() {
  cudaError_t e = cudaDeviceSynchronize();
  std::printf("download: %s\n", cudaGetErrorString(e));
  return e != cudaSuccess;
}

struct Ticket {
  uint32_t slot, kind, key, spare;
};

constexpr uint32_t N = 1U << 20, target = 0x5a1c3;

__host__ __device__ uint32_t mix(uint32_t x, unsigned k) {
  for (unsigned j = 0; j < 16; ++j) {
    x ^= x >> 13;
    x *= 0x85ebca6bU + 2U * k + 2U * j;
    x ^= x >> 11;
    x += 0x9e3779b9U * (k + j + 1U);
  }
  return x;
}

template <unsigned K>
__device__ __forceinline__ uint32_t route(uint32_t key) {
  uint32_t x = key;
#pragma unroll
  for (unsigned j = 0; j < 16; ++j) {
    x ^= x >> 13;
    x *= 0x85ebca6bU + 2U * K + 2U * j;
    x ^= x >> 11;
    x += 0x9e3779b9U * (K + j + 1U);
  }
  return x & (K == 173 ? 0x1ffffU : 0x3ffU);
}

template <unsigned Lo, unsigned Hi>
__device__ __forceinline__ uint32_t choose(unsigned kind, uint32_t key) {
  if constexpr (Hi - Lo == 1)
    return route<Lo>(key);
  else {
    constexpr unsigned Mid = (Lo + Hi) / 2;
    return kind < Mid ? choose<Lo, Mid>(kind, key) : choose<Mid, Hi>(kind, key);
  }
}

__global__ void produce(Ticket* tickets, uint32_t* indices) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  Ticket t = tickets[i];
  indices[t.slot] = choose<0, 256>(t.kind, t.key);
  tickets[i] = Ticket{};
}

__global__ void consume(uint64_t base, volatile uint32_t* indices, uint32_t slot, uint32_t* out) {
  unsigned lane = threadIdx.x;
  uint32_t index = indices[slot];
  __syncthreads();
  if (lane == 0) indices[slot] = 0;
  __syncthreads();
  out[lane] = load32(base + 4ULL * index);
}

int main() {
  Region r = region();
  std::vector<Ticket> h(N);
  std::vector<uint32_t> perm(N);
  for (unsigned i = 0; i < N; ++i) perm[i] = i;
  uint32_t rng = 0x6d2b79f5U;
  for (unsigned i = N - 1; i > 0; --i) {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    std::swap(perm[i], perm[rng % (i + 1)]);
  }
  for (unsigned i = 0; i < N; ++i) {
    unsigned kind = (i * 73U + 19U) & 255U;
    if (kind == 173) kind = 172;
    h[i] = {perm[i], kind, i * 0x9e3779b9U, 0};
    if (perm[i] == target) {
      h[i].kind = 173;
      h[i].key = 1;
      while ((mix(h[i].key, 173) & 0x1ffffU) < 1024) ++h[i].key;
    }
  }
  Ticket* tickets = alloc<Ticket>(N);
  uint32_t* indices = alloc<uint32_t>(N);
  uint32_t* out = alloc<uint32_t>(32);
  RC(cudaMemcpy(tickets, h.data(), N * sizeof(Ticket), cudaMemcpyHostToDevice));
  for (unsigned i = 0; i < N; ++i) {
    volatile Ticket& t = h[i];
    t.slot = 0;
    t.kind = 0;
    t.key = 0;
    t.spare = 0;
    volatile uint32_t& p = perm[i];
    p = 0;
  }
  produce<<<N / 256, 256>>>(tickets, indices);
  consume<<<1, 32>>>(r.base, indices, target, out);
  return finish();
}
