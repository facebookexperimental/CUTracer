#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define DC(call)                              \
  do {                                        \
    CUresult e = (call);                      \
    if (e != CUDA_SUCCESS) {                  \
      const char* s = nullptr;                \
      cuGetErrorString(e, &s);                \
      std::fprintf(stderr, "setup: %s\n", s); \
      std::exit(2);                           \
    }                                         \
  } while (0)

#define RC(call)                                                  \
  do {                                                            \
    cudaError_t e = (call);                                       \
    if (e != cudaSuccess) {                                       \
      std::fprintf(stderr, "setup: %s\n", cudaGetErrorString(e)); \
      std::exit(2);                                               \
    }                                                             \
  } while (0)

struct Region {
  uint64_t base;
  size_t granule, bytes;
};

Region region(unsigned pages = 1) {
  RC(cudaFree(nullptr));
  CUdevice dev;
  DC(cuCtxGetDevice(&dev));
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = dev;
  size_t g;
  DC(cuMemGetAllocationGranularity(&g, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  CUdeviceptr reservation;
  DC(cuMemAddressReserve(&reservation, 1ULL << 35, g, 0, 0));
  CUmemGenericAllocationHandle backing;
  size_t bytes = pages * g;
  DC(cuMemCreate(&backing, bytes, &prop, 0));
  CUdeviceptr base = reservation + g;
  DC(cuMemMap(base, bytes, 0, backing, 0));
  CUmemAccessDesc access{};
  access.location = prop.location;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  DC(cuMemSetAccess(base, bytes, &access, 1));
  DC(cuMemsetD32(base, 0, bytes / 4));
  std::printf("buffer=%p bytes=%zu\n", (void*)base, bytes);
  std::fflush(stdout);
  return {base, g, bytes};
}

uint32_t* output(size_t n) {
  uint32_t* p;
  RC(cudaMalloc(&p, n * sizeof(uint32_t)));
  return p;
}

__device__ __forceinline__ uint32_t read32(uint64_t address) {
  uint32_t v;
  asm volatile("ld.global.u32 %0, [%1];" : "=r"(v) : "l"(address) : "memory");
  return v;
}

__device__ __forceinline__ uint64_t read64(uint64_t address) {
  uint64_t v;
  asm volatile("ld.global.u64 %0, [%1];" : "=l"(v) : "l"(address) : "memory");
  return v;
}

int report(cudaError_t e, const char* operation = "status") {
  std::printf("%s: %s\n", operation, cudaGetErrorString(e));
  return e != cudaSuccess;
}

// Bug: refresh_entries increments pointer word 1 instead of generation word 2.
struct alignas(16) Entry {
  uint64_t pointer;
  uint32_t generation, padding;
};

__global__ void refresh_entries(Entry* entries) {
  unsigned i = blockIdx.x;
  if (i == 37) {
    uint64_t address = (uint64_t)entries + 4ULL * (4 * i + 1);
    uint32_t value = read32(address) + 1;
    asm volatile("st.global.u32 [%0], %1;" : : "l"(address), "r"(value) : "memory");
  }
}

__global__ void traverse(const Entry* entries, uint32_t* out) {
  unsigned i = blockIdx.x;
  uint64_t pointer = read64((uint64_t)entries + sizeof(Entry) * i);
  out[i * 32 + threadIdx.x] = read32(pointer + 4ULL * threadIdx.x);
}

int main() {
  Region r = region();
  Entry host[64];
  for (unsigned i = 0; i < 64; ++i) host[i] = {r.base, 0, 0};
  Entry* entries;
  RC(cudaMalloc(&entries, sizeof(host)));
  RC(cudaMemcpy(entries, host, sizeof(host), cudaMemcpyHostToDevice));
  uint32_t* out = output(64 * 32);
  refresh_entries<<<64, 1>>>(entries);
  traverse<<<64, 32>>>(entries, out);
  return report(cudaDeviceSynchronize());
}
