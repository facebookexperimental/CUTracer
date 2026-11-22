// Copyright (c) Meta Platforms, Inc. and affiliates.
// Manual semantic counterexample: run the intentionally invalid aligned variant
// with a host timeout. See README.md for the observed behavior and its scope.
// Paired probes: only the PTX .aligned qualifier changes between each pair.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

struct Result {
  int mismatches;
  int first_iteration;
  int observed;
  int expected;
};

static void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "CUDA_ERROR operation=%s error=%s\n", operation, cudaGetErrorString(status));
    std::exit(2);
  }
}

template <bool Aligned, int Id, int Count>
__device__ __forceinline__ void named_sync() {
  if constexpr (Aligned) {
    asm volatile("barrier.cta.sync.aligned %0, %1;" : : "n"(Id), "n"(Count) : "memory");
  } else {
    asm volatile("barrier.cta.sync %0, %1;" : : "n"(Id), "n"(Count) : "memory");
  }
}

__device__ __forceinline__ void record(Result& result, int iteration, int observed, int expected) {
  if (observed != expected) {
    if (result.mismatches == 0) {
      result.first_iteration = iteration;
      result.observed = observed;
      result.expected = expected;
    }
    ++result.mismatches;
  }
}

template <bool Aligned, bool SplitLane>
__global__ void barrier_handoff(Result* output, int iterations, unsigned delay_ns) {
  constexpr int width = SplitLane ? 16 : 32;
  constexpr int count = width * 2;
  __shared__ volatile int payload[count];
  const auto tid = threadIdx.x;
  payload[tid] = 0;
  __syncthreads();
  Result result{0, -1, 0, 0};

  // Both roles write their own slots, then read the other role's slots.
  // The second barrier prevents overwriting values before peers finish reading.
  // Lane splitting deliberately introduces intra-warp divergence. It is a
  // semantic counterexample, not a minimized reproduction of the whole-warp
  // role split in the original FA4 investigation.
  if (tid < width) {
    for (int iteration = 0; iteration < iterations; ++iteration) {
      if ((iteration & 1) == 0) {
        __nanosleep(delay_ns);
      }
      payload[tid] = iteration * 4096 + tid + 17;
      named_sync<Aligned, 1, count>();
      const int peer = tid + width;
      record(result, iteration, payload[peer], iteration * 4096 + peer + 17);
      named_sync<Aligned, 2, count>();
    }
  } else {
    for (int iteration = 0; iteration < iterations; ++iteration) {
      if ((iteration & 1) != 0) {
        __nanosleep(delay_ns);
      }
      payload[tid] = iteration * 4096 + tid + 17;
      named_sync<Aligned, 1, count>();
      const int peer = tid - width;
      record(result, iteration, payload[peer], iteration * 4096 + peer + 17);
      named_sync<Aligned, 2, count>();
    }
  }
  output[blockIdx.x * count + tid] = result;
}

int main(int argc, char** argv) {
  if (argc < 3 || argc > 6 || (std::strcmp(argv[1], "warp") && std::strcmp(argv[1], "lane")) ||
      (std::strcmp(argv[2], "aligned") && std::strcmp(argv[2], "unaligned"))) {
    std::fprintf(stderr, "usage: %s warp|lane aligned|unaligned [delay_ns] [iterations] [blocks]\n", argv[0]);
    return 2;
  }
  const bool split_lane = std::strcmp(argv[1], "lane") == 0;
  const bool aligned = std::strcmp(argv[2], "aligned") == 0;
  const unsigned delay_ns = argc > 3 ? std::strtoul(argv[3], nullptr, 10) : 10000;
  const int iterations = argc > 4 ? std::atoi(argv[4]) : 32;
  const int blocks = argc > 5 ? std::atoi(argv[5]) : 16;
  if (delay_ns > 1000000 || iterations < 1 || iterations > 1000 || blocks < 1 || blocks > 256) {
    return 2;
  }
  const int threads = split_lane ? 32 : 64;
  const int size = blocks * threads;
  Result* device = nullptr;
  check(cudaMalloc(&device, size * sizeof(Result)), "allocate");
  if (split_lane && aligned) {
    barrier_handoff<true, true><<<blocks, threads>>>(device, iterations, delay_ns);
  } else if (split_lane) {
    barrier_handoff<false, true><<<blocks, threads>>>(device, iterations, delay_ns);
  } else if (aligned) {
    barrier_handoff<true, false><<<blocks, threads>>>(device, iterations, delay_ns);
  } else {
    barrier_handoff<false, false><<<blocks, threads>>>(device, iterations, delay_ns);
  }
  check(cudaGetLastError(), "launch");
  std::printf(
      "LAUNCHED role=%s alignment=%s blocks=%d threads=%d iterations=%d delay_ns=%u; entering cudaDeviceSynchronize\n",
      argv[1], argv[2], blocks, threads, iterations, delay_ns);
  std::fflush(stdout);
  check(cudaDeviceSynchronize(), "synchronize");
  std::vector<Result> host(size);
  check(cudaMemcpy(host.data(), device, size * sizeof(Result), cudaMemcpyDeviceToHost), "copy");
  check(cudaFree(device), "free");
  int bad_threads = 0;
  int mismatches = 0;
  for (int tid = 0; tid < size; ++tid) {
    if (host[tid].mismatches) {
      if (bad_threads == 0) {
        std::printf("FIRST_MISMATCH thread=%d iteration=%d observed=%d expected=%d\n", tid, host[tid].first_iteration,
                    host[tid].observed, host[tid].expected);
      }
      ++bad_threads;
      mismatches += host[tid].mismatches;
    }
  }
  std::printf(
      "RESULT role=%s alignment=%s blocks=%d threads=%d iterations=%d "
      "delay_ns=%u bad_threads=%d mismatches=%d\n",
      argv[1], argv[2], blocks, threads, iterations, delay_ns, bad_threads, mismatches);
  return mismatches ? 1 : 0;
}
