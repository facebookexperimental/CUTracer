/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 * See LICENSE file in the root directory for Meta's license terms.
 */

// A two-CTA cluster pipeline with a consumer that never releases ring slots.
//
// The producer warp of cluster rank 0 writes a tile id into slot `i % kSlots`
// of both CTAs and arrives on each CTA's full barrier. In each CTA a releaser
// warp reads the slot and arrives on the empty barrier of rank 0 (count 2); the
// producer waits on it before it reuses the slot. A follower warp in each CTA
// also waits on the full barrier and reads the slot, only to decide whether to
// continue. In rank 0 the follower is on the release path (the releaser waits
// for it), in rank 1 nobody waits for it, so the producer can lap it. A follower
// that falls two phases behind waits on the parity of the phase in flight; once
// the stream has ended that phase never completes and the kernel hangs.
//
// In rank 0 the follower also waits on an operand barrier for every tile, so a
// delay before every wait of the follower warp in both CTAs slows rank 0 twice
// as much as rank 1, and rank 1 never falls behind. Only a delay on the follower
// of rank 1 alone exposes the bug, as with the MMA warp of the peer CTA of a
// 2-CTA GEMM.
//
// Usage: lagging_consumer [--fixed] [--oracle] [--tiles N] [--clusters N]
//   --fixed   rank 1's follower does not enter the loop (the fix)
//   --oracle  exit 0 when a follower read a wrong value and 1 when every read
//             was right (the `cutracer stress` / `cutracer reduce` convention);
//             without it, the exit codes are the other way around
// Exit codes 2 = CUDA error, 77 = the GPU has no thread block clusters.

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int kSlots = 3;
constexpr int kThreads = 96;
constexpr int kClusterSize = 2;
constexpr int kSentinel = -1;

struct SharedState {
  int slot[kSlots];
  alignas(8) uint64_t full[kSlots];
  alignas(8) uint64_t empty[kSlots];
  alignas(8) uint64_t operand[kSlots];
  alignas(8) uint64_t follower_done[kSlots];
};

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
constexpr int kProducerWarp = 0;
constexpr int kReleaserWarp = 1;
constexpr int kFollowerWarp = 2;

__device__ __forceinline__ uint32_t shared_address(const void* pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  return rank;
}

// The address of `pointer`'s shared-memory variable in the CTA with `rank`.
__device__ __forceinline__ uint32_t address_in_rank(const void* pointer, uint32_t rank) {
  uint32_t mapped;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(mapped) : "r"(shared_address(pointer)), "r"(rank));
  return mapped;
}

__device__ __forceinline__ void barrier_init(uint64_t* barrier, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(shared_address(barrier)), "r"(count));
}

__device__ __forceinline__ void barrier_arrive(uint64_t* barrier, uint32_t rank) {
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(address_in_rank(barrier, rank))
               : "memory");
}

__device__ __forceinline__ void barrier_wait(uint64_t* barrier, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred ready;\n"
      "WAIT:\n"
      "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 ready, [%0], %1;\n"
      "@!ready bra WAIT;\n"
      "}\n" ::"r"(shared_address(barrier)),
      "r"(parity)
      : "memory");
}

__device__ __forceinline__ void store_in_rank(int* pointer, uint32_t rank, int value) {
  asm volatile("st.shared::cluster.u32 [%0], %1;" ::"r"(address_in_rank(pointer, rank)), "r"(value) : "memory");
}

__device__ __forceinline__ void cluster_sync() {
  asm volatile(
      "barrier.cluster.arrive.release.aligned;\n"
      "barrier.cluster.wait.acquire.aligned;\n" ::
          : "memory");
}

__device__ void produce(SharedState& state, int total, int tiles) {
  for (int i = 0; i < total; ++i) {
    const int slot = i % kSlots;
    const uint32_t parity = (i / kSlots) & 1;
    if (i >= kSlots) {
      barrier_wait(&state.empty[slot], parity ^ 1);
    }
    const int value = i < tiles ? i : kSentinel;
    for (uint32_t rank = 0; rank < kClusterSize; ++rank) {
      store_in_rank(&state.slot[slot], rank, value);
      barrier_arrive(&state.full[slot], rank);
    }
    barrier_arrive(&state.operand[slot], 0);
  }
}

__device__ void release(SharedState& state, uint32_t rank) {
  for (int i = 0;; ++i) {
    const int slot = i % kSlots;
    const uint32_t parity = (i / kSlots) & 1;
    barrier_wait(&state.full[slot], parity);
    const int value = *reinterpret_cast<volatile int*>(&state.slot[slot]);
    if (rank == 0) {
      barrier_wait(&state.follower_done[slot], parity);
    }
    barrier_arrive(&state.empty[slot], 0);
    if (value == kSentinel) {
      return;
    }
  }
}

// Records every value read into `reads` (at most `total`) and returns how many
// reads there were.
__device__ int follow(SharedState& state, uint32_t rank, int total, int* reads) {
  for (int i = 0;; ++i) {
    const int slot = i % kSlots;
    const uint32_t parity = (i / kSlots) & 1;
    barrier_wait(&state.full[slot], parity);
    if (rank == 0) {
      barrier_wait(&state.operand[slot], parity);
    }
    const int value = *reinterpret_cast<volatile int*>(&state.slot[slot]);
    if (rank == 0) {
      barrier_arrive(&state.follower_done[slot], 0);
    }
    if (i < total) {
      reads[i] = value;
    }
    if (value == kSentinel) {
      return i + 1;
    }
  }
}
#endif

}  // namespace

// Launched with clusters of kClusterSize CTAs (see launch_kernel).
__global__ void __launch_bounds__(kThreads)
    lagging_consumer_kernel(int tiles, int fixed, int* reads, int* read_counts) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  __shared__ SharedState state;
  const uint32_t rank = cluster_rank();
  const auto warp = threadIdx.x / 32;
  const auto lane = threadIdx.x % 32;
  const auto follower = blockIdx.x;  // one follower per CTA
  const int total = tiles + 1;       // the tiles, then the sentinel

  if (threadIdx.x == 0) {
    for (int i = 0; i < kSlots; ++i) {
      barrier_init(&state.full[i], 1);
      barrier_init(&state.empty[i], kClusterSize);
      barrier_init(&state.operand[i], 1);
      barrier_init(&state.follower_done[i], 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  cluster_sync();

  if (lane == 0) {
    if (warp == kProducerWarp && rank == 0) {
      produce(state, total, tiles);
    } else if (warp == kReleaserWarp) {
      release(state, rank);
    } else if (warp == kFollowerWarp && (rank == 0 || !fixed)) {
      read_counts[follower] = follow(state, rank, total, reads + follower * total);
    }
  }
  __syncwarp();
  // No CTA may exit while the other one can still reach its shared memory.
  cluster_sync();
#endif
}

namespace {

struct Options {
  bool fixed = false;
  bool oracle = false;
  int tiles = 11;
  int clusters = 4;
};

bool parse_options(int argc, char** argv, Options& options) {
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--fixed") == 0) {
      options.fixed = true;
    } else if (std::strcmp(argv[i], "--oracle") == 0) {
      options.oracle = true;
    } else if (std::strcmp(argv[i], "--tiles") == 0 && i + 1 < argc) {
      options.tiles = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--clusters") == 0 && i + 1 < argc) {
      options.clusters = std::atoi(argv[++i]);
    } else {
      return false;
    }
  }
  return options.tiles > 0 && options.clusters > 0;
}

cudaError_t launch_kernel(const Options& options, int* reads, int* read_counts) {
  cudaLaunchAttribute cluster{};
  cluster.id = cudaLaunchAttributeClusterDimension;
  cluster.val.clusterDim.x = kClusterSize;
  cluster.val.clusterDim.y = 1;
  cluster.val.clusterDim.z = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(options.clusters * kClusterSize);
  config.blockDim = dim3(kThreads);
  config.attrs = &cluster;
  config.numAttrs = 1;
  return cudaLaunchKernelEx(&config, lagging_consumer_kernel, options.tiles, options.fixed ? 1 : 0, reads, read_counts);
}

bool check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(status));
    return false;
  }
  return true;
}

// Counts the followers whose reads differ from the stream 0, 1, ..., sentinel.
int count_wrong_followers(const Options& options, const std::vector<int>& reads, const std::vector<int>& read_counts) {
  const int total = options.tiles + 1;
  int wrong = 0;
  for (int follower = 0; follower < options.clusters * kClusterSize; ++follower) {
    const bool skipped = options.fixed && follower % kClusterSize == 1;
    bool ok = read_counts[follower] == (skipped ? 0 : total);
    for (int i = 0; ok && !skipped && i < total; ++i) {
      ok = reads[follower * total + i] == (i < options.tiles ? i : kSentinel);
    }
    if (!ok) {
      std::printf("follower %d (cluster rank %d) read the wrong stream: %d reads\n", follower, follower % kClusterSize,
                  read_counts[follower]);
      ++wrong;
    }
  }
  return wrong;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!parse_options(argc, argv, options)) {
    std::fprintf(stderr, "usage: %s [--fixed] [--oracle] [--tiles N] [--clusters N]\n", argv[0]);
    return 2;
  }
  int device = 0;
  cudaDeviceProp properties{};
  if (!check(cudaGetDevice(&device), "cudaGetDevice") ||
      !check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties")) {
    return 2;
  }
  if (properties.major < 9) {
    std::printf("skipped: compute capability %d.%d has no thread block clusters\n", properties.major, properties.minor);
    return 77;
  }

  const int followers = options.clusters * kClusterSize;
  const int total = options.tiles + 1;
  int* reads = nullptr;
  int* read_counts = nullptr;
  if (!check(cudaMalloc(&reads, sizeof(int) * followers * total), "cudaMalloc") ||
      !check(cudaMalloc(&read_counts, sizeof(int) * followers), "cudaMalloc") ||
      !check(cudaMemset(read_counts, 0, sizeof(int) * followers), "cudaMemset")) {
    return 2;
  }
  if (!check(launch_kernel(options, reads, read_counts), "kernel launch") ||
      !check(cudaDeviceSynchronize(), "kernel")) {
    return 2;
  }
  std::vector<int> host_reads(followers * total);
  std::vector<int> host_counts(followers);
  if (!check(cudaMemcpy(host_reads.data(), reads, sizeof(int) * host_reads.size(), cudaMemcpyDeviceToHost),
             "cudaMemcpy") ||
      !check(cudaMemcpy(host_counts.data(), read_counts, sizeof(int) * host_counts.size(), cudaMemcpyDeviceToHost),
             "cudaMemcpy")) {
    return 2;
  }
  const int wrong = count_wrong_followers(options, host_reads, host_counts);
  std::printf("%d of %d followers read the wrong stream\n", wrong, options.fixed ? followers / 2 : followers);
  const bool bug = wrong > 0;
  if (options.oracle) {
    return bug ? 0 : 1;
  }
  return bug ? 1 : 0;
}
