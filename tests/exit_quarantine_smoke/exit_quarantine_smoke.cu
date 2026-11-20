/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace cg = cooperative_groups;

static void check_cuda(cudaError_t error) {
  if (error != cudaSuccess) {
    std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
    std::exit(2);
  }
}

__device__ unsigned long long global_time() {
  unsigned long long value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__device__ void delay_at_least(unsigned long long ns) {
  const auto started = global_time();
  while (global_time() - started < ns) {
    __nanosleep(100);
  }
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
__device__ unsigned int shared_address(const void* pointer) {
  return static_cast<unsigned int>(__cvta_generic_to_shared(pointer));
}

__device__ unsigned int rank_address(unsigned int address, unsigned int rank) {
  unsigned int mapped;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(mapped) : "r"(address), "r"(rank));
  return mapped;
}

__device__ bool completed_phase(unsigned int barrier) {
  unsigned int ready;
  asm volatile(
      "{ .reg .pred p; mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0; "
      "selp.u32 %0, 1, 0, p; }"
      : "=r"(ready)
      : "r"(barrier)
      : "memory");
  return ready != 0;
}
#endif

__global__ void exit_quarantine_basic(int* result) {
  __shared__ volatile unsigned long long words[8];
  if (threadIdx.x == 0) words[0] = 0xfedcba9801234567ull;
  __syncthreads();
  if (threadIdx.x < 32) return;
  delay_at_least(20000);
  if (threadIdx.x == 32) result[blockIdx.x] = words[0] == 0xfedcba9801234567ull ? 1 : -1;
}

__global__ void exit_quarantine_mixed(int* result) {
  __shared__ volatile int flag;
  extern __shared__ volatile int dynamic[];
  if (threadIdx.x == 0) {
    flag = 17;
    dynamic[0] = 29;
  }
  __syncthreads();
  if (threadIdx.x < 32) return;
  delay_at_least(20000);
  if (threadIdx.x == 32) result[blockIdx.x] = flag == 17 && dynamic[0] == 29 ? 1 : -1;
}

__global__ void exit_quarantine_four_byte(int* result) {
  __shared__ volatile int flag;
  if (threadIdx.x == 0) flag = 17;
  __syncthreads();
  if (threadIdx.x == 0) result[blockIdx.x] = flag == 17 ? 1 : -1;
}

__global__ void exit_quarantine_fault() {
  if (threadIdx.x == 0) asm volatile("trap;");
}

__global__ void exit_quarantine_dynamic(int* result) {
  extern __shared__ volatile int dynamic[];
  if (threadIdx.x == 0) dynamic[0] = 17;
  __syncthreads();
  if (threadIdx.x == 32) result[blockIdx.x] = dynamic[0] == 17 ? 1 : -1;
}

__device__ __noinline__ void exit_quarantine_selected_helper(volatile unsigned long long* value) {
  *value = 17;
}

__global__ void exit_quarantine_calls_helper(int* result) {
  __shared__ volatile unsigned long long value;
  if (threadIdx.x == 0) exit_quarantine_selected_helper(&value);
  __syncthreads();
  if (threadIdx.x == 32) result[blockIdx.x] = value == 17 ? 1 : -1;
}

__global__ void exit_quarantine_cluster(bool ordered, int* result) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  __shared__ unsigned long long words[8];
  __shared__ __align__(8) unsigned long long barrier;
  const auto cluster = cg::this_cluster();
  const unsigned int rank = cluster.block_rank();
  const unsigned int local_barrier = shared_address(&barrier);
  if (threadIdx.x == 0) {
    words[0] = 0;
    if (rank == 0) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" : : "r"(local_barrier) : "memory");
      asm volatile("mbarrier.arrive.expect_tx.relaxed.cluster.shared::cta.b64 _, [%0], 4;"
                   :
                   : "r"(local_barrier)
                   : "memory");
      asm volatile("fence.mbarrier_init.release.cluster;" : : : "memory");
    }
  }
  __syncthreads();
  cluster.sync();
  if (rank == 1 && threadIdx.x == 0) {
    delay_at_least(20000);
    const unsigned int destination = rank_address(shared_address(words), 0);
    const unsigned int remote_barrier = rank_address(local_barrier, 0);
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                 :
                 : "r"(destination), "r"(0x1234u), "r"(remote_barrier)
                 : "memory");
  }
  if (ordered) {
    if (rank == 0 && threadIdx.x == 0) {
      const auto started = global_time();
      bool ready = false;
      while (!(ready = completed_phase(local_barrier)) && global_time() - started < 50000000ull) {
        __nanosleep(100);
      }
      result[blockIdx.x / 2] = ready ? 1 : -1;
    }
    __syncthreads();
    cluster.sync();
  }
#else
  (void)ordered;
  (void)result;
#endif
}

static cudaError_t launch_cluster(bool ordered, int* result) {
  cudaLaunchAttribute cluster{};
  cluster.id = cudaLaunchAttributeClusterDimension;
  cluster.val.clusterDim.x = 2;
  cluster.val.clusterDim.y = 1;
  cluster.val.clusterDim.z = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(4);
  config.blockDim = dim3(64);
  config.attrs = &cluster;
  config.numAttrs = 1;
  return cudaLaunchKernelEx(&config, exit_quarantine_cluster, ordered, result);
}

static void run_basic(int* result, int threads = 64, int blocks = 1) {
  for (int iteration = 0; iteration < 3; ++iteration) {
    exit_quarantine_basic<<<blocks, threads>>>(result);
    check_cuda(cudaGetLastError());
    check_cuda(cudaDeviceSynchronize());
    int observed[4]{};
    check_cuda(cudaMemcpy(observed, result, blocks * sizeof(int), cudaMemcpyDeviceToHost));
    for (int block = 0; block < blocks; ++block) {
      if (observed[block] != 1) {
        std::fprintf(stderr, "Basic CTA read shared memory poisoned before its last EXIT\n");
        std::exit(3);
      }
    }
  }
}

static void run_shared_tail(int* result, const char* mode) {
  if (std::strcmp(mode, "mixed") == 0) {
    exit_quarantine_mixed<<<2, 48, sizeof(int)>>>(result);
  } else if (std::strcmp(mode, "dynamic") == 0) {
    exit_quarantine_dynamic<<<2, 48, sizeof(int)>>>(result);
  } else if (std::strcmp(mode, "callee") == 0) {
    exit_quarantine_calls_helper<<<2, 48>>>(result);
  } else {
    exit_quarantine_four_byte<<<2, 48>>>(result);
  }
  check_cuda(cudaGetLastError());
  check_cuda(cudaDeviceSynchronize());
  int observed[2]{};
  check_cuda(cudaMemcpy(observed, result, sizeof(observed), cudaMemcpyDeviceToHost));
  if (observed[0] != 1 || observed[1] != 1) std::exit(3);
}

static void run_graph(int* result) {
  run_basic(result);
  cudaGraph_t graph;
  cudaGraphExec_t executable;
  cudaGraphNode_t first, second;
  check_cuda(cudaGraphCreate(&graph, 0));
  void* arguments[] = {&result};
  cudaKernelNodeParams params{};
  params.func = reinterpret_cast<void*>(exit_quarantine_basic);
  params.gridDim = dim3(1);
  params.blockDim = dim3(64);
  params.kernelParams = arguments;
  check_cuda(cudaGraphAddKernelNode(&first, graph, nullptr, 0, &params));
  check_cuda(cudaGraphAddKernelNode(&second, graph, nullptr, 0, &params));
  check_cuda(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  check_cuda(cudaGraphLaunch(executable, nullptr));
  check_cuda(cudaDeviceSynchronize());
  check_cuda(cudaGraphExecDestroy(executable));
  check_cuda(cudaGraphDestroy(graph));
}

int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "basic";
  int device = 0;
  cudaDeviceProp properties{};
  check_cuda(cudaGetDevice(&device));
  check_cuda(cudaGetDeviceProperties(&properties, device));
  if (properties.major < 9) {
    std::fprintf(stderr, "SKIP: exit-quarantine smoke checks require SM90 or newer\n");
    return 77;
  }
  int* result = nullptr;
  check_cuda(cudaMalloc(&result, 4 * sizeof(int)));
  check_cuda(cudaMemset(result, 0, 4 * sizeof(int)));
  if (std::strcmp(mode, "basic") == 0) {
    run_basic(result);
  } else if (std::strcmp(mode, "partial-warp") == 0) {
    run_basic(result, 48);
  } else if (std::strcmp(mode, "multi-cta") == 0) {
    run_basic(result, 64, 4);
  } else if (std::strcmp(mode, "mixed") == 0 || std::strcmp(mode, "four-byte") == 0 ||
             std::strcmp(mode, "dynamic") == 0 || std::strcmp(mode, "callee") == 0) {
    run_shared_tail(result, mode);
  } else if (std::strcmp(mode, "fault") == 0) {
    exit_quarantine_fault<<<1, 48>>>();
    check_cuda(cudaGetLastError());
    check_cuda(cudaDeviceSynchronize());
    std::fprintf(stderr, "Expected the controlled kernel trap to fail\n");
    return 3;
  } else if (std::strcmp(mode, "graph") == 0) {
    run_graph(result);
  } else if (std::strcmp(mode, "late-write") == 0 || std::strcmp(mode, "ordered") == 0) {
    const bool ordered = std::strcmp(mode, "ordered") == 0;
    check_cuda(launch_cluster(ordered, result));
    check_cuda(cudaGetLastError());
    check_cuda(cudaDeviceSynchronize());
    if (ordered) {
      int observed[2]{};
      check_cuda(cudaMemcpy(observed, result, sizeof(observed), cudaMemcpyDeviceToHost));
      if (observed[0] != 1 || observed[1] != 1) return 3;
    }
  } else {
    std::fprintf(stderr, "Expected basic, partial-warp, multi-cta, mixed, four-byte, graph, late-write, or ordered\n");
    return 2;
  }
  check_cuda(cudaFree(result));
  std::printf("PASS %s\n", mode);
  return 0;
}
