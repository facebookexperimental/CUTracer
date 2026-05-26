/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: MIT AND BSD-3-Clause
 *
 * This source code contains modifications by Meta Platforms, Inc. licensed under MIT,
 * based on original NVIDIA nvbit sample code licensed under BSD-3-Clause.
 * See LICENSE file in the root directory for Meta's license terms.
 * See LICENSE-BSD file in the root directory for NVIDIA's license terms.
 */

#include <cuda_runtime.h>
#include <stdarg.h>
#include <stdint.h>

#include "utils/utils.h"

/* for channel */
#include "common.h"
#include "utils/channel.hpp"

/* Based on NVIDIA NVBit reg_trace example with Meta modifications for
message type, unified register, and kernel launch id support*/
extern "C" __device__ __noinline__ void instrument_reg_val(int pred, int opcode_id, uint64_t pchannel_dev,
                                                           uint64_t kernel_launch_id, uint64_t pc, int32_t num_regs,
                                                           int32_t num_uregs, ...) {
  if (!pred) {
    return;
  }

  int active_mask = __ballot_sync(__activemask(), 1);
  const int laneid = get_laneid();
  const int first_laneid = __ffs(active_mask) - 1;

  reg_info_t ri;

  ri.header.type = MSG_TYPE_REG_INFO;

  int4 cta = get_ctaid();
  ri.cta_id_x = cta.x;
  ri.cta_id_y = cta.y;
  ri.cta_id_z = cta.z;
  ri.warp_id = get_global_warp_id();
  ri.opcode_id = opcode_id;
  ri.num_regs = num_regs;
  ri.num_uregs = num_uregs;
  ri.kernel_launch_id = kernel_launch_id;
  ri.pc = pc;

  if (num_regs || num_uregs) {
    // Initialize variable argument list
    va_list vl;
    va_start(vl, num_uregs);

    // First batch: collect register values
    for (int i = 0; i < num_regs; i++) {
      uint32_t val = va_arg(vl, uint32_t);

      /* collect register values from other threads */
      for (int tid = 0; tid < 32; tid++) {
        ri.reg_vals[tid][i] = __shfl_sync(active_mask, val, tid);
      }
    }
    // Only the first thread in the warp needs to process unified registers
    if (first_laneid == laneid) {
      for (int i = 0; i < num_uregs; i++) {
        ri.ureg_vals[i] = va_arg(vl, uint32_t);
      }
    }

    va_end(vl);
  }

  if (first_laneid == laneid) {
    ChannelDev* channel_dev = (ChannelDev*)pchannel_dev;
    channel_dev->push(&ri, sizeof(reg_info_t));
  }
}

/* Based on NVIDIA NVBit mem_trace example with Meta modifications for message type */
extern "C" __device__ __noinline__ void instrument_mem(int pred, int opcode_id, uint64_t addr,
                                                       uint64_t kernel_launch_id, uint64_t pc, uint64_t pchannel_dev) {
  /* if thread is predicated off, return */
  if (!pred) {
    return;
  }

  int active_mask = __ballot_sync(__activemask(), 1);
  const int laneid = get_laneid();
  const int first_laneid = __ffs(active_mask) - 1;

  mem_addr_access_t ma;

  ma.header.type = MSG_TYPE_MEM_ADDR_ACCESS;

  /* collect memory address information from other threads */
  for (int i = 0; i < 32; i++) {
    ma.addrs[i] = __shfl_sync(active_mask, addr, i);
  }
  ma.kernel_launch_id = kernel_launch_id;
  int4 cta = get_ctaid();
  ma.cta_id_x = cta.x;
  ma.cta_id_y = cta.y;
  ma.cta_id_z = cta.z;
  ma.pc = pc;
  ma.warp_id = get_global_warp_id();
  ma.opcode_id = opcode_id;

  /* first active lane pushes information on the channel */
  if (first_laneid == laneid) {
    ChannelDev* channel_dev = (ChannelDev*)pchannel_dev;
    channel_dev->push(&ma, sizeof(mem_addr_access_t));
  }
}

extern "C" __device__ __noinline__ void instrument_opcode(int pred, int opcode_id, uint64_t pchannel_dev,
                                                          uint64_t kernel_launch_id, uint64_t pc) {
  if (!pred) {
    return;
  }

  int active_mask = __ballot_sync(__activemask(), 1);
  const int laneid = get_laneid();
  const int first_laneid = __ffs(active_mask) - 1;

  opcode_only_t oi;
  oi.header.type = MSG_TYPE_OPCODE_ONLY;

  int4 cta = get_ctaid();
  oi.cta_id_x = cta.x;
  oi.cta_id_y = cta.y;
  oi.cta_id_z = cta.z;
  oi.warp_id = get_global_warp_id();
  oi.opcode_id = opcode_id;
  oi.kernel_launch_id = kernel_launch_id;
  oi.pc = pc;

  if (first_laneid == laneid) {
    ChannelDev* channel_dev = (ChannelDev*)pchannel_dev;
    channel_dev->push(&oi, sizeof(opcode_only_t));
  }
}

/**
 * @brief Helper function to read a 32-bit value from shared memory using byte-level access.
 *
 * Uses __cvta_shared_to_generic to convert shared memory segment address
 * to a generic pointer, then reads bytes individually to handle arbitrary alignment.
 * This avoids "misaligned address" errors that occur when reading uint32_t from
 * non-4-byte-aligned addresses.
 *
 * @param smemAddr The shared memory segment address (not a generic pointer)
 * @return The 32-bit value at the given shared memory address (little-endian)
 */
__device__ __forceinline__ uint32_t loadSmemValue32(uint64_t smemAddr) {
  const auto ptr = static_cast<unsigned char*>(__cvta_shared_to_generic(static_cast<unsigned>(smemAddr)));
  // Read bytes individually and assemble into uint32_t (little-endian)
  uint32_t value = 0;
  value |= static_cast<uint32_t>(ptr[0]);
  value |= static_cast<uint32_t>(ptr[1]) << 8;
  value |= static_cast<uint32_t>(ptr[2]) << 16;
  value |= static_cast<uint32_t>(ptr[3]) << 24;
  return value;
}

/* Memory space constants matching InstrType::MemorySpace */
#define MEM_SPACE_NONE 0
#define MEM_SPACE_GLOBAL 1
#define MEM_SPACE_SHARED 4
#define MEM_SPACE_LOCAL 5

/**
 * @brief Device function to trace memory access with values.
 *
 * This function collects both memory addresses AND values for detailed data flow analysis.
 * It uses IPOINT_AFTER timing to capture values after the memory operation completes.
 *
 * For Global/Local memory: values are read from registers (passed as variadic args)
 * For Shared memory: values are read directly from memory using address space conversion
 *
 * @param pred Guard predicate value
 * @param opcode_id The opcode identifier for this instruction
 * @param addr Memory address accessed by this thread
 * @param access_size Access size in bytes (1, 2, 4, 8, 16)
 * @param mem_space Memory space type (GLOBAL=1, SHARED=4, LOCAL=5)
 * @param is_load 1 for load operations, 0 for store operations
 * @param num_regs Number of register values following (variadic)
 * @param pchannel_dev Pointer to the channel device
 * @param kernel_launch_id Global kernel launch identifier
 * @param pc Program counter (instruction offset)
 * @param ... Variadic register values (num_regs uint32_t values)
 */
extern "C" __device__ __noinline__ void instrument_mem_value(int pred, int opcode_id, uint64_t addr, int access_size,
                                                             int mem_space, int is_load, int num_regs,
                                                             uint64_t pchannel_dev, uint64_t kernel_launch_id,
                                                             uint64_t pc, ...) {
  if (!pred) {
    return;
  }

  int active_mask = __ballot_sync(__activemask(), 1);
  const int laneid = get_laneid();
  const int first_laneid = __ffs(active_mask) - 1;

  mem_value_access_t mv;
  mv.header.type = MSG_TYPE_MEM_VALUE_ACCESS;

  // Fill basic fields
  mv.kernel_launch_id = kernel_launch_id;
  int4 cta = get_ctaid();
  mv.cta_id_x = cta.x;
  mv.cta_id_y = cta.y;
  mv.cta_id_z = cta.z;
  mv.pc = pc;
  mv.warp_id = get_global_warp_id();
  mv.opcode_id = opcode_id;
  mv.mem_space = mem_space;
  mv.is_load = is_load;
  mv.access_size = access_size;

  // Collect addresses from all lanes
  for (int i = 0; i < 32; i++) {
    mv.addrs[i] = __shfl_sync(active_mask, addr, i);
  }

  // Initialize values array to zero
  for (int i = 0; i < 32; i++) {
    for (int j = 0; j < 4; j++) {
      mv.values[i][j] = 0;
    }
  }

  // Collect values based on memory space type
  if (mem_space == MEM_SPACE_SHARED) {
    // Shared memory: use address space conversion to read values directly
    // Read up to 4 32-bit words based on access_size
    int words_to_read = (access_size + 3) / 4;
    if (words_to_read > 4) words_to_read = 4;

    uint32_t my_values[4] = {0, 0, 0, 0};
    if (addr != 0) {
      for (int w = 0; w < words_to_read; w++) {
        my_values[w] = loadSmemValue32(addr + w * 4);
      }
    }

    // Broadcast values from each lane
    for (int i = 0; i < 32; i++) {
      for (int w = 0; w < words_to_read; w++) {
        mv.values[i][w] = __shfl_sync(active_mask, my_values[w], i);
      }
    }
  } else {
    // Global/Local memory: read values from variadic register arguments
    if (num_regs > 0) {
      va_list vl;
      va_start(vl, pc);

      int regs_to_process = num_regs;
      if (regs_to_process > 4) regs_to_process = 4;

      for (int r = 0; r < regs_to_process; r++) {
        uint32_t val = va_arg(vl, uint32_t);
        // Broadcast this register value from all lanes
        for (int tid = 0; tid < 32; tid++) {
          mv.values[tid][r] = __shfl_sync(active_mask, val, tid);
        }
      }

      va_end(vl);
    }
  }

  // First active lane pushes the data to the channel
  if (first_laneid == laneid) {
    ChannelDev* channel_dev = (ChannelDev*)pchannel_dev;
    channel_dev->push(&mv, sizeof(mem_value_access_t));
  }
}

/**
 * @brief Device function to inject delay before any instrumented instruction.
 *
 * Injects a nanosleep delay before an instruction to expose potential race
 * conditions. The delay value is computed on the host during instrumentation,
 * so each instruction receives a unique, fixed delay value.
 *
 * Uses __nanosleep() intrinsic on SM 7.0+ (Volta and later), with a fallback
 * to busy-wait using clock64() on older architectures.
 *
 * @param pred Guard predicate value (from nvbit_add_call_arg_guard_pred_val)
 * @param delay_ns Delay in nanoseconds (passed from host, 0 = no delay)
 */
extern "C" __device__ __noinline__ void instrument_delay(int pred, uint32_t delay_ns) {
  if (!pred) {
    return;
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
  __nanosleep(delay_ns);
#else
  // Fallback: busy-wait using clock64 (approximate 2 cycles per ns)
  uint64_t delay_cycles = delay_ns * 2;
  uint64_t start = clock64();
  while ((clock64() - start) < delay_cycles) {
  }
#endif
}

/**
 * @brief Device function to inject a per-thread random delay.
 *
 * Each thread gets a random delay in [0, max_delay_ns] using thread-local
 * entropy from threadIdx, blockIdx, and clock64(). This creates asymmetric
 * timing between threads/warps, which is critical for exposing data races.
 *
 * The key insight: fixed delays preserve relative thread timing (all threads
 * slow down equally), while random per-thread delays create differential
 * timing skew that forces race conditions to manifest.
 *
 * Uses xorshift32 PRNG seeded with thread-specific entropy for fast,
 * lightweight random number generation on the GPU.
 *
 * @param pred Guard predicate value (from nvbit_add_call_arg_guard_pred_val)
 * @param min_delay_ns Minimum delay in nanoseconds (floor)
 * @param max_delay_ns Maximum delay in nanoseconds (ceiling)
 */
extern "C" __device__ __noinline__ void instrument_delay_random(int pred, uint32_t min_delay_ns,
                                                                uint32_t max_delay_ns) {
  if (!pred) {
    return;
  }

  // Generate per-thread entropy using threadIdx, blockIdx, and clock
  uint32_t seed = (uint32_t)clock() ^ (threadIdx.x * 2654435761u) ^ (threadIdx.y * 2246822519u) ^
                  (threadIdx.z * 3266489917u) ^ (blockIdx.x * 668265263u) ^ (blockIdx.y * 374761393u) ^
                  (blockIdx.z * 1103515245u);

  // xorshift32 for fast PRNG
  seed ^= seed << 13;
  seed ^= seed >> 17;
  seed ^= seed << 5;

  // Map to [min_delay_ns, max_delay_ns]
  uint32_t range = max_delay_ns - min_delay_ns;
  uint32_t delay = min_delay_ns + (range > 0 ? seed % (range + 1) : 0);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
  __nanosleep(delay);
#else
  uint64_t delay_cycles = delay * 2;
  uint64_t start = clock64();
  while ((clock64() - start) < delay_cycles) {
  }
#endif
}

/**
 * @brief Device function to inject a random delay on exactly one CTA within a cluster.
 *
 * Uses cluster_ctaid/cluster_nctaid PTX registers to determine the CTA's
 * position within its cluster. A host-provided seed selects which CTA index
 * in the cluster receives the delay; all other CTAs skip entirely. This
 * creates timing asymmetry between CTAs in the same cluster, exposing
 * missing inter-CTA synchronization.
 *
 * If the cluster has only one CTA (no clustering), the delay is skipped
 * since there is no inter-CTA synchronization to test.
 *
 * Within the selected CTA, each thread still gets a per-thread random
 * delay (same as instrument_delay_random) for additional timing skew.
 *
 * @param pred Guard predicate value (from nvbit_add_call_arg_guard_pred_val)
 * @param min_delay_ns Minimum delay in nanoseconds (floor)
 * @param max_delay_ns Maximum delay in nanoseconds (ceiling)
 * @param cluster_seed Host-generated seed that determines which CTA in the
 *        cluster gets delayed. Stored in delay config JSON for replay.
 */
extern "C" __device__ __noinline__ void instrument_delay_random_cluster(int pred, uint32_t min_delay_ns,
                                                                        uint32_t max_delay_ns, uint32_t cluster_seed,
                                                                        int32_t cta_id_override,
                                                                        int32_t use_fixed_delay) {
  // The %cluster_ctaid / %cluster_nctaid PTX special registers used by
  // get_cluster_ctaid()/get_cluster_nctaid() require sm_90+. On older arches
  // (e.g. sm_80) ptxas rejects them, so compile this whole body into a no-op
  // there. Cluster mode itself is meaningless on pre-Hopper hardware.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if (!pred) {
    return;
  }

  // Determine cluster geometry
  int4 cluster_nctaid = get_cluster_nctaid();
  int total_ctas = cluster_nctaid.x * cluster_nctaid.y * cluster_nctaid.z;

  // No clustering or single-CTA cluster — skip delay
  if (total_ctas <= 1) {
    return;
  }

  // Compute this CTA's linear index within the cluster
  int4 cluster_ctaid = get_cluster_ctaid();
  int my_cta_linear =
      cluster_ctaid.x + cluster_ctaid.y * cluster_nctaid.x + cluster_ctaid.z * cluster_nctaid.x * cluster_nctaid.y;

  // Select which CTA in the cluster gets delayed. Precedence:
  //   1. cta_id_override >= 0 (set host-side via --delay-cluster-cta-id):
  //      target_cta = cta_id_override % total_ctas. ALWAYS WINS — including
  //      in --delay-load-path replay mode. Setting the override during replay
  //      intentionally overrides the recorded cluster_seed for every point,
  //      so replay is no longer bit-identical to the recording. Useful for
  //      A/B-bisecting "did the bug fire because CTA X was delayed?".
  //   2. cluster_seed (host-generated per instrumentation point in record
  //      mode, loaded from config in replay mode):
  //      target_cta = cluster_seed % total_ctas. Deterministic across runs
  //      that share the same delay config AND do not set the override.
  int target_cta = (cta_id_override >= 0) ? (cta_id_override % total_ctas) : (int)(cluster_seed % (uint32_t)total_ctas);

  if (my_cta_linear != target_cta) {
    return;  // Not the selected CTA, no delay
  }

  // Selected CTA: compute delay either as fixed (= max_delay_ns) or per-thread
  // random in [min_delay_ns, max_delay_ns]. Cluster targeting is orthogonal to
  // distribution: cluster + fixed delays the selected CTA's threads in lockstep,
  // cluster + random adds intra-CTA timing skew on top of inter-CTA targeting.
  uint32_t delay;
  if (use_fixed_delay) {
    delay = max_delay_ns;
  } else {
    uint32_t seed = (uint32_t)clock() ^ (threadIdx.x * 2654435761u) ^ (threadIdx.y * 2246822519u) ^
                    (threadIdx.z * 3266489917u) ^ (blockIdx.x * 668265263u) ^ (blockIdx.y * 374761393u) ^
                    (blockIdx.z * 1103515245u);

    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;

    uint32_t range = max_delay_ns - min_delay_ns;
    delay = min_delay_ns + (range > 0 ? seed % (range + 1) : 0);
  }

  // CUDA __nanosleep has an implementation-defined per-call maximum
  // (~1ms in practice on Hopper/Blackwell). For larger requested delays we
  // loop in 1ms chunks so a 50ms or 500ms request actually sleeps that long
  // instead of silently capping at the hardware limit.
  const uint32_t CHUNK_NS = 1000000u;  // 1 ms
  uint32_t remaining = delay;
  while (remaining > CHUNK_NS) {
    __nanosleep(CHUNK_NS);
    remaining -= CHUNK_NS;
  }
  if (remaining > 0) {
    __nanosleep(remaining);
  }
#else
  // Reference args to silence unused-parameter warnings under -Wall.
  (void)pred;
  (void)min_delay_ns;
  (void)max_delay_ns;
  (void)cluster_seed;
  (void)cta_id_override;
  (void)use_fixed_delay;
#endif
}

/**
 * @brief Device function to inject a warp-targeted fixed delay within a CTA.
 *
 * Stalls only the warps whose CTA-local warp id is set in warp_mask. Within a
 * selected warp every thread sleeps the same amount, preserving warp lock-step
 * so the warp stalls as a single scheduling unit. This is what produces the
 * warpgroup-vs-warpgroup timing skew that intra-CTA races (e.g. consumer
 * warpgroup vs. destructor warpgroup) require — per-thread randomness washes
 * out at the warp boundary and cannot reproduce this class of bug.
 *
 * CTA-local warp id is computed from threadIdx and blockDim. We do not use
 * NVBit's get_global_warp_id() (grid-wide) or PTX %warpid (physical slot,
 * unstable across scheduler) — warp targeting is a CTA-local concept and must
 * be based on the logical warp position within the block.
 *
 * Warps whose id is >= 32 (kernels with > 1024 threads/CTA cannot occur on
 * current hardware, but defensively) are silently skipped: the mask is 32-bit
 * and addresses warps 0..31 only.
 *
 * @param pred       Guard predicate value (per-thread, from nvbit_add_call_arg_guard_pred_val).
 * @param delay_ns   Fixed delay in nanoseconds for every thread in each selected warp.
 * @param warp_mask  Bitmask of CTA-local warp ids (bit i set = delay warp i).
 *                   Use 0 to select no warps (no-op).
 */
extern "C" __device__ __noinline__ void instrument_delay_warpgroup(int pred, uint32_t delay_ns, uint32_t warp_mask) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
  if (!pred) {
    return;
  }

  const uint32_t tid_linear = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const uint32_t warp_id = tid_linear >> 5;

  // warp-uniform: all 32 lanes compute the same warp_id, so the branch does
  // not introduce intra-warp divergence. Hit warps stall together.
  const bool hit = (warp_id < 32u) && (((warp_mask >> warp_id) & 1u) != 0u);
  if (!hit) {
    return;
  }

  // CUDA __nanosleep has an implementation-defined per-call maximum
  // (~1ms in practice on Hopper/Blackwell). Loop in 1ms chunks so a 50ms or
  // 500ms request actually sleeps that long instead of silently capping.
  const uint32_t CHUNK_NS = 1000000u;
  uint32_t remaining = delay_ns;
  while (remaining > CHUNK_NS) {
    __nanosleep(CHUNK_NS);
    remaining -= CHUNK_NS;
  }
  if (remaining > 0u) {
    __nanosleep(remaining);
  }
#else
  // Reference args to silence unused-parameter warnings under -Wall.
  (void)pred;
  (void)delay_ns;
  (void)warp_mask;
#endif
}
