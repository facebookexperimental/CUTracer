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

#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>

/* Maximum number of register operands tracked per instruction.
 * This is a configurable buffer limit, NOT a hardware constraint.
 * Increase if instructions with more operands need to be traced. */
#define MAX_REG_OPERANDS 16
#define MAX_UREG_OPERANDS 16

/* Message type enum to identify different message types */
enum message_type_t {
  MSG_TYPE_REG_INFO = 0,
  MSG_TYPE_MEM_ADDR_ACCESS = 1,
  MSG_TYPE_OPCODE_ONLY = 2,
  MSG_TYPE_MEM_VALUE_ACCESS = 3,  // Memory access with value tracing
  MSG_TYPE_TMA_ACCESS = 4         // TMA (Tensor Memory Accelerator) access
};

/* Common header for all message types */
struct message_header_t {
  message_type_t type;  // Type of the message
};

/* Based on NVIDIA record_reg_vals example with Meta modifications for message type support and adds CUTracer specific
 * extensions */
struct reg_info_t {
  message_header_t header;  // Common header with type=MSG_TYPE_REG_INFO
  int32_t cta_id_x;
  int32_t cta_id_y;
  int32_t cta_id_z;
  int32_t warp_id;
  int32_t opcode_id;
  int32_t num_regs;
  /* 32 lanes, each thread can store up to MAX_REG_OPERANDS register values */
  uint32_t reg_vals[32][MAX_REG_OPERANDS];

  // CUTracer extensions
  uint64_t kernel_launch_id;              // Global kernel launch id
  uint64_t pc;                            // Instruction byte offset within the kernel (from Instr::getOffset())
  int32_t num_uregs;                      // Number of unified registers
  uint32_t ureg_vals[MAX_UREG_OPERANDS];  // Unified registers shared by all threads in the same warp
  uint32_t active_mask;                   // Warp-level active lane mask from __ballot_sync(__activemask(), 1)
                                          // (after predicate filtering). 0 is invalid at runtime since
                                          // instrument functions return early when no lane is active.
};

/* Based on NVIDIA mem_trace example with Meta modifications for message type support */
struct mem_addr_access_t {
  message_header_t header;  // Common header with type=MSG_TYPE_MEM_ADDR_ACCESS
  uint64_t kernel_launch_id;
  int cta_id_x;
  int cta_id_y;
  int cta_id_z;
  uint64_t pc;  // Instruction byte offset within the kernel (from Instr::getOffset())
  int warp_id;
  int opcode_id;
  uint64_t addrs[32];
  uint32_t active_mask;  // Warp-level active lane mask after predicate filtering; 0 is invalid at runtime.
};

/**
 * @brief Memory access with value tracing structure.
 *
 * This structure captures both memory addresses AND values for detailed
 * data flow analysis. It is larger than mem_addr_access_t (~820 bytes vs ~300 bytes)
 * due to the values array.
 *
 * Used when CUTRACER_INSTRUMENT=mem_value_trace is enabled.
 * Always captured at IPOINT_AFTER for consistent timing semantics.
 */
struct mem_value_access_t {
  message_header_t header;  // type=MSG_TYPE_MEM_VALUE_ACCESS
  uint64_t kernel_launch_id;
  int cta_id_x;
  int cta_id_y;
  int cta_id_z;
  uint64_t pc;  // Instruction byte offset within the kernel (from Instr::getOffset())
  int warp_id;
  int opcode_id;
  int mem_space;           // Memory space: GLOBAL=1, SHARED=4, LOCAL=5 (matches InstrType::MemorySpace)
  int is_load;             // 1=load, 0=store
  int access_size;         // Access size in bytes (1, 2, 4, 8, 16)
  uint64_t addrs[32];      // Memory addresses for each lane
  uint32_t values[32][4];  // Values: [lane][reg_idx], max 128-bit (4x32-bit) per lane
  uint32_t active_mask;    // Warp-level active lane mask after predicate filtering; 0 is invalid at runtime.
};

/**
 * @brief A lightweight data packet for instruction histogram analysis.
 *
 * This structure is sent from the GPU to the CPU when `OPCODE_ONLY`
 * instrumentation is enabled. It contains the minimal information required
 * to identify an instruction and its execution context without the overhead
 * of register or memory data.
 */
struct opcode_only_t {
  message_header_t header;  // Common header with type=MSG_TYPE_OPCODE_ONLY
  uint64_t kernel_launch_id;
  int cta_id_x;
  int cta_id_y;
  int cta_id_z;
  uint64_t pc;  // Instruction byte offset within the kernel (from Instr::getOffset())
  int warp_id;
  int opcode_id;
  uint32_t active_mask;  // Warp-level active lane mask after predicate filtering; 0 is invalid at runtime.
};

/* Host-only C++ structures */
#ifdef __cplusplus
#include <cstdint>
#include <vector>

/**
 * @brief Register indices for CPU-side static mapping.
 *
 * Maps reg_vals/ureg_vals array positions to actual register numbers.
 * Collected at instrumentation time, not transmitted over GPU channel.
 * This avoids runtime overhead and buffer size increase for static data.
 */
struct RegIndices {
  std::vector<uint8_t> reg_indices;   // R register numbers: 0-254 (R0-R254)
  std::vector<uint8_t> ureg_indices;  // UR register numbers (uint8_t; UR0-UR62 + special, e.g. SM100 emits UR63/UR64)
};

#endif /* __cplusplus */

/* Maximum size of the TMA parameter handle buffer (bytes).
 * Mirrors the NVBit reference tool (tools/mem_trace_tma/common.h). */
#define TMA_PARAM_HANDLE_MAX_BYTES 128

/**
 * @brief TMA (Tensor Memory Accelerator) access tracing structure.
 *
 * Captures the TMA parameter handle for UBLKCP/UBLKPF/UBLKRED/UTMALDG/UTMAPF/
 * UTMAREDG/UTMASTG instructions. The handle is a runtime-provided buffer that
 * NVBit hands to the injected device function via
 * nvbit_add_call_arg_tma_param_handle_and_size(). Host-side parsing is done in
 * recv_thread_fun() via nvbit_parse_tma_transfer_info().
 *
 * Enabled via CUTRACER_INSTRUMENT=tma_trace.
 */
struct tma_access_t {
  message_header_t header;  // type=MSG_TYPE_TMA_ACCESS
  uint64_t kernel_launch_id;
  int cta_id_x;
  int cta_id_y;
  int cta_id_z;
  uint64_t pc;  // Instruction byte offset within the kernel (from Instr::getOffset())
  int warp_id;
  int opcode_id;

  // TMA parameter handle (NVBit 1.8): raw bytes captured on the GPU side and
  // parsed on the CPU side via nvbit_parse_tma_transfer_info().
  uint32_t tma_param_size;
  uint8_t tma_param_handle_raw[TMA_PARAM_HANDLE_MAX_BYTES];

  // Deprecated: legacy 128-byte CUtensorMap snapshot from the pre-NVBit-1.8
  // pipeline. Left in the struct so old trace files remain readable; not
  // populated by the current GPU path.
  uint64_t desc_addr;
  uint64_t desc_raw[16];

  uint32_t active_mask;  // Warp-level active lane mask after predicate filtering; 0 is invalid at runtime.
};

#endif /* COMMON_H */
