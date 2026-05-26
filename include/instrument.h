/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 * See LICENSE file in the root directory for Meta's license terms.
 */

#ifndef INSTRUMENT_H
#define INSTRUMENT_H

#include <vector>

#include "analysis.h"

/**
 * @brief Instrumentation types for different data collection modes
 */
enum class InstrumentType {
  OPCODE_ONLY,      // Lightweight: only collect opcode information
  REG_TRACE,        // Medium: collect register values
  MEM_ADDR_TRACE,   // Heavy: collect memory access information (address only)
  MEM_VALUE_TRACE,  // Heavy: collect memory access with values
  TMA_TRACE,        // TMA descriptor tracing for Tensor Memory Accelerator instructions
  RANDOM_DELAY,     // Inject random delays on synchronization instructions
};

/**
 * @brief Structure to hold operand information for instrumentation
 *
 * This structure groups operand data needed for instrumentation,
 * making it easy to extend support for additional operand types without
 * changing function signatures.
 */
struct OperandLists {
  std::vector<int> reg_nums;   // Regular register numbers
  std::vector<int> ureg_nums;  // Uniform register numbers
  std::vector<int> desc_urs;   // UR numbers from MEM_DESC operands (TMA descriptor pointers, NVBit 1.8+)
  // Future: add support for other types like pred_nums, generic_vals, etc.
};

/**
 * @brief Insert lightweight opcode-only instrumentation for instruction histogram analysis
 *
 * This is optimized for Proton instruction statistic analysis where only opcode
 * information is needed for histogram generation.
 *
 * @param instr The instruction to instrument
 * @param opcode_id The opcode identifier for this instruction
 * @param ctx_state The context state containing channel information
 */
void instrument_opcode_only(Instr* instr, int opcode_id, CTXstate* ctx_state);

/**
 * @brief Insert register tracing instrumentation
 *
 * Collects register values for detailed register flow analysis.
 *
 * @param instr The instruction to instrument
 * @param opcode_id The opcode identifier for this instruction
 * @param ctx_state The context state containing channel information
 * @param operands Structure containing all operand information (reg, ureg, etc.)
 */
void instrument_register_trace(Instr* instr, int opcode_id, CTXstate* ctx_state, const OperandLists& operands);

/**
 * @brief Insert memory access tracing instrumentation
 *
 * Collects memory access information for memory pattern analysis.
 *
 * @param instr The instruction to instrument
 * @param opcode_id The opcode identifier for this instruction
 * @param ctx_state The context state containing channel information
 * @param mref_idx Memory reference index
 */
void instrument_memory_addr_trace(Instr* instr, int opcode_id, CTXstate* ctx_state, int mref_idx);

/**
 * @brief Insert memory access tracing with value capture instrumentation
 *
 * Collects memory addresses AND values for data flow analysis.
 * Always uses IPOINT_AFTER for consistent timing semantics.
 *
 * @param instr The instruction to instrument
 * @param opcode_id The opcode identifier for this instruction
 * @param ctx_state The context state containing channel information
 * @param mref_idx Memory reference index
 * @param mem_space Memory space type (obtained via instr->getMemorySpace() in cutracer.cu)
 */
void instrument_memory_value_trace(Instr* instr, int opcode_id, CTXstate* ctx_state, int mref_idx, int mem_space);

/**
 * @brief Instruments an instruction to inject a fixed delay.
 *
 * Inserts a call to the `instrument_delay` device function before the
 * instruction. The delay value is a fixed value determined by CUTRACER_DELAY_NS.
 *
 * @param instr The instruction to instrument
 * @param delay_ns Fixed delay in nanoseconds
 */
void instrument_delay_injection(Instr* instr, uint32_t delay_ns);

/**
 * @brief Instruments an instruction to inject a per-thread random delay.
 *
 * Inserts a call to the `instrument_delay_random` device function before the
 * instruction. Each thread gets a random delay in [min_delay_ns, max_delay_ns]
 * using thread-local entropy (threadIdx, blockIdx, clock) to create asymmetric
 * timing skew that better exposes data races.
 *
 * @param instr The instruction to instrument
 * @param min_delay_ns Minimum delay in nanoseconds (0 = no minimum)
 * @param max_delay_ns Maximum delay in nanoseconds
 */
void instrument_random_delay_injection(Instr* instr, uint32_t min_delay_ns, uint32_t max_delay_ns);

/**
 * @brief Instruments an instruction to inject a random delay on one CTA per cluster.
 *
 * Inserts a call to the `instrument_delay_random_cluster` device function before
 * the instruction. Only the CTA selected by `cluster_seed % cluster_size` receives
 * the delay; other CTAs in the cluster skip entirely. This creates inter-CTA timing
 * asymmetry to expose missing cluster-level synchronization.
 *
 * @param instr The instruction to instrument
 * @param min_delay_ns Minimum delay in nanoseconds
 * @param max_delay_ns Maximum delay in nanoseconds
 * @param cluster_seed Host-generated seed for CTA selection (stored in config for replay)
 * @param cta_id_override If >= 0, force this CTA index for every instrumentation
 *        point (overrides cluster_seed-based random selection). -1 = random.
 * @param use_fixed_delay 0 = per-thread random delay in [min, max] within the
 *        selected CTA. 1 = fixed delay = max_delay_ns for all threads in the
 *        selected CTA (cluster targeting orthogonal to delay distribution).
 */
void instrument_cluster_delay_injection(Instr* instr, uint32_t min_delay_ns, uint32_t max_delay_ns,
                                        uint32_t cluster_seed, int32_t cta_id_override, int32_t use_fixed_delay);

/**
 * @brief Instruments an instruction to inject a warp-targeted fixed delay
 *        within a CTA.
 *
 * Inserts a call to the `instrument_delay_warpgroup` device function before
 * the instruction. Only warps whose CTA-local warp id is in the selected set
 * are stalled, and within those warps every thread sleeps the same amount —
 * preserving warp lock-step. Designed to expose intra-CTA warpgroup-vs-
 * warpgroup races (e.g., consumer warpgroup still reading TMEM while
 * destructor warps free it) that per-thread random delays cannot reproduce.
 *
 * Selection precedence: if @p warpgroup_id >= 0, the selected warps are
 * [4*warpgroup_id .. 4*warpgroup_id+3] and @p warp_mask is ignored.
 * Otherwise the selection is exactly @p warp_mask.
 *
 * @param instr        Instruction to instrument.
 * @param delay_ns     Fixed delay in nanoseconds applied to every thread in
 *                     each selected warp.
 * @param warp_mask    Bitmask of CTA-local warp ids (bit i = warp i). Only
 *                     used when @p warpgroup_id < 0.
 * @param warpgroup_id If >= 0, selects warps [4N..4N+3] and overrides mask.
 *                     -1 = use mask.
 */
void instrument_warpgroup_delay_injection(Instr* instr, uint32_t delay_ns, uint32_t warp_mask, int32_t warpgroup_id);

/**
 * @brief SASS instruction patterns for delay injection.
 */
static const std::vector<const char*> DELAY_INJECTION_PATTERNS = {
    "SYNCS.EXCH",                      // mbarrier init
    "SYNCS.PHASECHK.TRANS64.TRYWAIT",  // mbarrier try_wait
    "SYNCS.ARRIVE.TRANS64.RED.A1T0",   // mbarrier arrive
    "UTMASTG",                         // TMA store
    "UTMALDG",                         // TMA load
    "UTMAREDG",                        // TMA store with reduction
    "WARPGROUP.DEPBAR.LE",             // MMA wait
    "UTCQMMA",
    "UTCHMMA",
    "UTCIMMA",
    "UTCOMMA",
    "UTCBAR",  // mbarrier arrive
    // Tensor memory instructions
    "LDTM",
    "LDT",
    "STT",
    "STTM",
    "LD",
    "ST",
};

/**
 * @brief Check if an instruction should have delay injected
 *
 * @param instr The instruction to check
 * @param patterns Vector of SASS substrings to match against
 * @return true if the instruction matches any delay injection pattern
 */
bool shouldInjectDelay(Instr* instr, const std::vector<const char*>& patterns);

// Include internal TMA instrumentation if available, otherwise provide stubs
#if __has_include("fb/instrument_fb.h")
#include "fb/instrument_fb.h"
#else

/**
 * @brief Instrument a TMA instruction for tracing.
 *
 * OSS stub - does nothing.
 *
 * @param instr The instruction to instrument
 * @param opcode_id The opcode identifier
 * @param ctx_state The context state
 * @param ctx The CUDA context
 */
inline void instrument_tma_trace(Instr* instr, int opcode_id, CTXstate* ctx_state, CUcontext ctx) {
  (void)instr;
  (void)opcode_id;
  (void)ctx_state;
  (void)ctx;
}

#endif  // __has_include

#endif /* INSTRUMENT_H */
