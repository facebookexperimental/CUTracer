/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: MIT AND BSD-3-Clause
 *
 * This source code contains modifications by Meta Platforms, Inc. licensed
 * under MIT, based on original NVIDIA NVBit sample code licensed under
 * BSD-3-Clause. See LICENSE file in the root directory for Meta's license
 * terms. See LICENSE-BSD file in the root directory for NVIDIA's license terms.
 */

#include <cstdlib>

#include "analysis.h"
#include "env_config.h"
#include "instrument.h"
#include "log.h"
#include "nvbit.h"

/**
 * @brief Convert IPointType to NVBit's ipoint_t
 */
static ipoint_t to_ipoint(IPointType type) {
  return (type == IPointType::BEFORE) ? IPOINT_BEFORE : IPOINT_AFTER;
}

/**
 * @brief Get the IPOINT to use based on configuration
 *
 * Priority:
 * 1. Per-instrument override (from CUTRACER_INSTRUMENT_IPOINT list)
 * 2. Uniform IPOINT (from CUTRACER_INSTRUMENT_IPOINT_UNIFORM)
 * 3. Default IPOINT passed as parameter
 *
 * @param type The instrument type to get IPOINT for
 * @param default_ipoint The default IPOINT to use if no override is set
 * @return The IPOINT to use for instrumentation
 */
static ipoint_t get_ipoint_from_config(InstrumentType type, ipoint_t default_ipoint) {
  // Check per-instrument override (indexed by position in enabled_instrument_types_ordered)
  auto it = instrument_type_to_index.find(type);
  if (it != instrument_type_to_index.end()) {
    int index = it->second;
    if (static_cast<size_t>(index) < ipoint_overrides.size()) {
      IPointType override_val = ipoint_overrides[index];
      if (override_val != IPointType::DEFAULT) {
        return to_ipoint(override_val);
      }
    }
  }

  // Then, check uniform IPOINT
  if (uniform_ipoint != IPointType::DEFAULT) {
    return to_ipoint(uniform_ipoint);
  }

  // Finally, use default
  return default_ipoint;
}

/**
 * @brief Instruments an instruction to record its opcode for lightweight analysis.
 *
 * This function was developed by Meta to support analyses like instruction
 * histograms, where only the instruction's identity is needed, not its operand
 * values.
 *
 * It injects a call to the `instrument_opcode` device function, passing the
 * opcode ID, program counter (PC), and global kernel launch ID.
 */
void instrument_opcode_only(Instr* instr, int opcode_id, CTXstate* ctx_state) {
  /* insert call to the instrumentation function with its arguments */
  nvbit_insert_call(instr, "instrument_opcode", get_ipoint_from_config(InstrumentType::OPCODE_ONLY, IPOINT_BEFORE));
  /* guard predicate value */
  nvbit_add_call_arg_guard_pred_val(instr);
  /* opcode id */
  nvbit_add_call_arg_const_val32(instr, opcode_id);
  /* pass the pointer to the channel on the device */
  nvbit_add_call_arg_const_val64(instr, (uint64_t)ctx_state->channel_dev);
  /* add "space" for kernel function pointer that will be set
   * at launch time (64 bit value at offset 0 of the dynamic
   * arguments). it is used to pass global kernel launch id*/
  nvbit_add_call_arg_launch_val64(instr, 0);
  /* add instruction offset */
  nvbit_add_call_arg_const_val64(instr, instr->getOffset());
}

/**
 * @brief Instruments an instruction to trace the values of its register operands.
 *
 * This function is based on the instrumentation logic from NVIDIA's
 * `record_reg_vals` example (`third_party/nvbit/tools/record_reg_vals/record_reg_vals.cu`).
 * It injects a call to the `instrument_reg_val` device function to capture operand values.
 *
 * Key enhancements by Meta include:
 *  - **Unified Register (UREG) Support**: Added tracing for UREGs, which was
 *    not present in the original example.
 *  - **Enhanced Context**: Passes the global kernel launch ID and the
 *    instruction's program counter (PC) to correlate data more effectively
 *    during analysis.
 *  - **Refactoring**: Encapsulated the logic into this dedicated function,
 *    separating it from the main instruction iteration loop in `cutracer.cu`.
 */
void instrument_register_trace(Instr* instr, int opcode_id, CTXstate* ctx_state, const OperandLists& operands) {
  /* insert call to the instrumentation function with its arguments */
  nvbit_insert_call(instr, "instrument_reg_val", get_ipoint_from_config(InstrumentType::REG_TRACE, IPOINT_BEFORE));
  /* guard predicate value */
  nvbit_add_call_arg_guard_pred_val(instr);
  /* opcode id */
  nvbit_add_call_arg_const_val32(instr, opcode_id);
  /* add pointer to channel_dev*/
  nvbit_add_call_arg_const_val64(instr, (uint64_t)ctx_state->channel_dev);
  /* add "space" for kernel function pointer that will be set
   * at launch time (64 bit value at offset 0 of the dynamic
   * arguments). it is used to pass global kernel launch id*/
  nvbit_add_call_arg_launch_val64(instr, 0);
  /* add instruction offset */
  nvbit_add_call_arg_const_val64(instr, instr->getOffset());
  /* how many register values are passed next */
  nvbit_add_call_arg_const_val32(instr, operands.reg_nums.size());
  nvbit_add_call_arg_const_val32(instr, operands.ureg_nums.size());

  // Pass register values (variadic)
  for (int num : operands.reg_nums) {
    /* last parameter tells it is a variadic parameter passed to
     * the instrument function record_reg_val() */
    nvbit_add_call_arg_reg_val(instr, num, true);
  }
  for (int num : operands.ureg_nums) {
    nvbit_add_call_arg_ureg_val(instr, num, true);
  }
}

/**
 * @brief Instruments a memory instruction to trace memory access details.
 *
 * This function is based on the instrumentation logic from NVIDIA's `mem_trace`
 * example. It injects a call to the `instrument_mem` device function.
 *
 * Meta's enhancements include:
 *  - **Refactoring**: Moving the instrumentation logic from the main loop in
 *    `cutracer.cu` into this modular function.
 *  - **Contextual Information**: Passing the global kernel launch ID and the
 *    instruction's program counter (PC) alongside the memory address. This
 *    allows for more detailed analysis by linking each memory access to a
 *    specific kernel launch and instruction.
 *
 * Note: This function implements `mem_addr_trace` mode which only captures
 * memory addresses, not values. For value tracing, use `mem_value_trace` mode.
 */
void instrument_memory_addr_trace(Instr* instr, int opcode_id, CTXstate* ctx_state, int mref_idx) {
  /* insert call to the instrumentation function with its
   * arguments */
  nvbit_insert_call(instr, "instrument_mem", get_ipoint_from_config(InstrumentType::MEM_ADDR_TRACE, IPOINT_BEFORE));
  /* predicate value */
  nvbit_add_call_arg_guard_pred_val(instr);
  /* opcode id */
  nvbit_add_call_arg_const_val32(instr, opcode_id);
  /* memory reference 64 bit address */
  nvbit_add_call_arg_mref_addr64(instr, mref_idx);
  /* add "space" for kernel function pointer that will be set
   * at launch time (64 bit value at offset 0 of the dynamic
   * arguments). it is used to pass global kernel launch id*/
  nvbit_add_call_arg_launch_val64(instr, 0);
  /* add instruction offset */
  nvbit_add_call_arg_const_val64(instr, instr->getOffset());
  /* add pointer to channel_dev*/
  nvbit_add_call_arg_const_val64(instr, (uint64_t)ctx_state->channel_dev);
}

/**
 * @brief Check if an instruction should have delay injected.
 *
 * @param instr The instruction to check
 * @param patterns Vector of SASS substrings to match against
 * @return true if the instruction's SASS matches any pattern
 */
bool shouldInjectDelay(Instr* instr, const std::vector<const char*>& patterns) {
  const char* sass = instr->getSass();
  if (sass == nullptr) {
    return false;
  }

  for (const char* pattern : patterns) {
    if (strstr(sass, pattern) != nullptr) {
      return true;
    }
  }

  return false;
}

/**
 * @brief Instruments an instruction to inject a fixed delay.
 *
 * Inserts a call to the `instrument_delay` device function before the
 * instruction. The delay value is a fixed value determined by CUTRACER_DELAY_NS.
 *
 * @param instr The instruction to instrument
 * @param delay_ns Fixed delay in nanoseconds
 */
void instrument_delay_injection(Instr* instr, uint32_t delay_ns) {
  /* insert call to the instrumentation function with its arguments */
  nvbit_insert_call(instr, "instrument_delay", get_ipoint_from_config(InstrumentType::RANDOM_DELAY, IPOINT_BEFORE));
  /* guard predicate value */
  nvbit_add_call_arg_guard_pred_val(instr);
  /* delay in nanoseconds */
  nvbit_add_call_arg_const_val32(instr, delay_ns);
}

void instrument_random_delay_injection(Instr* instr, uint32_t min_delay_ns, uint32_t max_delay_ns) {
  /* insert call to the per-thread random delay function */
  nvbit_insert_call(instr, "instrument_delay_random",
                    get_ipoint_from_config(InstrumentType::RANDOM_DELAY, IPOINT_BEFORE));
  /* guard predicate value */
  nvbit_add_call_arg_guard_pred_val(instr);
  /* min delay in nanoseconds */
  nvbit_add_call_arg_const_val32(instr, min_delay_ns);
  /* max delay in nanoseconds */
  nvbit_add_call_arg_const_val32(instr, max_delay_ns);
}

void instrument_cluster_delay_injection(Instr* instr, uint32_t min_delay_ns, uint32_t max_delay_ns,
                                        uint32_t cluster_seed, int32_t cta_id_override, int32_t use_fixed_delay) {
  /* insert call to the cluster-level delay function */
  nvbit_insert_call(instr, "instrument_delay_random_cluster",
                    get_ipoint_from_config(InstrumentType::RANDOM_DELAY, IPOINT_BEFORE));
  /* guard predicate value */
  nvbit_add_call_arg_guard_pred_val(instr);
  /* min delay in nanoseconds */
  nvbit_add_call_arg_const_val32(instr, min_delay_ns);
  /* max delay in nanoseconds */
  nvbit_add_call_arg_const_val32(instr, max_delay_ns);
  /* seed for selecting which CTA in the cluster gets delayed */
  nvbit_add_call_arg_const_val32(instr, cluster_seed);
  /* host-side CTA selection override: -1 = random via seed, >=0 = force this CTA index */
  nvbit_add_call_arg_const_val32(instr, (uint32_t)cta_id_override);
  /* delay distribution: 0 = per-thread random in [min, max], 1 = fixed = max */
  nvbit_add_call_arg_const_val32(instr, (uint32_t)use_fixed_delay);
}

void instrument_warpgroup_delay_injection(Instr* instr, uint32_t delay_ns, uint32_t warp_mask, int32_t warpgroup_id) {
  /* warpgroup_id is sugar: "warpgroup N" expands to warps [4N .. 4N+3].
   * If warpgroup_id >= 0 it wins over warp_mask; the device function only
   * sees a single resolved mask. */
  const uint32_t effective_mask = (warpgroup_id >= 0) ? (0xFu << (warpgroup_id * 4)) : warp_mask;

  nvbit_insert_call(instr, "instrument_delay_warpgroup",
                    get_ipoint_from_config(InstrumentType::RANDOM_DELAY, IPOINT_BEFORE));
  /* guard predicate value */
  nvbit_add_call_arg_guard_pred_val(instr);
  /* fixed delay in nanoseconds */
  nvbit_add_call_arg_const_val32(instr, delay_ns);
  /* bitmask of CTA-local warp ids to delay */
  nvbit_add_call_arg_const_val32(instr, effective_mask);
}

/**
 * @brief Instruments a memory instruction to trace memory access with values.
 *
 * This function instruments memory instructions to capture both addresses AND
 * values for detailed data flow analysis. Unlike instrument_memory_trace() which
 * uses IPOINT_BEFORE, this function defaults to IPOINT_AFTER to capture values after
 * the memory operation completes, but can be overridden via CUTRACER_INSTRUMENT_IPOINT.
 *
 * For Global/Local memory: values are read from registers
 *   - Load instructions: read from destination register (value is in reg after load)
 *   - Store instructions: read from source register (source reg is not modified by store)
 *
 * For Shared memory: values are read directly from memory using address space conversion
 *
 * @param instr The instruction to instrument
 * @param opcode_id The opcode identifier for this instruction
 * @param ctx_state The context state containing channel information
 * @param mref_idx Memory reference index
 * @param mem_space Memory space type (obtained via instr->getMemorySpace() in cutracer.cu)
 */
void instrument_memory_value_trace(Instr* instr, int opcode_id, CTXstate* ctx_state, int mref_idx, int mem_space) {
  bool is_load = instr->isLoad();
  int access_size = instr->getSize();

  // Default to IPOINT_AFTER for value capture (values available after load completes)
  nvbit_insert_call(instr, "instrument_mem_value",
                    get_ipoint_from_config(InstrumentType::MEM_VALUE_TRACE, IPOINT_AFTER));

  // Guard predicate value
  nvbit_add_call_arg_guard_pred_val(instr);
  // Opcode id
  nvbit_add_call_arg_const_val32(instr, opcode_id);
  // Memory reference 64 bit address
  nvbit_add_call_arg_mref_addr64(instr, mref_idx);
  // Access size in bytes
  nvbit_add_call_arg_const_val32(instr, access_size);
  // Memory space type (GLOBAL=1, SHARED=4, LOCAL=5)
  nvbit_add_call_arg_const_val32(instr, mem_space);
  // Is load operation (1=load, 0=store)
  nvbit_add_call_arg_const_val32(instr, is_load ? 1 : 0);

  // For Global/Local memory, we need to pass register values
  // For Shared memory, device function will read directly from memory
  int num_regs = 0;
  std::vector<int> value_reg_nums;

  // InstrType::MemorySpace::SHARED = 4
  if (mem_space != 4) {
    // Find the register operand that holds the value
    // For load: destination register (usually first REG operand)
    // For store: source register (usually last REG operand that's not part of address)
    int num_operands = instr->getNumOperands();
    int regs_needed = (access_size + 3) / 4;  // Number of 32-bit registers needed
    if (regs_needed > 4) regs_needed = 4;     // Cap at 4 (128 bits)

    for (int i = 0; i < num_operands && (int)value_reg_nums.size() < regs_needed; i++) {
      const InstrType::operand_t* op = instr->getOperand(i);
      if (op->type == InstrType::OperandType::REG) {
        // For load: first REG operand is the destination
        // For store: we want the source data register, which is typically at a different position
        // The MREF operand contains address, REG operands are data
        if (is_load) {
          // First REG operand is destination for loads
          for (int reg_idx = 0; reg_idx < regs_needed && (int)value_reg_nums.size() < regs_needed; reg_idx++) {
            value_reg_nums.push_back(op->u.reg.num + reg_idx);
          }
          break;  // Found destination register(s)
        } else {
          // For stores, we want the source data register
          // Skip if this looks like an address register (part of MREF)
          // Typically for stores like STG [R10], R8 - R8 is the source
          // We collect all REG operands that aren't part of address
          for (int reg_idx = 0; reg_idx < regs_needed && (int)value_reg_nums.size() < regs_needed; reg_idx++) {
            value_reg_nums.push_back(op->u.reg.num + reg_idx);
          }
        }
      }
    }
    num_regs = value_reg_nums.size();
  }

  // Number of register values to follow
  nvbit_add_call_arg_const_val32(instr, num_regs);
  // Pointer to channel_dev
  nvbit_add_call_arg_const_val64(instr, (uint64_t)ctx_state->channel_dev);
  // Kernel launch id (set at launch time)
  nvbit_add_call_arg_launch_val64(instr, 0);
  // Instruction offset (PC)
  nvbit_add_call_arg_const_val64(instr, instr->getOffset());

  // Add register values as variadic arguments
  for (int reg_num : value_reg_nums) {
    nvbit_add_call_arg_reg_val(instr, reg_num, true);
  }
}
