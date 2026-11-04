/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 * See LICENSE file in the root directory for Meta's license terms.
 */

#ifndef DELAY_INJECT_CONFIG_H
#define DELAY_INJECT_CONFIG_H

#include <stdint.h>

#include <map>
#include <string>

#include "nvbit.h"

/**
 * @brief Represents a single instrumentation point for delay injection.
 *
 * Each instrumentation point captures:
 * - pc_offset: Program counter offset for the instruction (used as key)
 * - sass: The SASS assembly instruction string
 * - delay_ns: The delay value in nanoseconds
 * - enabled: Whether this point is enabled (randomly set, 50% probability)
 */
struct DelayInstrumentationPoint {
  uint64_t pc_offset;
  std::string sass;
  uint32_t delay_ns;
  bool enabled;
  uint32_t cluster_seed;  // Seed for cluster mode CTA selection (0 when not in cluster mode)
};

/**
 * @brief Configuration for delay instrumentation points in a single kernel.
 */
struct KernelDelayInjectConfig {
  std::string kernel_name;
  // FNV-1a hash of kernel name + all SASS instructions (hex string).
  // This provides robust kernel identification across recompilations:
  // same kernel name with different SASS produces different checksum.
  std::string kernel_checksum;
  std::string timestamp;  // ISO 8601 timestamp when kernel was instrumented
  std::map<uint64_t, DelayInstrumentationPoint> instrumentation_points;  // Keyed by pc_offset
};

/**
 * @brief Master configuration containing all kernel delay configs.
 */
struct DelayInjectConfig {
  std::string version = "1.0";
  uint32_t delay_ns;
  std::map<std::string, KernelDelayInjectConfig> kernels;  // Indexed by kernel name
};

// Global delay injection configuration
extern DelayInjectConfig g_delay_inject_config;

/**
 * @brief Initialize delay JSON configuration for export.
 *
 * Sets up the global delay config with the delay value from env.
 * This is used for exporting instrumentation points to JSON for replay.
 */
void init_delay_json_config();

/**
 * @brief Create a new kernel delay config for a given kernel name.
 *
 * Should be called once per kernel before the instruction iteration loop.
 *
 * @param kernel_name The kernel name
 * @param kernel_checksum FNV-1a hash of kernel name + SASS instructions (hex string) for robust identification
 * @return Pointer to the newly created kernel config
 */
KernelDelayInjectConfig* create_kernel_delay_config(const std::string& kernel_name, const std::string& kernel_checksum);

/**
 * @brief Register an instrumentation point for a kernel.
 *
 * Creates and registers a DelayInstrumentationPoint from the instruction.
 *
 * @param kdc Pointer to the kernel delay config (from create_kernel_delay_config)
 * @param instr The NVBit instruction to register
 * @param delay_ns The delay value in nanoseconds
 * @param enabled Whether this point is enabled
 */
void register_delay_instrumentation_point(KernelDelayInjectConfig* kdc, Instr* instr, uint32_t delay_ns, bool enabled,
                                          uint32_t cluster_seed = 0);

/**
 * @brief Write the delay config to CUTRACER_DELAY_DUMP_PATH, if set.
 *
 * Called after each newly instrumented kernel and when the last context ends:
 * a process can exit without ending its contexts (torch elastic workers do).
 */
void save_delay_config();

/**
 * @brief Load delay configuration from a JSON file for replay mode.
 *
 * @param filepath Path to the JSON config file
 * @return true if loading succeeded, false otherwise
 */
bool load_delay_config(const std::string& filepath);

/**
 * @brief Check if delay replay mode is active.
 *
 * Delay replay mode uses a previously saved delay config file to deterministically
 * reproduce the same instrumentation pattern.
 *
 * @return true if delay replay mode is active, false otherwise
 */
bool is_delay_replay_mode();

/**
 * @brief Look up an instrumentation point configuration for replay.
 *
 * @param replay_points Pointer to the instrumentation points map (from get_replay_instrumentation_points)
 * @param pc_offset The program counter offset of the instruction
 * @param[out] enabled Output: whether the point is enabled
 * @param[out] delay_ns Output: the delay value in nanoseconds
 * @return true if found, false if not found
 */
bool lookup_replay_config(const std::map<uint64_t, DelayInstrumentationPoint>* replay_points, uint64_t pc_offset,
                          bool& enabled, uint32_t& delay_ns, uint32_t& cluster_seed);

/**
 * @brief Get the instrumentation points map for a kernel in replay mode.
 *
 * Should be called once per kernel before the instruction iteration loop.
 * Matches kernels by kernel_checksum (computed from kernel name + SASS) for robust identification.
 *
 * @param kernel_name The kernel name to look up
 * @param kernel_checksum The kernel checksum to match (FNV-1a hash of kernel name + SASS)
 * @return Pointer to the instrumentation points map, or nullptr if not found
 */
const std::map<uint64_t, DelayInstrumentationPoint>* get_replay_instrumentation_points(
    const std::string& kernel_name, const std::string& kernel_checksum);

#endif /* DELAY_INJECT_CONFIG_H */
