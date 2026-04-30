/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 *
 * See LICENSE file in the root directory for Meta's license terms.
 */

#ifndef KERNEL_TYPES_H
#define KERNEL_TYPES_H

#include <cstdint>
#include <unordered_set>

/* Structure to uniquely identify a warp */
struct WarpKey {
  int cta_id_x;
  int cta_id_y;
  int cta_id_z;
  // global warp id
  int warp_id;

  // Operator for map comparison
  bool operator<(const WarpKey& other) const {
    if (cta_id_x != other.cta_id_x) return cta_id_x < other.cta_id_x;
    if (cta_id_y != other.cta_id_y) return cta_id_y < other.cta_id_y;
    if (cta_id_z != other.cta_id_z) return cta_id_z < other.cta_id_z;
    return warp_id < other.warp_id;
  }

  // Hash function for unordered_map
  struct Hash {
    size_t operator()(const WarpKey& k) const {
      return (size_t)k.cta_id_x ^ ((size_t)k.cta_id_y << 10) ^ ((size_t)k.cta_id_z << 20) ^ ((size_t)k.warp_id << 30);
    }
  };

  // Equality operator for unordered_map
  bool operator==(const WarpKey& other) const {
    return cta_id_x == other.cta_id_x && cta_id_y == other.cta_id_y && cta_id_z == other.cta_id_z &&
           warp_id == other.warp_id;
  }
};

/**
 * @brief Grid and block dimensions for a kernel launch.
 */
struct KernelDimensions {
  unsigned int gridDimX;
  unsigned int gridDimY;
  unsigned int gridDimZ;
  unsigned int blockDimX;
  unsigned int blockDimY;
  unsigned int blockDimZ;
};

/**
 * @brief Tracks warp statistics for a single kernel launch.
 *
 * This structure maintains complete information about all warps in a kernel:
 * - Total number of warps (calculated from grid/block dimensions)
 * - All warps ever seen executing
 * - Warps that have finished execution
 * - Currently active warps (maintained elsewhere in CTXstate)
 */
struct KernelWarpStats {
  // Total number of warps in this kernel launch
  uint32_t total_warps;

  // Grid and block dimensions
  KernelDimensions dimensions;

  // All warps that have ever been observed executing
  std::unordered_set<WarpKey, WarpKey::Hash> all_seen_warps;

  // Warps that were once active but have finished
  std::unordered_set<WarpKey, WarpKey::Hash> finished_warps;

  KernelWarpStats() : total_warps(0) {
  }
};

inline uint32_t calculate_total_warps(const KernelDimensions& dims) {
  uint32_t threads_per_block = dims.blockDimX * dims.blockDimY * dims.blockDimZ;
  uint32_t warps_per_block = (threads_per_block + 31) / 32;
  uint32_t total_blocks = dims.gridDimX * dims.gridDimY * dims.gridDimZ;
  return total_blocks * warps_per_block;
}

#endif /* KERNEL_TYPES_H */
