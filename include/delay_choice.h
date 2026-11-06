/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>
#include <string_view>

namespace cutracer {

// A delay point's on/off choice and cluster seed depend only on the run seed,
// the kernel checksum and the PC, so processes that share CUTRACER_DELAY_SEED
// make the same choice at the same point, whatever order they instrument in.

inline uint64_t delay_mix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

inline uint64_t delay_point_hash(uint64_t seed, std::string_view kernel_checksum, uint64_t pc_offset, uint64_t stream) {
  uint64_t checksum_hash = 0xcbf29ce484222325ULL;
  for (const unsigned char c : kernel_checksum) {
    checksum_hash = (checksum_hash ^ c) * 0x100000001b3ULL;
  }
  return delay_mix64(delay_mix64(delay_mix64(seed ^ stream) ^ checksum_hash) ^ pc_offset);
}

// Probabilities of exactly 0 and 1 give all-off and all-on points.
inline bool delay_point_enabled(uint64_t seed, std::string_view kernel_checksum, uint64_t pc_offset,
                                float enable_prob) {
  if (enable_prob >= 1.0f) {
    return true;
  }
  if (enable_prob <= 0.0f) {
    return false;
  }
  const double unit = static_cast<double>(delay_point_hash(seed, kernel_checksum, pc_offset, 1) >> 11) * 0x1.0p-53;
  return unit < enable_prob;
}

inline uint32_t delay_point_cluster_seed(uint64_t seed, std::string_view kernel_checksum, uint64_t pc_offset) {
  return static_cast<uint32_t>(delay_point_hash(seed, kernel_checksum, pc_offset, 2));
}

}  // namespace cutracer
