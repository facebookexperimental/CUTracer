/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 * See LICENSE file in the root directory for Meta's license terms.
 */

/**
 * @file test_kernel_types.cpp
 * @brief Unit tests for kernel_types.h (WarpKey, KernelDimensions, KernelWarpStats, calculate_total_warps).
 *
 * Compile and run:
 *   g++ -std=c++17 -I../../include -o test_kernel_types test_kernel_types.cpp
 *   ./test_kernel_types
 */

#include <iostream>
#include <map>
#include <unordered_set>

#include "kernel_types.h"

// Test counter
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name)                             \
  std::cout << "Testing: " << #name << "... "; \
  if (test_##name()) {                         \
    std::cout << "PASSED" << std::endl;        \
    tests_passed++;                            \
  } else {                                     \
    std::cout << "FAILED" << std::endl;        \
    tests_failed++;                            \
  }

// ============================================================================
// WarpKey Tests
// ============================================================================

bool test_warpkey_equality() {
  WarpKey a = {1, 2, 3, 4};
  WarpKey b = {1, 2, 3, 4};
  return a == b;
}

bool test_warpkey_inequality() {
  WarpKey base = {1, 2, 3, 4};
  WarpKey diff_x = {0, 2, 3, 4};
  WarpKey diff_y = {1, 0, 3, 4};
  WarpKey diff_z = {1, 2, 0, 4};
  WarpKey diff_w = {1, 2, 3, 0};
  return !(base == diff_x) && !(base == diff_y) && !(base == diff_z) && !(base == diff_w);
}

bool test_warpkey_ordering() {
  WarpKey a = {0, 0, 0, 0};
  WarpKey b = {1, 0, 0, 0};
  WarpKey c = {0, 1, 0, 0};
  WarpKey d = {0, 0, 1, 0};
  WarpKey e = {0, 0, 0, 1};

  // strict weak ordering: a < b, a < c, a < d, a < e
  if (!(a < b)) return false;
  if (!(a < c)) return false;
  if (!(a < d)) return false;
  if (!(a < e)) return false;
  // irreflexive
  if (a < a) return false;
  // usable in std::map
  std::map<WarpKey, int> m;
  m[a] = 1;
  m[b] = 2;
  return m.size() == 2 && m[a] == 1 && m[b] == 2;
}

bool test_warpkey_hash_consistency() {
  WarpKey a = {5, 10, 15, 20};
  WarpKey b = {5, 10, 15, 20};
  WarpKey::Hash hasher;
  return hasher(a) == hasher(b);
}

bool test_warpkey_unordered_set() {
  std::unordered_set<WarpKey, WarpKey::Hash> s;
  WarpKey a = {1, 2, 3, 4};
  WarpKey b = {1, 2, 3, 4};  // duplicate
  WarpKey c = {5, 6, 7, 8};
  s.insert(a);
  s.insert(b);
  s.insert(c);
  return s.size() == 2;
}

// ============================================================================
// KernelDimensions + calculate_total_warps Tests
// ============================================================================

bool test_total_warps_1d_grid() {
  // 4 blocks x 64 threads = 4 * 2 warps = 8
  KernelDimensions dims = {4, 1, 1, 64, 1, 1};
  return calculate_total_warps(dims) == 8;
}

bool test_total_warps_3d_grid() {
  // 2*3*4 = 24 blocks, 32 threads = 1 warp/block => 24
  KernelDimensions dims = {2, 3, 4, 32, 1, 1};
  return calculate_total_warps(dims) == 24;
}

bool test_total_warps_non_aligned() {
  // 1 block x 33 threads => (33+31)/32 = 2 warps
  KernelDimensions dims = {1, 1, 1, 33, 1, 1};
  return calculate_total_warps(dims) == 2;
}

bool test_total_warps_single_thread() {
  // 1 block x 1 thread => 1 warp
  KernelDimensions dims = {1, 1, 1, 1, 1, 1};
  return calculate_total_warps(dims) == 1;
}

bool test_total_warps_full_block() {
  // 1 block x 1024 threads => 32 warps
  KernelDimensions dims = {1, 1, 1, 1024, 1, 1};
  return calculate_total_warps(dims) == 32;
}

bool test_total_warps_3d_block() {
  // 2 blocks, blockDim 4*4*4=64 threads => 2 warps/block => 4 total
  KernelDimensions dims = {2, 1, 1, 4, 4, 4};
  return calculate_total_warps(dims) == 4;
}

// ============================================================================
// KernelWarpStats Tests
// ============================================================================

bool test_warpstats_default_init() {
  KernelWarpStats stats;
  return stats.total_warps == 0 && stats.all_seen_warps.empty() && stats.finished_warps.empty();
}

bool test_warpstats_tracking() {
  KernelWarpStats stats;
  stats.total_warps = 4;
  WarpKey w0 = {0, 0, 0, 0};
  WarpKey w1 = {0, 0, 0, 1};
  stats.all_seen_warps.insert(w0);
  stats.all_seen_warps.insert(w1);
  stats.finished_warps.insert(w0);
  return stats.all_seen_warps.size() == 2 && stats.finished_warps.size() == 1;
}

bool test_warpstats_all_finished() {
  KernelWarpStats stats;
  WarpKey w0 = {0, 0, 0, 0};
  WarpKey w1 = {0, 0, 0, 1};
  stats.total_warps = 2;
  stats.all_seen_warps.insert(w0);
  stats.all_seen_warps.insert(w1);
  stats.finished_warps.insert(w0);
  stats.finished_warps.insert(w1);
  return stats.finished_warps.size() == stats.all_seen_warps.size();
}

// ============================================================================
// Main
// ============================================================================

int main() {
  std::cout << "========================================" << std::endl;
  std::cout << "Kernel Types Unit Tests" << std::endl;
  std::cout << "========================================" << std::endl;

  TEST(warpkey_equality);
  TEST(warpkey_inequality);
  TEST(warpkey_ordering);
  TEST(warpkey_hash_consistency);
  TEST(warpkey_unordered_set);

  TEST(total_warps_1d_grid);
  TEST(total_warps_3d_grid);
  TEST(total_warps_non_aligned);
  TEST(total_warps_single_thread);
  TEST(total_warps_full_block);
  TEST(total_warps_3d_block);

  TEST(warpstats_default_init);
  TEST(warpstats_tracking);
  TEST(warpstats_all_finished);

  std::cout << "========================================" << std::endl;
  std::cout << "Results: " << tests_passed << " passed, " << tests_failed << " failed" << std::endl;
  std::cout << "========================================" << std::endl;

  return tests_failed > 0 ? 1 : 0;
}
