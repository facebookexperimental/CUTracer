/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace cutracer {

// PlanLoader sorts selections lexicographically before the host flattens them
// for the device. An empty table means all coordinates/warps are selected.
template <std::size_t Width>
#if defined(__CUDACC__)
__host__ __device__
#endif
    inline bool plan_selection_contains(const uint64_t* values, uint32_t count, const uint64_t (&key)[Width]) {
  if (!count) {
    return true;
  }
  uint32_t first = 0, last = count;
  while (first < last) {
    const auto middle = first + (last - first) / 2;
    const auto* candidate = values + static_cast<std::size_t>(middle) * Width;
    std::size_t axis = 0;
    while (axis < Width && candidate[axis] == key[axis]) {
      ++axis;
    }
    if (axis == Width) {
      return true;
    }
    if (candidate[axis] < key[axis]) {
      first = middle + 1;
    } else {
      last = middle;
    }
  }
  return false;
}

}  // namespace cutracer
