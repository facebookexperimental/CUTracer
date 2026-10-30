/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>

#include <vector>

#include "plan_selection.h"

namespace {

TEST(PlanSelectionTest, EmptyAndSingletonSelections) {
  const uint64_t cta[] = {7, 3, 9};
  const uint64_t warp[] = {31};
  EXPECT_TRUE(cutracer::plan_selection_contains(nullptr, 0, cta));
  EXPECT_TRUE(cutracer::plan_selection_contains(nullptr, 0, warp));
  EXPECT_TRUE(cutracer::plan_selection_contains(cta, 1, cta));
  EXPECT_TRUE(cutracer::plan_selection_contains(warp, 1, warp));
  const uint64_t absent_cta[] = {7, 3, 8};
  const uint64_t absent_warp[] = {30};
  EXPECT_FALSE(cutracer::plan_selection_contains(cta, 1, absent_cta));
  EXPECT_FALSE(cutracer::plan_selection_contains(warp, 1, absent_warp));
}

TEST(PlanSelectionTest, CtaCoordinatesUseLexicographicOrderAndFullWidth) {
  const uint64_t values[] = {1, 8, 2, 1, 8, 4, 1, 9, 0, 2, 0, 0, uint64_t{1} << 40, 0, 0};
  for (uint64_t x : {0ULL, 1ULL, 2ULL, 3ULL, 1ULL << 40}) {
    for (uint64_t y = 0; y < 11; ++y) {
      for (uint64_t z = 0; z < 6; ++z) {
        const uint64_t key[] = {x, y, z};
        const bool expected = (x == 1 && y == 8 && (z == 2 || z == 4)) || (x == 1 && y == 9 && z == 0) ||
                              ((x == 2 || x == (uint64_t{1} << 40)) && y == 0 && z == 0);
        EXPECT_EQ(cutracer::plan_selection_contains(values, 5, key), expected);
      }
    }
  }
}

TEST(PlanSelectionTest, LargeSelectionsIncludeBothEndsAndRejectHoles) {
  constexpr uint32_t count = 65536;
  std::vector<uint64_t> ctas, warps;
  for (uint64_t i = 0; i < count; ++i) {
    ctas.insert(ctas.end(), {i / 256, 2 * (i % 256), 7});
    warps.push_back(2 * i);
  }
  for (uint64_t i = 0; i <= count; ++i) {
    const uint64_t cta[] = {i / 256, 2 * (i % 256), 7};
    const uint64_t hole[] = {i / 256, 2 * (i % 256) + 1, 7};
    const uint64_t warp[] = {2 * i};
    const uint64_t absent_warp[] = {2 * i + 1};
    EXPECT_EQ(cutracer::plan_selection_contains(ctas.data(), count, cta), i < count);
    EXPECT_FALSE(cutracer::plan_selection_contains(ctas.data(), count, hole));
    EXPECT_EQ(cutracer::plan_selection_contains(warps.data(), count, warp), i < count);
    EXPECT_FALSE(cutracer::plan_selection_contains(warps.data(), count, absent_warp));
  }
}

}  // namespace
