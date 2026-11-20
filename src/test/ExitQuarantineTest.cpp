/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>

#include <array>

#include "exit_quarantine.h"

namespace {
TEST(ExitQuarantineTest, OnlyEnabledMatchingLaunchCanUseState) {
  ExitQuarantineState state{};
  state.launch_id = 0;
  EXPECT_FALSE(exit_quarantine_launch_matches(state, 0));
  state.enabled = 1;
  EXPECT_TRUE(exit_quarantine_launch_matches(state, 0));
  EXPECT_FALSE(exit_quarantine_launch_matches(state, 1));
  state.launch_id = 7;
  EXPECT_FALSE(exit_quarantine_launch_matches(state, 0));
  EXPECT_TRUE(exit_quarantine_launch_matches(state, 7));
  state.enabled = 0;
  EXPECT_FALSE(exit_quarantine_launch_matches(state, 7));
}

TEST(ExitQuarantineTest, SameOffsetAcrossCtasHasDifferentPoison) {
  const uint32_t offset = 0x480;
  const uint64_t first = exit_quarantine_poison(offset, 0);
  for (uint64_t cta : std::array<uint64_t, 4>{1, 17, uint64_t{1} << 32, UINT64_MAX}) {
    EXPECT_NE(first, exit_quarantine_poison(offset, cta));
  }
  EXPECT_NE(exit_quarantine_poison(offset, 17), exit_quarantine_poison(offset + 8, 17));
}

TEST(ExitQuarantineTest, PartialWordMasksAreBounded) {
  EXPECT_EQ(exit_quarantine_word_mask(0), 0u);
  EXPECT_EQ(exit_quarantine_word_mask(1), 0xffu);
  EXPECT_EQ(exit_quarantine_word_mask(4), 0xffffffffull);
  EXPECT_EQ(exit_quarantine_word_mask(7), 0x00ffffffffffffffull);
  EXPECT_EQ(exit_quarantine_word_mask(8), UINT64_MAX);
}
}  // namespace
