/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>

#include "hang_check_policy.h"

namespace {

using cutracer::HangCheckPolicy;
using Outcome = HangCheckPolicy::Outcome;
using std::chrono::milliseconds;

const HangCheckPolicy::Clock::time_point kStart{};
constexpr size_t kActive = 8;

HangCheckPolicy::Clock::time_point at(int64_t ms) {
  return kStart + milliseconds(ms);
}

TEST(HangCheckPolicy, ThreeChecksWithoutProgressTerminate) {
  HangCheckPolicy policy;
  policy.restart(at(0));
  EXPECT_EQ(policy.record(at(1000), kActive, 0), Outcome::Stall);
  EXPECT_EQ(policy.record(at(2000), kActive, 0), Outcome::Stall);
  EXPECT_EQ(policy.record(at(3000), kActive, 0), Outcome::Terminate);
  EXPECT_EQ(policy.hits(), 3u);
}

TEST(HangCheckPolicy, ChecksAreAtLeastOneIntervalApart) {
  HangCheckPolicy policy;
  policy.restart(at(0));
  EXPECT_FALSE(policy.due(at(999)));
  EXPECT_TRUE(policy.due(at(1000)));
  policy.record(at(1000), kActive, 0);
  // A second boundary 10 ms later is not a new check.
  EXPECT_FALSE(policy.due(at(1010)));
  EXPECT_TRUE(policy.due(at(2000)));
}

TEST(HangCheckPolicy, ProgressStartsTheCountOver) {
  HangCheckPolicy policy;
  policy.record(at(1000), kActive, 0);
  policy.record(at(2000), kActive, 0);
  EXPECT_EQ(policy.record(at(3000), kActive, 1), Outcome::Progress);
  EXPECT_EQ(policy.hits(), 0u);
  EXPECT_EQ(policy.record(at(4000), kActive, 0), Outcome::Stall);
}

TEST(HangCheckPolicy, HitsOfAnEarlierLaunchDoNotCarryOver) {
  HangCheckPolicy policy;
  policy.record(at(1000), kActive, 0);
  policy.record(at(2000), kActive, 0);
  policy.restart(at(2500));
  EXPECT_EQ(policy.hits(), 0u);
  EXPECT_FALSE(policy.due(at(3000)));
  EXPECT_EQ(policy.record(at(3500), kActive, 0), Outcome::Stall);
  EXPECT_EQ(policy.record(at(4500), kActive, 0), Outcome::Stall);
}

TEST(HangCheckPolicy, ACheckWithoutActiveWarpsStartsTheCountOver) {
  HangCheckPolicy policy;
  policy.record(at(1000), kActive, 0);
  policy.record(at(2000), kActive, 0);
  // The last warps exited: nothing is stalled, so nothing counts as a hit.
  EXPECT_EQ(policy.record(at(3000), 0, 0), Outcome::Idle);
  EXPECT_EQ(policy.hits(), 0u);
  EXPECT_FALSE(policy.due(at(3500)));
  EXPECT_EQ(policy.record(at(4000), kActive, 0), Outcome::Stall);
}

TEST(HangCheckPolicy, FirstCheckIsDueWithoutARestart) {
  HangCheckPolicy policy;
  EXPECT_TRUE(policy.due(at(0)));
}

}  // namespace
