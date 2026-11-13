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

TEST(HangCheckPolicy, MinimumStallHoldsTerminationUntilTheStallLastsLongEnough) {
  HangCheckPolicy policy({milliseconds(1000), 3, milliseconds(10000)});
  policy.restart(at(0));
  for (int64_t t = 1000; t < 11000; t += 1000) {
    EXPECT_EQ(policy.record(at(t), kActive, 0), Outcome::Stall) << "t=" << t;
  }
  EXPECT_EQ(policy.stalled_for(at(11000)), milliseconds(10000));
  EXPECT_EQ(policy.record(at(11000), kActive, 0), Outcome::Terminate);
}

TEST(HangCheckPolicy, ProgressRestartsTheMinimumStall) {
  HangCheckPolicy policy({milliseconds(1000), 3, milliseconds(5000)});
  policy.record(at(1000), kActive, 0);
  policy.record(at(5000), kActive, 0);
  policy.record(at(5500), kActive, 1);
  EXPECT_EQ(policy.stalled_for(at(5500)), milliseconds(0));
  policy.record(at(6500), kActive, 0);
  policy.record(at(7500), kActive, 0);
  EXPECT_EQ(policy.record(at(8500), kActive, 0), Outcome::Stall);
  EXPECT_EQ(policy.record(at(11500), kActive, 0), Outcome::Terminate);
}

TEST(HangCheckPolicy, ANewLaunchOrNoActiveWarpRestartsTheMinimumStall) {
  HangCheckPolicy policy({milliseconds(1000), 1, milliseconds(3000)});
  policy.record(at(1000), kActive, 0);
  policy.record(at(3000), kActive, 0);
  policy.restart(at(3500));
  EXPECT_EQ(policy.stalled_for(at(3500)), milliseconds(0));
  EXPECT_EQ(policy.record(at(4500), kActive, 0), Outcome::Stall);
  EXPECT_EQ(policy.record(at(5500), 0, 0), Outcome::Idle);
  EXPECT_EQ(policy.stalled_for(at(5500)), milliseconds(0));
  EXPECT_EQ(policy.record(at(6500), kActive, 0), Outcome::Stall);
  EXPECT_EQ(policy.record(at(9500), kActive, 0), Outcome::Terminate);
}

TEST(HangCheckPolicy, HitsStillBindWhenTheMinimumStallIsShorter) {
  HangCheckPolicy policy({milliseconds(5000), 3, milliseconds(1000)});
  policy.restart(at(0));
  EXPECT_EQ(policy.record(at(5000), kActive, 0), Outcome::Stall);
  EXPECT_EQ(policy.record(at(10000), kActive, 0), Outcome::Stall);
  EXPECT_EQ(policy.record(at(15000), kActive, 0), Outcome::Terminate);
}

TEST(HangCheckPolicy, OneHitWithoutMinimumTerminatesAtTheFirstStall) {
  HangCheckPolicy policy({milliseconds(250), 1, milliseconds(0)});
  EXPECT_EQ(policy.record(at(250), kActive, 0), Outcome::Terminate);
}

TEST(HangCheckPolicy, FirstCheckIsDueWithoutARestart) {
  HangCheckPolicy policy;
  EXPECT_TRUE(policy.due(at(0)));
}

}  // namespace
