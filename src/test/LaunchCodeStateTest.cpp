/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>

#include <vector>

#include "launch_code_state.h"

namespace {

using cutracer::LaunchCodeState;

const int kRoot = 1;
const int kOtherRoot = 2;
const int kSharedCallee = 3;
const void* const kFunc = &kRoot;
const void* const kOtherFunc = &kOtherRoot;
const void* const kCallee = &kSharedCallee;
const std::vector<const void*> kKernel{kFunc};
const std::vector<const void*> kKernelWithCallee{kFunc, kCallee};
const std::vector<const void*> kOtherKernelWithCallee{kOtherFunc, kCallee};

TEST(LaunchCodeState, UninstrumentedLaunchesNeverWait) {
  LaunchCodeState state;
  EXPECT_FALSE(state.update(kKernel, false, false));
  EXPECT_FALSE(state.update(kKernel, false, false));
}

TEST(LaunchCodeState, AFunctionsFirstLaunchNeverWaitsEvenWhenInstrumented) {
  LaunchCodeState state;
  EXPECT_FALSE(state.update(kKernel, true, true));
  EXPECT_FALSE(state.update(kKernel, true, false));
}

TEST(LaunchCodeState, InstrumentingAFunctionThatAlreadyRanWaits) {
  LaunchCodeState state;
  state.update(kKernel, false, false);
  EXPECT_TRUE(state.update(kKernel, true, true));
  EXPECT_FALSE(state.update(kKernel, true, false));
}

TEST(LaunchCodeState, SwitchingBetweenOriginalAndInstrumentedCodeWaits) {
  LaunchCodeState state;
  state.update(kKernel, true, true);
  EXPECT_TRUE(state.update(kKernel, false, false));
  EXPECT_FALSE(state.update(kKernel, false, false));
  EXPECT_TRUE(state.update(kKernel, true, false));
}

TEST(LaunchCodeState, ASharedCalleeSwitchedByAnotherRootWaits) {
  LaunchCodeState state;
  state.update(kKernelWithCallee, true, true);
  // Another kernel runs the callee's original code: NVBit swaps it in place.
  EXPECT_TRUE(state.update(kOtherKernelWithCallee, false, false));
  EXPECT_TRUE(state.update(kKernelWithCallee, true, false));
}

TEST(LaunchCodeState, AForgottenHandleStartsOver) {
  LaunchCodeState state;
  state.update(kKernel, true, true);
  state.forget(kFunc);
  EXPECT_FALSE(state.update(kKernel, false, false));
}

}  // namespace
