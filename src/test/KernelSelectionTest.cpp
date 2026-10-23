/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>

#include "kernel_selection.h"

namespace {

TEST(ExactKernelSelector, MissingSelectorLeavesLegacyFilteringEnabled) {
  EXPECT_FALSE(ExactKernelSelector{}.enabled());
}

TEST(ExactKernelSelector, DemangledCommaIsPartOfTheName) {
  const ExactKernelSelector selector{"bug(int*, int*)", "hash"};
  EXPECT_TRUE(selector.matches("_Z3bugPiS_", "bug(int*, int*)", "hash"));
  EXPECT_FALSE(selector.matches("_Z5cleanPiS_", "clean(int*, int*)", "hash"));
  EXPECT_FALSE(selector.matches("_Z3bugPi", "bug(int*)", "hash"));
}

TEST(ExactKernelSelector, MangledPrefixDoesNotSelectAnotherOverload) {
  const ExactKernelSelector selector{"_Z3fooPi", "hash"};
  EXPECT_TRUE(selector.matches("_Z3fooPi", "foo(int*)", "hash"));
  EXPECT_FALSE(selector.matches("_Z3fooPiS_", "foo(int*, int*)", "hash"));
}

TEST(ExactKernelSelector, SameSymbolWithDifferentCodeDoesNotMatch) {
  const ExactKernelSelector selector{"_Z3fooPi", "hash"};
  EXPECT_FALSE(selector.matches("_Z3fooPi", "foo(int*)", "other-hash"));
}

TEST(ExactKernelSelector, IncompleteIdentityNeverMatches) {
  const ExactKernelSelector missing_checksum{"_Z3fooPi", ""};
  const ExactKernelSelector missing_name{"", "hash"};
  EXPECT_TRUE(missing_checksum.enabled());
  EXPECT_TRUE(missing_name.enabled());
  EXPECT_FALSE(missing_checksum.matches("_Z3fooPi", "foo(int*)", ""));
  EXPECT_FALSE(missing_name.matches("", "", "hash"));
}

}  // namespace
