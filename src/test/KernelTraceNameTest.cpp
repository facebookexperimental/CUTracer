/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>

#include <string>

#include "kernel_trace_name.h"

namespace {

using cutracer::kernel_trace_basename;

TEST(KernelTraceName, NamesTheLaunchAndTheProcess) {
  const std::string expected = "/tmp/out/kernel_7fa21c3_iter2_pid4242_add_kernel";
  EXPECT_EQ(kernel_trace_basename("/tmp/out", "7fa21c3", 2, 4242, "add_kernel"), expected);
  EXPECT_EQ(kernel_trace_basename("/tmp/out/", "7fa21c3", 2, 4242, "add_kernel"), expected);
  EXPECT_EQ(kernel_trace_basename("", "7fa21c3", 2, 4242, "add_kernel"), "kernel_7fa21c3_iter2_pid4242_add_kernel");
}

TEST(KernelTraceName, TwoProcessesLaunchingTheSameKernelGetDifferentNames) {
  EXPECT_NE(kernel_trace_basename("/tmp/out", "7fa21c3", 0, 100, "add_kernel"),
            kernel_trace_basename("/tmp/out", "7fa21c3", 0, 101, "add_kernel"));
}

TEST(KernelTraceName, KeepsTheFirst150CharactersOfTheName) {
  EXPECT_EQ(kernel_trace_basename("", "abc", 0, 1, std::string(200, 'k')),
            "kernel_abc_iter0_pid1_" + std::string(150, 'k'));
}

}  // namespace
