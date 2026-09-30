/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include "tma_registers.h"

namespace {

std::vector<int> collect_registers(uint32_t sm_family, const std::array<int, InstrType::MAX_TMA_REGS>& registers) {
  InstrType::operand_t operand{};
  operand.type = InstrType::OperandType::TMA_PARAM_HANDLE;
  std::copy(registers.begin(), registers.end(), operand.u.tma_param_handle.ureg_num);
  std::vector<int> captured{4};
  collect_tma_uniform_registers(operand, sm_family, captured);
  return captured;
}

TEST(TmaRegistersTest, HopperSkipsZeroRegisterAndAbsentSlots) {
  EXPECT_EQ(collect_registers(90, {8, 63, 16, -1}), (std::vector<int>{4, 8, 16}));
}

TEST(TmaRegistersTest, BlackwellCapturesUr63AndSkipsUr255) {
  for (const uint32_t sm_family : {100, 103, 120}) {
    SCOPED_TRACE(sm_family);
    EXPECT_EQ(collect_registers(sm_family, {63, 255, 128, -1}), (std::vector<int>{4, 63, 128}));
  }
}

TEST(TmaRegistersTest, PreservesOperandOrderAndRepeatedRegisters) {
  EXPECT_EQ(collect_registers(100, {63, 62, 63, 255}), (std::vector<int>{4, 63, 62, 63}));
}

}  // namespace
