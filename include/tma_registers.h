/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>
#include <vector>

#include "instr_types.h"

inline void collect_tma_uniform_registers(const InstrType::operand_t& operand, uint32_t sm_family,
                                          std::vector<int>& registers) {
  // NVBit 1.8.1 uses 255 for URZ on Blackwell and newer GPUs. UR63 is a
  // regular register on those architectures and must still be captured.
  const int zero_register = sm_family >= 100 ? InstrType::URZ_BLACKWELL_PLUS : InstrType::URZ;
  for (int reg : operand.u.tma_param_handle.ureg_num) {
    if (reg >= 0 && reg != zero_register) {
      registers.push_back(reg);
    }
  }
}
