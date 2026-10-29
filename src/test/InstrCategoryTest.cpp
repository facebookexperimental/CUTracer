/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>

#include "instr_category.h"

namespace {

TEST(InstrCategory, CapturesSynchronizationFamilies) {
  const char* instructions[] = {
      "WARPGROUP.DEPBAR.LE 0x0 ;",
      "SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [UR15+0x380c0], R0 ;",
      "@!P0 SYNCS.ARRIVE.TRANS64 [UR4+0x20] ;",
      "BAR.SYNC.DEFER_BLOCKING 0x5, 0x40 ;",
      "BAR.ARV 0x5, 0x40 ;",
      "DEPBAR.LE SB0, 0x0 ;",
  };
  for (const auto* instruction : instructions) {
    EXPECT_EQ(InstrCategory::SYNC, detect_instr_category(instruction)) << instruction;
  }
}

TEST(InstrCategory, CompletionAndMemoryInstructionsAreNotSynchronization) {
  const char* instructions[] = {
      "EXIT ;", "@P0 EXIT ;", "LDGDEPBAR ;", "STSM.16 [R8], R12 ;", "LDG.E R0, [R2] ;",
  };
  for (const auto* instruction : instructions) {
    EXPECT_EQ(InstrCategory::NONE, detect_instr_category(instruction)) << instruction;
  }
}

TEST(InstrCategory, SynchronizationExpansionPreservesMmaAndTma) {
  EXPECT_EQ(InstrCategory::MMA, detect_instr_category("HGMMA.16816.F32 R0, desc[UR8], desc[UR12] ;"));
  EXPECT_EQ(InstrCategory::TMA, detect_instr_category("UTMALDG.2D [UR8], [R16], P0 ;"));
  EXPECT_EQ(InstrCategory::TMA, detect_instr_category("UTMASTG.2D [UR8], [R16], R32 ;"));
}

}  // namespace
