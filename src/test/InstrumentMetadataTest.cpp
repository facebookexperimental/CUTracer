/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>

#include "instrument_metadata.h"

// Host tests supply the parsed configuration without linking the CUDA tracer.
std::unordered_map<InstrumentType, int> instrument_type_to_index;
IPointType uniform_ipoint = IPointType::DEFAULT;
std::vector<IPointType> ipoint_overrides;

namespace {

class InstrumentMetadataTest : public testing::Test {
 protected:
  void SetUp() override {
    instrument_type_to_index = {{InstrumentType::REG_TRACE, 0}};
    uniform_ipoint = IPointType::DEFAULT;
    ipoint_overrides.clear();
  }

  void TearDown() override {
    instrument_type_to_index.clear();
    uniform_ipoint = IPointType::DEFAULT;
    ipoint_overrides.clear();
  }
};

TEST_F(InstrumentMetadataTest, DefaultRegisterTraceIsBefore) {
  EXPECT_EQ(instrument_ipoints_to_json(), (nlohmann::json{{"reg_trace", "BEFORE"}}));
  EXPECT_EQ(resolve_instrument_ipoint(InstrumentType::REG_TRACE, IPointType::BEFORE), IPointType::BEFORE);
  EXPECT_EQ(resolve_instrument_ipoint(InstrumentType::MEM_VALUE_TRACE, IPointType::AFTER), IPointType::AFTER);
}

TEST_F(InstrumentMetadataTest, UniformAfterIsSerialized) {
  uniform_ipoint = IPointType::AFTER;
  EXPECT_EQ(instrument_ipoints_to_json(), (nlohmann::json{{"reg_trace", "AFTER"}}));
  EXPECT_EQ(resolve_instrument_ipoint(InstrumentType::REG_TRACE, IPointType::BEFORE), IPointType::AFTER);
}

TEST_F(InstrumentMetadataTest, PerModeOverrideUsesEnabledOrder) {
  instrument_type_to_index = {{InstrumentType::MEM_VALUE_TRACE, 0}, {InstrumentType::REG_TRACE, 1}};
  ipoint_overrides = {IPointType::BEFORE, IPointType::AFTER};
  EXPECT_EQ(instrument_ipoints_to_json(), (nlohmann::json{{"reg_trace", "AFTER"}}));
  EXPECT_EQ(resolve_instrument_ipoint(InstrumentType::MEM_VALUE_TRACE, IPointType::AFTER), IPointType::BEFORE);
}

TEST_F(InstrumentMetadataTest, PerModeBeforeOverridesUniformAfter) {
  uniform_ipoint = IPointType::AFTER;
  ipoint_overrides = {IPointType::BEFORE};
  EXPECT_EQ(instrument_ipoints_to_json(), (nlohmann::json{{"reg_trace", "BEFORE"}}));
}

TEST_F(InstrumentMetadataTest, DefaultOverrideFallsThroughToUniform) {
  uniform_ipoint = IPointType::AFTER;
  ipoint_overrides = {IPointType::DEFAULT};
  EXPECT_EQ(instrument_ipoints_to_json(), (nlohmann::json{{"reg_trace", "AFTER"}}));
}

TEST_F(InstrumentMetadataTest, DisabledRegisterTraceHasNoPosition) {
  instrument_type_to_index = {{InstrumentType::OPCODE_ONLY, 0}};
  uniform_ipoint = IPointType::AFTER;
  EXPECT_EQ(instrument_ipoints_to_json(), nlohmann::json::object());
}

TEST_F(InstrumentMetadataTest, KernelMetadataRoundTripKeepsResolvedPosition) {
  ipoint_overrides = {IPointType::AFTER};
  const nlohmann::json metadata = {{"type", "kernel_metadata"},
                                   {"instrument_modes", {"reg_trace"}},
                                   {"instrument_ipoints", instrument_ipoints_to_json()}};
  const auto restored = nlohmann::json::parse(metadata.dump());
  EXPECT_EQ(restored["instrument_ipoints"]["reg_trace"], "AFTER");
}

}  // namespace
