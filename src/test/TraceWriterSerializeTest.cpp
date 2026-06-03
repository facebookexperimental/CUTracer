/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

// Goldens for the rapidjson trace serializer (serialize_record_rapidjson).
//
// Each test builds a GPU record struct with distinctive, non-uniform values,
// serializes it through the rapidjson path, and compares the parsed result to
// an independently-constructed nlohmann::json object. The comparison is
// semantic (parse -> JSON-value equality), so it is agnostic to key order --
// matching the migration's contract that no consumer depends on key order.
//
// These run with no GPU and cover all four types the rapidjson path handles,
// including mem_value_trace and opcode_only, which the case15 workload never
// emits (so the runtime A/B mode cannot reach them).

#include <gtest/gtest.h>

#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>

#include "common.h"
#include "trace_serialize.h"
#include "trace_writer.h"

namespace {

// A fixed, recognizable context pointer and its expected lowercase-hex string.
CUcontext testCtx() {
  return reinterpret_cast<CUcontext>(static_cast<uintptr_t>(0xABCDEF12ull));
}

constexpr const char* kCtxHex = "0xabcdef12";

// Serialize via the rapidjson seam and parse the line. Asserts the line is
// non-empty (the rapidjson path handled the type).
nlohmann::json parseRj(const TraceRecord& rec) {
  const std::string line = cutracer::serialize_record_rapidjson(rec);
  EXPECT_FALSE(line.empty()) << "rapidjson returned an empty line for a handled type";
  if (line.empty()) {
    return nlohmann::json::object();
  }
  return nlohmann::json::parse(line);
}

TEST(SerializeRecordRapidjson, RegTrace_BasicNoIndicesNoUregs) {
  reg_info_t r{};
  r.header.type = MSG_TYPE_REG_INFO;
  r.cta_id_x = 1;
  r.cta_id_y = 2;
  r.cta_id_z = 3;
  r.warp_id = 7;
  r.opcode_id = 42;
  r.kernel_launch_id = 5;
  r.pc = 0x1234;
  r.num_regs = 2;
  r.num_uregs = 0;
  // Distinct per (thread, reg) so the [thread][reg] -> [reg][thread] transpose
  // is actually exercised rather than mirrored.
  for (int t = 0; t < 32; t++) {
    for (int reg = 0; reg < r.num_regs; reg++) {
      r.reg_vals[t][reg] = 1000u * reg + t;
    }
  }
  const TraceRecord rec = TraceRecord::create_reg_trace(testCtx(), "SASS", /*trace_idx=*/11, /*ts=*/99, &r);

  nlohmann::json expected;
  expected["type"] = "reg_trace";
  expected["cta"] = {1, 2, 3};
  expected["ctx"] = kCtxHex;
  expected["grid_launch_id"] = 5;
  expected["opcode_id"] = 42;
  expected["pc"] = "0x1234";
  nlohmann::json regs = nlohmann::json::array();
  for (int reg = 0; reg < 2; reg++) {
    nlohmann::json col = nlohmann::json::array();
    for (int t = 0; t < 32; t++) {
      col.push_back(1000u * reg + t);
    }
    regs.push_back(col);
  }
  expected["regs"] = regs;
  expected["timestamp"] = 99;
  expected["trace_index"] = 11;
  expected["warp"] = 7;

  EXPECT_EQ(parseRj(rec), expected);
}

TEST(SerializeRecordRapidjson, RegTrace_WithIndicesAndUregs) {
  reg_info_t r{};
  r.header.type = MSG_TYPE_REG_INFO;
  r.warp_id = 4;
  r.opcode_id = 1;
  r.kernel_launch_id = 8;
  r.pc = 0x20;
  r.num_regs = 1;
  for (int t = 0; t < 32; t++) {
    r.reg_vals[t][0] = 7u * t;
  }
  r.num_uregs = 3;
  r.ureg_vals[0] = 100;
  r.ureg_vals[1] = 200;
  r.ureg_vals[2] = 300;

  RegIndices idx;
  idx.reg_indices = {5};
  idx.ureg_indices = {1, 2, 3};

  const TraceRecord rec = TraceRecord::create_reg_trace(testCtx(), "SASS", 12, 100, &r, &idx);

  nlohmann::json expected;
  expected["type"] = "reg_trace";
  expected["cta"] = {0, 0, 0};
  expected["ctx"] = kCtxHex;
  expected["grid_launch_id"] = 8;
  expected["opcode_id"] = 1;
  expected["pc"] = "0x20";
  nlohmann::json col = nlohmann::json::array();
  for (int t = 0; t < 32; t++) {
    col.push_back(7u * t);
  }
  expected["regs"] = nlohmann::json::array({col});
  expected["regs_indices"] = {5};
  expected["timestamp"] = 100;
  expected["trace_index"] = 12;
  expected["uregs"] = {100, 200, 300};
  expected["uregs_indices"] = {1, 2, 3};
  expected["warp"] = 4;

  EXPECT_EQ(parseRj(rec), expected);
}

TEST(SerializeRecordRapidjson, RegTrace_UregsPresentButUregIndicesEmpty) {
  reg_info_t r{};
  r.header.type = MSG_TYPE_REG_INFO;
  r.warp_id = 2;
  r.opcode_id = 3;
  r.kernel_launch_id = 1;
  r.pc = 0x8;
  r.num_regs = 1;
  for (int t = 0; t < 32; t++) {
    r.reg_vals[t][0] = t;
  }
  r.num_uregs = 2;
  r.ureg_vals[0] = 7;
  r.ureg_vals[1] = 8;

  RegIndices idx;
  idx.reg_indices = {0};
  // ureg_indices intentionally empty -> "uregs_indices" key omitted.

  const TraceRecord rec = TraceRecord::create_reg_trace(testCtx(), "SASS", 1, 2, &r, &idx);
  const nlohmann::json out = parseRj(rec);

  EXPECT_EQ(out["uregs"], (nlohmann::json{7, 8}));
  EXPECT_TRUE(out.contains("regs_indices"));
  EXPECT_FALSE(out.contains("uregs_indices"));
}

TEST(SerializeRecordRapidjson, MemAddrTrace_AddrsAreUint64WithIpointB) {
  mem_addr_access_t m{};
  m.header.type = MSG_TYPE_MEM_ADDR_ACCESS;
  m.cta_id_x = 4;
  m.cta_id_y = 5;
  m.cta_id_z = 6;
  m.warp_id = 3;
  m.opcode_id = 8;
  m.kernel_launch_id = 2;
  m.pc = 0xDEAD;
  for (int i = 0; i < 32; i++) {
    m.addrs[i] = 0x100000000ull + i;  // > 2^32 to exercise Uint64 width
  }
  const TraceRecord rec = TraceRecord::create_mem_trace(testCtx(), "SASS", 3, 4, &m);

  nlohmann::json expected;
  expected["type"] = "mem_addr_trace";
  expected["ipoint"] = "B";
  nlohmann::json addrs = nlohmann::json::array();
  for (int i = 0; i < 32; i++) {
    addrs.push_back(0x100000000ull + i);
  }
  expected["addrs"] = addrs;
  expected["cta"] = {4, 5, 6};
  expected["ctx"] = kCtxHex;
  expected["grid_launch_id"] = 2;
  expected["opcode_id"] = 8;
  expected["pc"] = "0xdead";  // lowercase, no leading zeros
  expected["timestamp"] = 4;
  expected["trace_index"] = 3;
  expected["warp"] = 3;

  EXPECT_EQ(parseRj(rec), expected);
}

TEST(SerializeRecordRapidjson, OpcodeOnly_HasNoIpoint) {
  opcode_only_t o{};
  o.header.type = MSG_TYPE_OPCODE_ONLY;
  o.cta_id_x = 7;
  o.cta_id_y = 8;
  o.cta_id_z = 9;
  o.warp_id = 1;
  o.opcode_id = 99;
  o.kernel_launch_id = 3;
  o.pc = 0x40;
  const TraceRecord rec = TraceRecord::create_opcode_trace(testCtx(), "SASS", 5, 6, &o);

  nlohmann::json expected;
  expected["type"] = "opcode_only";
  expected["cta"] = {7, 8, 9};
  expected["ctx"] = kCtxHex;
  expected["grid_launch_id"] = 3;
  expected["opcode_id"] = 99;
  expected["pc"] = "0x40";
  expected["timestamp"] = 6;
  expected["trace_index"] = 5;
  expected["warp"] = 1;

  EXPECT_EQ(parseRj(rec), expected);
}

TEST(SerializeRecordRapidjson, MemValueTrace_LoadTwoRegsPerLane) {
  mem_value_access_t mv{};
  mv.header.type = MSG_TYPE_MEM_VALUE_ACCESS;
  mv.cta_id_x = 1;
  mv.cta_id_y = 1;
  mv.cta_id_z = 1;
  mv.warp_id = 6;
  mv.opcode_id = 10;
  mv.kernel_launch_id = 9;
  mv.pc = 0x55;
  mv.mem_space = 1;
  mv.is_load = 1;
  mv.access_size = 8;  // regs_needed = (8 + 3) / 4 = 2
  for (int i = 0; i < 32; i++) {
    mv.addrs[i] = 0x200000000ull + i;
  }
  for (int lane = 0; lane < 32; lane++) {
    for (int reg = 0; reg < 4; reg++) {
      mv.values[lane][reg] = 10u * lane + reg;
    }
  }
  const TraceRecord rec = TraceRecord::create_mem_value_trace(testCtx(), "SASS", 7, 8, &mv);

  nlohmann::json expected;
  expected["type"] = "mem_value_trace";
  expected["ipoint"] = "A";
  expected["access_size"] = 8;
  nlohmann::json addrs = nlohmann::json::array();
  for (int i = 0; i < 32; i++) {
    addrs.push_back(0x200000000ull + i);
  }
  expected["addrs"] = addrs;
  expected["cta"] = {1, 1, 1};
  expected["ctx"] = kCtxHex;
  expected["grid_launch_id"] = 9;
  expected["is_load"] = true;
  expected["mem_space"] = 1;
  expected["opcode_id"] = 10;
  expected["pc"] = "0x55";
  expected["timestamp"] = 8;
  expected["trace_index"] = 7;
  nlohmann::json values = nlohmann::json::array();
  for (int lane = 0; lane < 32; lane++) {
    nlohmann::json lane_vals = nlohmann::json::array();
    for (int reg = 0; reg < 2; reg++) {  // only regs_needed = 2 emitted
      lane_vals.push_back(10u * lane + reg);
    }
    values.push_back(lane_vals);
  }
  expected["values"] = values;
  expected["warp"] = 6;

  EXPECT_EQ(parseRj(rec), expected);
}

TEST(SerializeRecordRapidjson, MemValueTrace_StoreOneRegPerLane) {
  mem_value_access_t mv{};
  mv.header.type = MSG_TYPE_MEM_VALUE_ACCESS;
  mv.warp_id = 0;
  mv.opcode_id = 2;
  mv.kernel_launch_id = 4;
  mv.pc = 0x10;
  mv.mem_space = 4;
  mv.is_load = 0;      // -> is_load false
  mv.access_size = 4;  // regs_needed = 1
  for (int lane = 0; lane < 32; lane++) {
    mv.values[lane][0] = lane + 1;
  }
  const TraceRecord rec = TraceRecord::create_mem_value_trace(testCtx(), "SASS", 0, 0, &mv);
  const nlohmann::json out = parseRj(rec);

  EXPECT_EQ(out["is_load"], nlohmann::json(false));
  EXPECT_EQ(out["access_size"], nlohmann::json(4));
  ASSERT_TRUE(out["values"].is_array());
  EXPECT_EQ(out["values"].size(), 32u);
  EXPECT_EQ(out["values"][0].size(), 1u);  // regs_needed = 1
  EXPECT_EQ(out["values"][5], (nlohmann::json{6}));
}

TEST(SerializeRecordRapidjson, MemValueTrace_RegsPerLaneClampedToFour) {
  mem_value_access_t mv{};
  mv.header.type = MSG_TYPE_MEM_VALUE_ACCESS;
  mv.is_load = 1;
  mv.access_size = 64;  // (64 + 3) / 4 = 16, clamped to 4
  for (int lane = 0; lane < 32; lane++) {
    for (int reg = 0; reg < 4; reg++) {
      mv.values[lane][reg] = reg;
    }
  }
  const TraceRecord rec = TraceRecord::create_mem_value_trace(testCtx(), "SASS", 0, 0, &mv);
  const nlohmann::json out = parseRj(rec);

  ASSERT_TRUE(out["values"].is_array());
  // Clamp keeps the inner array at 4, never reading past values[lane][3].
  EXPECT_EQ(out["values"][0].size(), 4u);
}

TEST(SerializeRecordRapidjson, TmaTrace_ReturnsEmptyDeferredToNlohmann) {
  tma_access_t t{};
  t.header.type = MSG_TYPE_TMA_ACCESS;
  const TraceRecord rec = TraceRecord::create_tma_trace(testCtx(), "SASS", 0, 0, &t, nullptr);

  // The rapidjson path does not yet handle tma; the live writer falls back to
  // nlohmann. D3 ports tma and flips this expectation.
  EXPECT_TRUE(cutracer::serialize_record_rapidjson(rec).empty());
}

TEST(SerializeRecordRapidjson, NullData_ReturnsEmpty) {
  TraceRecord rec{};
  rec.type = MSG_TYPE_REG_INFO;
  rec.data.reg_info = nullptr;
  EXPECT_TRUE(cutracer::serialize_record_rapidjson(rec).empty());
}

}  // namespace
