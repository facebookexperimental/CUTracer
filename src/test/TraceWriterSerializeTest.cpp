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
  r.active_mask = 0xffffffff;
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
  expected["active_mask"] = "0xffffffff";
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
  r.active_mask = 0xf;  // first 4 lanes active

  RegIndices idx;
  idx.reg_indices = {5};
  idx.ureg_indices = {1, 2, 3};

  const TraceRecord rec = TraceRecord::create_reg_trace(testCtx(), "SASS", 12, 100, &r, &idx);

  nlohmann::json expected;
  expected["type"] = "reg_trace";
  expected["active_mask"] = "0xf";
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
  m.active_mask = 0x80000001;  // lane 0 and 31 active
  for (int i = 0; i < 32; i++) {
    m.addrs[i] = 0x100000000ull + i;  // > 2^32 to exercise Uint64 width
  }
  const TraceRecord rec = TraceRecord::create_mem_trace(testCtx(), "SASS", 3, 4, &m);

  nlohmann::json expected;
  expected["type"] = "mem_addr_trace";
  expected["ipoint"] = "B";
  expected["active_mask"] = "0x80000001";
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
  o.active_mask = 0x55555555;  // alternating lanes
  const TraceRecord rec = TraceRecord::create_opcode_trace(testCtx(), "SASS", 5, 6, &o);

  nlohmann::json expected;
  expected["type"] = "opcode_only";
  expected["active_mask"] = "0x55555555";
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
  mv.active_mask = 0xffffffff;
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
  expected["active_mask"] = "0xffffffff";
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

// With no parsed transfer info, only the base record is emitted (no
// tma_transfer_info / desc_addr) -- matches nlohmann's `if (info != nullptr)`.
TEST(SerializeRecordRapidjson, TmaTrace_NoTransferInfo_BaseFieldsOnly) {
  tma_access_t t{};
  t.header.type = MSG_TYPE_TMA_ACCESS;
  t.cta_id_x = 2;
  t.cta_id_y = 3;
  t.cta_id_z = 4;
  t.warp_id = 5;
  t.opcode_id = 11;
  t.kernel_launch_id = 6;
  t.pc = 0x77;
  t.tma_param_size = 128;
  t.active_mask = 0xffffffff;
  const TraceRecord rec = TraceRecord::create_tma_trace(testCtx(), "SASS", 9, 10, &t, nullptr);

  nlohmann::json expected;
  expected["type"] = "tma_trace";
  expected["active_mask"] = "0xffffffff";
  expected["cta"] = {2, 3, 4};
  expected["ctx"] = kCtxHex;
  expected["grid_launch_id"] = 6;
  expected["opcode_id"] = 11;
  expected["pc"] = "0x77";
  expected["timestamp"] = 10;
  expected["tma_param_size"] = 128;
  expected["trace_index"] = 9;
  expected["warp"] = 5;

  const nlohmann::json out = parseRj(rec);
  EXPECT_EQ(out, expected);
  EXPECT_FALSE(out.contains("tma_transfer_info"));
  EXPECT_FALSE(out.contains("desc_addr"));
}

// Tiled tensor with a shared-memory destination (UTMALDG) and a valid mbar:
// exercises the dst sub-object (with mbar keys), the tensor/tiled nesting, the
// enum-string tables, and desc_addr from the tiled global address.
TEST(SerializeRecordRapidjson, TmaTrace_TiledTensorWithSharedDst) {
  tma_access_t t{};
  t.header.type = MSG_TYPE_TMA_ACCESS;
  t.tma_param_size = 64;

  TMATransferInfo_t info{};
  info.is_tensor = true;
  info.transfer_count = 3;
  info.byte_count = 0x100000000ull;  // > 2^32 -> Uint64
  info.dst_memspace = InstrType::MemorySpace::SHARED;
  info.dst.shared.data_address = 0x400;
  info.dst.shared.data_address_offset = 0x10;
  info.dst.shared.data_address_cluster_cta_id = 1;
  info.dst.shared.is_mbar_valid = true;
  info.dst.shared.mbar_address = 0x800;
  info.dst.shared.mbar_address_offset = 0x20;
  info.dst.shared.mbar_address_cluster_cta_id = 2;
  info.tensor.dim = 2;
  info.tensor.mode = InstrType::TMALoadMode::TILED;
  auto& tiled = info.tensor.map.tiled;
  tiled.tensorDataType = static_cast<CUtensorMapDataType>(2);  // CUtensorMapDataTypeStr[2]
  tiled.tensorRank = 2;
  tiled.globalAddress = reinterpret_cast<void*>(0xDEADBEEF000ull);
  for (int i = 0; i < InstrType::MAX_TMA_DIM; i++) {
    tiled.globalDim[i] = 100 + i;
  }
  tiled.swizzle = static_cast<CUtensorMapSwizzle>(3);  // CUtensorMapSwizzleStr[3]

  const TraceRecord rec = TraceRecord::create_tma_trace(testCtx(), "SASS", 1, 2, &t, &info);
  const nlohmann::json out = parseRj(rec);

  ASSERT_TRUE(out.contains("tma_transfer_info"));
  const nlohmann::json& ti = out["tma_transfer_info"];
  EXPECT_EQ(ti["is_tensor"], nlohmann::json(true));
  EXPECT_EQ(ti["byte_count"], nlohmann::json(0x100000000ull));
  EXPECT_EQ(ti["dst_memspace"],
            nlohmann::json(InstrType::MemorySpaceStr[static_cast<int>(InstrType::MemorySpace::SHARED)]));

  ASSERT_TRUE(ti.contains("dst"));
  EXPECT_EQ(ti["dst"]["data_address"], nlohmann::json(0x400));
  EXPECT_EQ(ti["dst"]["is_mbar_valid"], nlohmann::json(true));
  EXPECT_EQ(ti["dst"]["mbar_address"], nlohmann::json(0x800));
  EXPECT_FALSE(ti.contains("src"));  // src memspace is not SHARED

  ASSERT_TRUE(ti.contains("tensor"));
  EXPECT_EQ(ti["tensor"]["mode"], nlohmann::json("TILED"));
  ASSERT_TRUE(ti["tensor"].contains("tiled"));
  const nlohmann::json& tj = ti["tensor"]["tiled"];
  EXPECT_EQ(tj["data_type"], nlohmann::json(InstrType::CUtensorMapDataTypeStr[2]));
  EXPECT_EQ(tj["data_type_id"], nlohmann::json(2));
  EXPECT_EQ(tj["swizzle"], nlohmann::json(InstrType::CUtensorMapSwizzleStr[3]));
  EXPECT_EQ(tj["global_address"], nlohmann::json("0xdeadbeef000"));
  ASSERT_TRUE(tj["global_dim"].is_array());
  EXPECT_EQ(tj["global_dim"].size(), static_cast<size_t>(InstrType::MAX_TMA_DIM));
  EXPECT_EQ(tj["global_dim"][0], nlohmann::json(100));

  EXPECT_EQ(out["desc_addr"], nlohmann::json("0xdeadbeef000"));  // from tiled global address
}

// Shared-memory source (UTMASTG) with an invalid mbar: the mbar keys are
// omitted, dst is absent, and a non-tensor non-bulk transfer has no desc_addr.
TEST(SerializeRecordRapidjson, TmaTrace_SharedSrcNoMbar) {
  tma_access_t t{};
  t.header.type = MSG_TYPE_TMA_ACCESS;
  t.tma_param_size = 16;

  TMATransferInfo_t info{};
  info.src_memspace = InstrType::MemorySpace::SHARED;
  info.src.shared.data_address = 0x10;
  info.src.shared.is_mbar_valid = false;

  const TraceRecord rec = TraceRecord::create_tma_trace(testCtx(), "SASS", 0, 0, &t, &info);
  const nlohmann::json out = parseRj(rec);
  const nlohmann::json& ti = out["tma_transfer_info"];

  ASSERT_TRUE(ti.contains("src"));
  EXPECT_EQ(ti["src"]["data_address"], nlohmann::json(0x10));
  EXPECT_EQ(ti["src"]["is_mbar_valid"], nlohmann::json(false));
  EXPECT_FALSE(ti["src"].contains("mbar_address"));  // omitted when invalid
  EXPECT_FALSE(ti.contains("dst"));
  EXPECT_FALSE(ti.contains("tensor"));
  EXPECT_FALSE(out.contains("desc_addr"));
}

// Bulk (non-tensor) copy between global spaces: no shared sub-objects, no
// tensor, and desc_addr falls back to the bulk source address.
TEST(SerializeRecordRapidjson, TmaTrace_BulkNonTensorDescAddrFromBulk) {
  tma_access_t t{};
  t.header.type = MSG_TYPE_TMA_ACCESS;
  t.tma_param_size = 8;

  TMATransferInfo_t info{};
  info.is_bulk = true;
  info.is_tensor = false;
  info.src_memspace = InstrType::MemorySpace::GLOBAL;
  info.dst_memspace = InstrType::MemorySpace::GLOBAL;
  info.src.global.bulk_copy_address = 0x5000;

  const TraceRecord rec = TraceRecord::create_tma_trace(testCtx(), "SASS", 0, 0, &t, &info);
  const nlohmann::json out = parseRj(rec);
  const nlohmann::json& ti = out["tma_transfer_info"];

  EXPECT_EQ(ti["is_bulk"], nlohmann::json(true));
  EXPECT_FALSE(ti.contains("dst"));
  EXPECT_FALSE(ti.contains("src"));
  EXPECT_FALSE(ti.contains("tensor"));
  EXPECT_EQ(out["desc_addr"], nlohmann::json("0x5000"));  // from bulk_copy_address
}

TEST(SerializeRecordRapidjson, NullData_ReturnsEmpty) {
  TraceRecord rec{};
  rec.type = MSG_TYPE_REG_INFO;
  rec.data.reg_info = nullptr;
  EXPECT_TRUE(cutracer::serialize_record_rapidjson(rec).empty());
}

}  // namespace
