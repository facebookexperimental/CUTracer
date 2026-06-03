/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

// rapidjson streaming serializer for trace records: the hot-path NDJSON writer
// (rj_serialize_to_buffer, used by TraceWriter::write_json_format) and a
// test/diagnostic seam (serialize_record_rapidjson).
//
// Split out of trace_writer.cpp so it can be compiled into host unit tests
// without pulling in libnvbit's CUDA-driver-bound global initializer: this TU
// needs only nvbit *headers* (for TraceRecord's field types), never any nvbit
// library symbol (the InstrType enum-string tables tma uses are `constexpr` in
// nvbit headers, not lib symbols). reg/mem/opcode/mem_value emit keys in
// lexicographic order so the output stays byte-identical to nlohmann; tma emits
// in natural order (validated semantically, since no consumer depends on key
// order — matching the migration's contract).

#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <string>

#include "common.h"
#include "trace_rapidjson_internal.h"
#include "trace_serialize.h"
#include "trace_writer.h"

namespace {

using JW = rapidjson::Writer<rapidjson::StringBuffer>;

// Emit `"key":"0x<hex>"` matching std::hex (lowercase, no leading zeros).
void rj_key_hex(JW& w, const char* key, uint64_t v) {
  char buf[2 + 16 + 1];
  const int n = snprintf(buf, sizeof(buf), "0x%" PRIx64, v);
  w.Key(key);
  w.String(buf, static_cast<rapidjson::SizeType>(n));
}

void rj_cta(JW& w, int x, int y, int z) {
  w.Key("cta");
  w.StartArray();
  w.Int(x);
  w.Int(y);
  w.Int(z);
  w.EndArray();
}

void rj_reg(JW& w, const TraceRecord& rec, const reg_info_t* r, const RegIndices* idx) {
  w.StartObject();
  rj_cta(w, r->cta_id_x, r->cta_id_y, r->cta_id_z);
  rj_key_hex(w, "ctx", reinterpret_cast<uintptr_t>(rec.context));
  w.Key("grid_launch_id");
  w.Uint64(r->kernel_launch_id);
  w.Key("opcode_id");
  w.Int(r->opcode_id);
  rj_key_hex(w, "pc", r->pc);
  w.Key("regs");
  w.StartArray();
  for (int ri = 0; ri < r->num_regs; ri++) {  // transpose: regs[reg][thread]
    w.StartArray();
    for (int t = 0; t < 32; t++) {
      w.Uint(r->reg_vals[t][ri]);
    }
    w.EndArray();
  }
  w.EndArray();
  if (idx != nullptr && !idx->reg_indices.empty()) {
    w.Key("regs_indices");
    w.StartArray();
    for (uint8_t v : idx->reg_indices) {
      w.Uint(v);
    }
    w.EndArray();
  }
  w.Key("timestamp");
  w.Uint64(rec.timestamp);
  w.Key("trace_index");
  w.Uint64(rec.trace_index);
  w.Key("type");
  w.String("reg_trace");
  if (r->num_uregs > 0) {
    w.Key("uregs");
    w.StartArray();
    for (int i = 0; i < r->num_uregs; i++) {
      w.Uint(r->ureg_vals[i]);
    }
    w.EndArray();
    if (idx != nullptr && !idx->ureg_indices.empty()) {
      w.Key("uregs_indices");
      w.StartArray();
      for (uint8_t v : idx->ureg_indices) {
        w.Uint(v);
      }
      w.EndArray();
    }
  }
  w.Key("warp");
  w.Int(r->warp_id);
  w.EndObject();
}

void rj_mem_addr(JW& w, const TraceRecord& rec, const mem_addr_access_t* m) {
  w.StartObject();
  w.Key("addrs");
  w.StartArray();
  for (int i = 0; i < 32; i++) {
    w.Uint64(m->addrs[i]);
  }
  w.EndArray();
  rj_cta(w, m->cta_id_x, m->cta_id_y, m->cta_id_z);
  rj_key_hex(w, "ctx", reinterpret_cast<uintptr_t>(rec.context));
  w.Key("grid_launch_id");
  w.Uint64(m->kernel_launch_id);
  w.Key("ipoint");
  w.String("B");
  w.Key("opcode_id");
  w.Int(m->opcode_id);
  rj_key_hex(w, "pc", m->pc);
  w.Key("timestamp");
  w.Uint64(rec.timestamp);
  w.Key("trace_index");
  w.Uint64(rec.trace_index);
  w.Key("type");
  w.String("mem_addr_trace");
  w.Key("warp");
  w.Int(m->warp_id);
  w.EndObject();
}

void rj_opcode(JW& w, const TraceRecord& rec, const opcode_only_t* o) {
  w.StartObject();
  rj_cta(w, o->cta_id_x, o->cta_id_y, o->cta_id_z);
  rj_key_hex(w, "ctx", reinterpret_cast<uintptr_t>(rec.context));
  w.Key("grid_launch_id");
  w.Uint64(o->kernel_launch_id);
  w.Key("opcode_id");
  w.Int(o->opcode_id);
  rj_key_hex(w, "pc", o->pc);
  w.Key("timestamp");
  w.Uint64(rec.timestamp);
  w.Key("trace_index");
  w.Uint64(rec.trace_index);
  w.Key("type");
  w.String("opcode_only");
  w.Key("warp");
  w.Int(o->warp_id);
  w.EndObject();
}

void rj_mem_value(JW& w, const TraceRecord& rec, const mem_value_access_t* m) {
  int regs_needed = (m->access_size + 3) / 4;
  if (regs_needed > 4) {
    regs_needed = 4;
  }
  w.StartObject();
  w.Key("access_size");
  w.Int(m->access_size);
  w.Key("addrs");
  w.StartArray();
  for (int i = 0; i < 32; i++) {
    w.Uint64(m->addrs[i]);
  }
  w.EndArray();
  rj_cta(w, m->cta_id_x, m->cta_id_y, m->cta_id_z);
  rj_key_hex(w, "ctx", reinterpret_cast<uintptr_t>(rec.context));
  w.Key("grid_launch_id");
  w.Uint64(m->kernel_launch_id);
  w.Key("ipoint");
  w.String("A");
  w.Key("is_load");
  w.Bool(m->is_load == 1);
  w.Key("mem_space");
  w.Int(m->mem_space);
  w.Key("opcode_id");
  w.Int(m->opcode_id);
  rj_key_hex(w, "pc", m->pc);
  w.Key("timestamp");
  w.Uint64(rec.timestamp);
  w.Key("trace_index");
  w.Uint64(rec.trace_index);
  w.Key("type");
  w.String("mem_value_trace");
  w.Key("values");
  w.StartArray();
  for (int lane = 0; lane < 32; lane++) {
    w.StartArray();
    for (int r = 0; r < regs_needed; r++) {
      w.Uint(m->values[lane][r]);
    }
    w.EndArray();
  }
  w.EndArray();
  w.Key("warp");
  w.Int(m->warp_id);
  w.EndObject();
}

// ---- tma_trace -------------------------------------------------------------
// Mirrors the nlohmann serialize_tma_* family in trace_writer.cpp, in natural
// (source) order. Uses only nvbit header types and the `constexpr` InstrType
// enum-string tables, so it stays free of any nvbit library symbol.

// Emit `"key":"<table[idx]>"`, falling back to the integer (as a string) when
// out of range — matches enum_str() in trace_writer.cpp.
template <size_t N>
void rj_key_enum(JW& w, const char* key, const char* const (&table)[N], int idx) {
  w.Key(key);
  if (idx < 0 || static_cast<size_t>(idx) >= N) {
    const std::string s = std::to_string(idx);
    w.String(s.c_str(), static_cast<rapidjson::SizeType>(s.size()));
  } else {
    w.String(table[idx]);
  }
}

void rj_key_u64_arr(JW& w, const char* key, const uint64_t* p, int n) {
  w.Key(key);
  w.StartArray();
  for (int i = 0; i < n; i++) {
    w.Uint64(p[i]);
  }
  w.EndArray();
}

void rj_key_u32_arr(JW& w, const char* key, const uint32_t* p, int n) {
  w.Key(key);
  w.StartArray();
  for (int i = 0; i < n; i++) {
    w.Uint(p[i]);
  }
  w.EndArray();
}

void rj_key_i32_arr(JW& w, const char* key, const int32_t* p, int n) {
  w.Key(key);
  w.StartArray();
  for (int i = 0; i < n; i++) {
    w.Int(p[i]);
  }
  w.EndArray();
}

// Shared-memory address sub-object (UTMALDG dst / UTMASTG src). Read the
// `.shared` union member only — callers guard on the memspace being SHARED.
void rj_tma_shared_addr(JW& w, const char* key, const TMAAddress_t& addr) {
  w.Key(key);
  w.StartObject();
  w.Key("data_address");
  w.Uint(addr.shared.data_address);
  w.Key("data_address_offset");
  w.Uint(addr.shared.data_address_offset);
  w.Key("data_address_cluster_cta_id");
  w.Uint(addr.shared.data_address_cluster_cta_id);
  w.Key("is_mbar_valid");
  w.Bool(addr.shared.is_mbar_valid);
  if (addr.shared.is_mbar_valid) {
    w.Key("mbar_address");
    w.Uint(addr.shared.mbar_address);
    w.Key("mbar_address_offset");
    w.Uint(addr.shared.mbar_address_offset);
    w.Key("mbar_address_cluster_cta_id");
    w.Uint(addr.shared.mbar_address_cluster_cta_id);
  }
  w.EndObject();
}

// Tiled tensor map. Only call when is_tensor && tensor.mode == TILED.
void rj_tma_tiled(JW& w, const TMATransferInfo_t& info) {
  const auto& tiled = info.tensor.map.tiled;
  w.Key("tiled");
  w.StartObject();
  rj_key_enum(w, "data_type", InstrType::CUtensorMapDataTypeStr, static_cast<int>(tiled.tensorDataType));
  w.Key("data_type_id");
  w.Int(static_cast<int>(tiled.tensorDataType));
  w.Key("rank");
  w.Uint(tiled.tensorRank);
  rj_key_hex(w, "global_address", reinterpret_cast<uintptr_t>(tiled.globalAddress));
  rj_key_u64_arr(w, "global_dim", tiled.globalDim, InstrType::MAX_TMA_DIM);
  rj_key_u64_arr(w, "global_strides", tiled.globalStrides, InstrType::MAX_TMA_DIM - 1);
  rj_key_u32_arr(w, "box_dim", tiled.boxDim, InstrType::MAX_TMA_DIM);
  rj_key_u32_arr(w, "element_strides", tiled.elementStrides, InstrType::MAX_TMA_DIM);
  rj_key_enum(w, "interleave", InstrType::CUtensorMapInterleaveStr, static_cast<int>(tiled.interleave));
  w.Key("interleave_id");
  w.Int(static_cast<int>(tiled.interleave));
  rj_key_enum(w, "swizzle", InstrType::CUtensorMapSwizzleStr, static_cast<int>(tiled.swizzle));
  w.Key("swizzle_id");
  w.Int(static_cast<int>(tiled.swizzle));
  rj_key_enum(w, "l2_promotion", InstrType::CUtensorMapL2promotionStr, static_cast<int>(tiled.l2Promotion));
  rj_key_enum(w, "oob_fill", InstrType::CUtensorMapFloatOOBfillStr, static_cast<int>(tiled.oobFill));
  w.EndObject();
}

void rj_tma_transfer_info(JW& w, const TMATransferInfo_t& info) {
  w.Key("is_bulk");
  w.Bool(info.is_bulk);
  w.Key("is_tensor");
  w.Bool(info.is_tensor);
  w.Key("is_prefetch");
  w.Bool(info.is_prefetch);
  w.Key("is_multicast");
  w.Bool(info.is_multicast);
  w.Key("transfer_count");
  w.Uint(info.transfer_count);
  w.Key("transfer_size");
  w.Uint(info.transfer_size);
  w.Key("byte_count");
  w.Uint64(info.byte_count);
  w.Key("multicast_cta_mask");
  w.Uint(info.multicast_cta_mask);
  rj_key_enum(w, "src_memspace", InstrType::MemorySpaceStr, static_cast<int>(info.src_memspace));
  rj_key_enum(w, "dst_memspace", InstrType::MemorySpaceStr, static_cast<int>(info.dst_memspace));

  if (info.dst_memspace == InstrType::MemorySpace::SHARED ||
      info.dst_memspace == InstrType::MemorySpace::DISTRIBUTED_SHARED) {
    rj_tma_shared_addr(w, "dst", info.dst);
  }
  if (info.src_memspace == InstrType::MemorySpace::SHARED ||
      info.src_memspace == InstrType::MemorySpace::DISTRIBUTED_SHARED) {
    rj_tma_shared_addr(w, "src", info.src);
  }

  if (!info.is_tensor) {
    return;
  }
  w.Key("tensor");
  w.StartObject();
  w.Key("dim");
  w.Uint(info.tensor.dim);
  rj_key_enum(w, "mode", InstrType::TMALoadModeStr, static_cast<int>(info.tensor.mode));
  rj_key_i32_arr(w, "coords", info.tensor.coords, InstrType::MAX_TMA_DIM);
  w.Key("intended_transfer_count");
  w.Uint(info.tensor.intended_transfer_count);
  w.Key("oob_transfer_count");
  w.Uint(info.tensor.oob_transfer_count);
  if (info.tensor.mode == InstrType::TMALoadMode::TILED) {
    rj_tma_tiled(w, info);
  }
  w.EndObject();
}

void rj_tma(JW& w, const TraceRecord& rec, const tma_access_t* t, const TMATransferInfo_t* info) {
  w.StartObject();
  rj_cta(w, t->cta_id_x, t->cta_id_y, t->cta_id_z);
  rj_key_hex(w, "ctx", reinterpret_cast<uintptr_t>(rec.context));
  w.Key("grid_launch_id");
  w.Uint64(t->kernel_launch_id);
  w.Key("opcode_id");
  w.Int(t->opcode_id);
  rj_key_hex(w, "pc", t->pc);
  w.Key("timestamp");
  w.Uint64(rec.timestamp);
  w.Key("tma_param_size");
  w.Uint(t->tma_param_size);
  w.Key("trace_index");
  w.Uint64(rec.trace_index);
  w.Key("type");
  w.String("tma_trace");
  w.Key("warp");
  w.Int(t->warp_id);
  if (info != nullptr) {
    w.Key("tma_transfer_info");
    w.StartObject();
    rj_tma_transfer_info(w, *info);
    w.EndObject();
    // Stable per-tensor key for downstream dedup: parsed global tensor address
    // for tiled tensors, else the raw bulk source address.
    if (info->is_tensor && info->tensor.mode == InstrType::TMALoadMode::TILED) {
      rj_key_hex(w, "desc_addr", reinterpret_cast<uintptr_t>(info->tensor.map.tiled.globalAddress));
    } else if (info->is_bulk) {
      rj_key_hex(w, "desc_addr", info->src.global.bulk_copy_address);
    }
  }
  w.EndObject();
}

}  // namespace

namespace cutracer {

// Serialize a record into `sb` via rapidjson. Returns false only for unknown
// types or null data so the caller falls back to nlohmann. On success `sb`
// holds the complete NDJSON line (no trailing newline); the caller appends it
// directly (or, in ab mode, materializes a string to compare).
bool rj_serialize_to_buffer(const TraceRecord& rec, rapidjson::StringBuffer& sb) {
  sb.Clear();  // reuse capacity across records
  JW w(sb);
  switch (rec.type) {
    case MSG_TYPE_REG_INFO:
      if (rec.data.reg_info == nullptr) {
        return false;
      }
      rj_reg(w, rec, rec.data.reg_info, rec.reg_indices);
      break;
    case MSG_TYPE_MEM_ADDR_ACCESS:
      if (rec.data.mem_access == nullptr) {
        return false;
      }
      rj_mem_addr(w, rec, rec.data.mem_access);
      break;
    case MSG_TYPE_MEM_VALUE_ACCESS:
      if (rec.data.mem_value_access == nullptr) {
        return false;
      }
      rj_mem_value(w, rec, rec.data.mem_value_access);
      break;
    case MSG_TYPE_OPCODE_ONLY:
      if (rec.data.opcode_only == nullptr) {
        return false;
      }
      rj_opcode(w, rec, rec.data.opcode_only);
      break;
    case MSG_TYPE_TMA_ACCESS:
      if (rec.data.tma_access == nullptr) {
        return false;
      }
      rj_tma(w, rec, rec.data.tma_access, rec.tma_info);
      break;
    default:
      return false;  // unknown type → nlohmann fallback
  }
  return true;
}

// Test/diagnostic seam (declared in trace_serialize.h): serialize one record
// through the rapidjson path, bypassing the CUTRACER_JSON_ENGINE dispatch and
// file IO. Returns "" for types the rapidjson path doesn't handle (tma / null /
// unknown), which fall back to nlohmann in the live writer.
std::string serialize_record_rapidjson(const TraceRecord& record) {
  rapidjson::StringBuffer sb;
  if (!rj_serialize_to_buffer(record, sb)) {
    return std::string();
  }
  return std::string(sb.GetString(), sb.GetSize());
}

}  // namespace cutracer
