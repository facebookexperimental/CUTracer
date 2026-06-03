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
// library symbol. reg/mem/opcode/mem_value emit keys in lexicographic order so
// the output stays byte-identical to nlohmann; tma is not handled here (the
// caller falls back to nlohmann — see D3).

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

}  // namespace

namespace cutracer {

// Serialize a record into `sb` via rapidjson. Returns false for types not
// handled here (MSG_TYPE_TMA_ACCESS / unknown / null data) so the caller falls
// back to nlohmann. On success `sb` holds the complete NDJSON line (no trailing
// newline); the caller appends it directly (or, in ab mode, materializes a
// string to compare).
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
    default:
      return false;  // MSG_TYPE_TMA_ACCESS and unknown → nlohmann fallback
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
