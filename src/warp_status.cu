/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 *
 * See LICENSE file in the root directory for Meta's license terms.
 */

#include <stdio.h>

#include <map>
#include <nlohmann/json.hpp>
#include <set>
#include <string>

#include "analysis.h"
#include "env_config.h"
#include "log.h"
#include "warp_status.h"

extern std::map<uint64_t, std::pair<CUcontext, CUfunction>> kernel_launch_to_func_map;

// ── helpers ──────────────────────────────────────────────────────────────────

static InstructionInfo make_instruction_info(const TraceRecordMerged& instr,
                                             const std::map<int, std::string>* sass_map) {
  InstructionInfo info;
  info.pc = instr.reg.pc;
  info.has_mem = instr.has_mem;
  if (sass_map && sass_map->count(instr.reg.opcode_id)) {
    info.mnemonic = sass_map->at(instr.reg.opcode_id);
  }
  return info;
}

// Fold a sorted set of integer IDs into compact range strings (e.g., {1,2,3,5} -> {"1-3","5"}).
// Used by both text and JSON renderers to keep the range-folding rule in one place.
static std::vector<std::string> build_range_strings(const std::set<int>& ids) {
  std::vector<std::string> out;
  if (ids.empty()) return out;

  auto it = ids.begin();
  int range_start = *it;
  int range_end = *it;

  auto flush = [&]() {
    if (range_start == range_end) {
      out.push_back(std::to_string(range_start));
    } else {
      out.push_back(std::to_string(range_start) + "-" + std::to_string(range_end));
    }
  };

  for (++it; it != ids.end(); ++it) {
    if (*it == range_end + 1) {
      range_end = *it;
    } else {
      flush();
      range_start = range_end = *it;
    }
  }
  flush();
  return out;
}

static std::string format_ranges(const std::set<int>& ids) {
  auto parts = build_range_strings(ids);
  if (parts.empty()) return "none";
  std::string result;
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i) result += ", ";
    result += parts[i];
  }
  return result;
}

static nlohmann::json format_ranges_json(const std::set<int>& ids) {
  using json = nlohmann::json;
  json arr = json::array();
  for (const auto& s : build_range_strings(ids)) {
    arr.push_back(s);
  }
  return arr;
}

// ── collect ──────────────────────────────────────────────────────────────────

WarpStatusSummary collect_warp_status(CTXstate* ctx_state, uint64_t current_kernel_launch_id) {
  WarpStatusSummary summary;
  summary.kernel_launch_id = current_kernel_launch_id;
  time_t now = time(nullptr);

  // Resolve SASS map
  const std::map<int, std::string>* sass_map = nullptr;
  {
    auto func_iter = kernel_launch_to_func_map.find(current_kernel_launch_id);
    if (func_iter != kernel_launch_to_func_map.end()) {
      CUfunction f_func = func_iter->second.second;
      if (ctx_state->id_to_sass_map.count(f_func)) {
        sass_map = &ctx_state->id_to_sass_map[f_func];
      }
    }
  }

  // Warp statistics
  if (ctx_state->kernel_warp_tracking.count(current_kernel_launch_id)) {
    summary.has_stats = true;
    const KernelWarpStats& stats = ctx_state->kernel_warp_tracking[current_kernel_launch_id];
    summary.dims = stats.dimensions;
    summary.total_warps = stats.total_warps;
    summary.finished_warps = stats.finished_warps.size();
    summary.active_warps = ctx_state->active_warps.size();

    for (const WarpKey& key : stats.finished_warps) summary.finished_ids.insert(key.warp_id);
    for (const WarpKey& key : ctx_state->active_warps) summary.active_ids.insert(key.warp_id);

    std::set<int> all_seen;
    for (const WarpKey& key : stats.all_seen_warps) all_seen.insert(key.warp_id);
    for (uint32_t wid = 0; wid < stats.total_warps; wid++) {
      if (all_seen.count(static_cast<int>(wid)) == 0) {
        summary.never_executed_ids.insert(static_cast<int>(wid));
      }
    }
    summary.never_executed = summary.never_executed_ids.size();
  }

  // Per-warp entries
  for (const auto& warp_key : ctx_state->active_warps) {
    WarpStatusEntry entry;
    entry.key = warp_key;

    auto loop_iter = ctx_state->loop_states.find(warp_key);
    bool is_looping = false;

    if (loop_iter != ctx_state->loop_states.end()) {
      entry.loop_period = loop_iter->second.last_period;
      entry.repeat_cnt = loop_iter->second.repeat_cnt;
      if (loop_iter->second.loop_flag) {
        is_looping = true;
        entry.status = WarpStatusKind::LOOPING;
        auto seen_it = ctx_state->last_seen_time_by_warp.find(warp_key);
        if (seen_it != ctx_state->last_seen_time_by_warp.end()) {
          entry.last_seen_secs = now - seen_it->second;
        }
      }
    }

    // Inactive duration
    auto seen_iter = ctx_state->last_seen_time_by_warp.find(warp_key);
    if (seen_iter != ctx_state->last_seen_time_by_warp.end()) {
      entry.inactive_secs = now - seen_iter->second;
    }

    if (!is_looping) {
      bool is_barrier = false;
      auto itBar = ctx_state->last_is_defer_blocking_by_warp.find(warp_key);
      if (itBar != ctx_state->last_is_defer_blocking_by_warp.end()) {
        is_barrier = itBar->second;
      }
      entry.status = is_barrier ? WarpStatusKind::BARRIER : WarpStatusKind::PROGRESSING;

      // Last instruction from ring buffer
      const WarpLoopState* ls = (loop_iter != ctx_state->loop_states.end()) ? &loop_iter->second : nullptr;
      if (ls && ls->filled > 0 && sass_map) {
        int idx_last = (ls->head + (int)ls->filled + PC_HISTORY_LEN - 1) % PC_HISTORY_LEN;
        entry.last_instruction = make_instruction_info(ls->history[idx_last], sass_map);
      }
    }

    // Exit candidate
    auto exit_iter = ctx_state->exit_candidate_since_by_warp.find(warp_key);
    if (exit_iter != ctx_state->exit_candidate_since_by_warp.end()) {
      entry.exit_candidate_secs = static_cast<time_t>(now - exit_iter->second);
    }

    // Loop body
    if (is_looping && loop_iter != ctx_state->loop_states.end()) {
      const WarpLoopState& ls = loop_iter->second;
      if (!ls.current_loop.instructions.empty()) {
        size_t upper = std::min(static_cast<size_t>(ls.current_loop.period), ls.current_loop.instructions.size());
        entry.loop_body.reserve(upper);
        for (size_t i = 0; i < upper; ++i) {
          entry.loop_body.push_back(make_instruction_info(ls.current_loop.instructions[i], sass_map));
        }
      }
    }

    summary.entries.push_back(std::move(entry));
  }

  return summary;
}

// ── text output ──────────────────────────────────────────────────────────────

static void print_instruction_line(size_t index, const InstructionInfo& info, int pc_dec_width, int hex_nibbles_max) {
  loprintf("        [%*zu] PC %*lu; Offset %*lu /*0x%0*lx*/;  %s", 1, index, pc_dec_width, (unsigned long)info.pc,
           pc_dec_width, (unsigned long)info.pc, hex_nibbles_max, (unsigned long)info.pc, info.mnemonic.c_str());
  if (info.has_mem) {
    loprintf(" (has_mem)");
  }
  loprintf("\n");
}

static void compute_pc_widths(uint64_t pc, int& pc_dec_width, int& hex_nibbles_max) {
  int dec_w = 1;
  uint64_t td = pc;
  while (td >= 10) {
    td /= 10;
    dec_w++;
  }
  if (dec_w > pc_dec_width) pc_dec_width = dec_w;
  int nibbles = 1;
  if (pc != 0) {
    nibbles = 0;
    uint64_t th = pc;
    while (th) {
      th >>= 4;
      nibbles++;
    }
  }
  if (nibbles > hex_nibbles_max) hex_nibbles_max = nibbles;
}

static void finalize_hex_nibbles(int& hex_nibbles_max) {
  hex_nibbles_max = ((hex_nibbles_max + 3) / 4) * 4;
  if (hex_nibbles_max < 4) hex_nibbles_max = 4;
  if (hex_nibbles_max > 16) hex_nibbles_max = 16;
}

// Print the "Last: ..." line for a non-looping warp. Shared by BARRIER and PROGRESSING branches.
static void print_last_instruction_line(const std::optional<InstructionInfo>& last_instruction) {
  if (last_instruction.has_value()) {
    const auto& li = last_instruction.value();
    int pc_dec_width = 1;
    int hex_nibbles_max = 4;
    compute_pc_widths(li.pc, pc_dec_width, hex_nibbles_max);
    finalize_hex_nibbles(hex_nibbles_max);
    loprintf("      Last: [%*d] PC %*lu; Offset %*lu /*0x%0*lx*/;  %s", 1, 0, pc_dec_width, (unsigned long)li.pc,
             pc_dec_width, (unsigned long)li.pc, hex_nibbles_max, (unsigned long)li.pc, li.mnemonic.c_str());
    if (li.has_mem) loprintf(" (has_mem)");
  } else {
    loprintf("      Last: [%*d] PC %*lu; Offset %*lu /*0x%0*lx*/;  %s", 1, 0, 1, 0UL, 1, 0UL, 4, 0UL, "UNKNOWN");
  }
}

void print_warp_status_text(const WarpStatusSummary& summary) {
  if (summary.has_stats) {
    loprintf("==> WARP STATISTICS for kernel_launch_id=%lu:\n", summary.kernel_launch_id);
    loprintf("    Grid: <%u,%u,%u>, Block: <%u,%u,%u>\n", summary.dims.gridDimX, summary.dims.gridDimY,
             summary.dims.gridDimZ, summary.dims.blockDimX, summary.dims.blockDimY, summary.dims.blockDimZ);
    loprintf("\n");
    loprintf("    Summary:\n");
    loprintf("      Total warps:           %5zu (100.0%%)\n", summary.total_warps);
    loprintf("      Finished warps:        %5zu (%5.1f%%)\n", summary.finished_warps,
             summary.total_warps > 0 ? 100.0 * summary.finished_warps / summary.total_warps : 0.0);
    loprintf("      Active warps:          %5zu (%5.1f%%)\n", summary.active_warps,
             summary.total_warps > 0 ? 100.0 * summary.active_warps / summary.total_warps : 0.0);
    loprintf("      Never executed warps:  %5zu (%5.1f%%)\n", summary.never_executed,
             summary.total_warps > 0 ? 100.0 * summary.never_executed / summary.total_warps : 0.0);

    loprintf("\n");
    loprintf("    Warp ID Ranges:\n");
    loprintf("      Finished:       %s\n", format_ranges(summary.finished_ids).c_str());
    loprintf("      Active:         %s\n", format_ranges(summary.active_ids).c_str());
    loprintf("      Never executed: %s\n", format_ranges(summary.never_executed_ids).c_str());
    loprintf("    -----------------------------------------------------------------------\n");
  }

  if (summary.entries.empty()) {
    loprintf("==> WARP STATUS: No active warps for kernel_launch_id=%lu\n", summary.kernel_launch_id);
    return;
  }

  loprintf("==> WARP STATUS SUMMARY for kernel_launch_id=%lu (%zu active warps):\n", summary.kernel_launch_id,
           summary.entries.size());
  loprintf("    Format: WarpID[CTA_x,y,z] - LoopStatus - Activity\n");
  loprintf("    -----------------------------------------------------------------------\n");

  for (const auto& entry : summary.entries) {
    loprintf("    Warp%d[%d,%d,%d]: ", entry.key.warp_id, entry.key.cta_id_x, entry.key.cta_id_y, entry.key.cta_id_z);

    if (entry.status == WarpStatusKind::LOOPING) {
      loprintf("LOOPING(inactive=%lds, period=%d, repeat=%d) last_seen=%lds ", entry.inactive_secs, entry.loop_period,
               entry.repeat_cnt, entry.last_seen_secs);
    } else if (entry.status == WarpStatusKind::BARRIER) {
      loprintf("BARRIER(inactive=%lds) no_loop(period=%d, repeat=%d)\n", entry.inactive_secs, entry.loop_period,
               entry.repeat_cnt);
      print_last_instruction_line(entry.last_instruction);
    } else {
      loprintf("PROGRESSING(inactive=%lds) no_loop(period=%d, repeat=%d)\n", entry.inactive_secs, entry.loop_period,
               entry.repeat_cnt);
      print_last_instruction_line(entry.last_instruction);
    }

    if (entry.exit_candidate_secs.has_value()) {
      loprintf("- EXIT_CANDIDATE(%lds)", entry.exit_candidate_secs.value());
    }

    loprintf("\n");

    // Loop body
    if (entry.status == WarpStatusKind::LOOPING && !entry.loop_body.empty()) {
      loprintf("      Loop Body (%zu instructions):\n", entry.loop_body.size());
      int pc_dec_width = 1;
      int hex_nibbles_max = 4;
      for (const auto& instr : entry.loop_body) {
        compute_pc_widths(instr.pc, pc_dec_width, hex_nibbles_max);
      }
      finalize_hex_nibbles(hex_nibbles_max);
      for (size_t i = 0; i < entry.loop_body.size(); ++i) {
        print_instruction_line(i, entry.loop_body[i], pc_dec_width, hex_nibbles_max);
      }
    }
  }
  loprintf("    -----------------------------------------------------------------------\n");
}

// ── JSON output ──────────────────────────────────────────────────────────────

static nlohmann::json instruction_to_json(const InstructionInfo& info) {
  using json = nlohmann::json;
  json j;
  j["pc"] = info.pc;

  std::ostringstream hex_oss;
  hex_oss << "0x" << std::hex << info.pc;
  j["offset_hex"] = hex_oss.str();

  j["mnemonic"] = info.mnemonic;
  j["has_mem"] = info.has_mem;
  return j;
}

void write_warp_status_json(const WarpStatusSummary& summary, const std::string& basename) {
  using json = nlohmann::json;

  json root;
  root["kernel_launch_id"] = summary.kernel_launch_id;

  if (summary.has_stats) {
    json warp_stats;
    warp_stats["grid"] = {{"x", summary.dims.gridDimX}, {"y", summary.dims.gridDimY}, {"z", summary.dims.gridDimZ}};
    warp_stats["block"] = {{"x", summary.dims.blockDimX}, {"y", summary.dims.blockDimY}, {"z", summary.dims.blockDimZ}};
    warp_stats["summary"] = {
        {"total_warps", summary.total_warps},
        {"finished_warps", summary.finished_warps},
        {"finished_warps_pct", summary.total_warps > 0 ? 100.0 * summary.finished_warps / summary.total_warps : 0.0},
        {"active_warps", summary.active_warps},
        {"active_warps_pct", summary.total_warps > 0 ? 100.0 * summary.active_warps / summary.total_warps : 0.0},
        {"never_executed_warps", summary.never_executed},
        {"never_executed_warps_pct",
         summary.total_warps > 0 ? 100.0 * summary.never_executed / summary.total_warps : 0.0}};
    warp_stats["warp_id_ranges"] = {{"finished", format_ranges_json(summary.finished_ids)},
                                    {"active", format_ranges_json(summary.active_ids)},
                                    {"never_executed", format_ranges_json(summary.never_executed_ids)}};
    root["warp_statistics"] = warp_stats;
  }

  json warps_array = json::array();
  for (const auto& entry : summary.entries) {
    json warp_obj;
    warp_obj["warp_id"] = entry.key.warp_id;
    warp_obj["cta"] = {{"x", entry.key.cta_id_x}, {"y", entry.key.cta_id_y}, {"z", entry.key.cta_id_z}};

    const char* status_str = "PROGRESSING";
    if (entry.status == WarpStatusKind::LOOPING)
      status_str = "LOOPING";
    else if (entry.status == WarpStatusKind::BARRIER)
      status_str = "BARRIER";
    warp_obj["status"] = status_str;

    warp_obj["loop"] = {{"period", entry.loop_period}, {"repeat", entry.repeat_cnt}};
    warp_obj["inactive_secs"] = entry.inactive_secs;

    if (entry.status == WarpStatusKind::LOOPING) {
      warp_obj["last_seen_secs"] = entry.last_seen_secs;
    }

    if (entry.last_instruction.has_value()) {
      warp_obj["last_instruction"] = instruction_to_json(entry.last_instruction.value());
    }

    if (entry.exit_candidate_secs.has_value()) {
      warp_obj["exit_candidate_secs"] = entry.exit_candidate_secs.value();
    }

    if (!entry.loop_body.empty()) {
      json loop_body;
      loop_body["instruction_count"] = entry.loop_body.size();
      json instructions = json::array();
      for (size_t i = 0; i < entry.loop_body.size(); ++i) {
        json instr_obj = instruction_to_json(entry.loop_body[i]);
        instr_obj["index"] = i;
        instructions.push_back(instr_obj);
      }
      loop_body["instructions"] = instructions;
      warp_obj["loop_body"] = loop_body;
    }

    warps_array.push_back(warp_obj);
  }
  root["active_warps"] = warps_array;

  std::string filename = basename + "_warp_status_summary.ndjson";

  // Append one JSON object per snapshot (NDJSON). Multiple invocations for the same
  // kernel launch (e.g., periodic hang detection) accumulate as separate lines.
  FILE* fp = fopen(filename.c_str(), "a");
  if (fp) {
    std::string json_str = root.dump();
    fwrite(json_str.c_str(), 1, json_str.size(), fp);
    fwrite("\n", 1, 1, fp);
    fclose(fp);
    loprintf("Warp status JSON appended to %s\n", filename.c_str());
  } else {
    loprintf("WARNING: Failed to append warp status JSON to %s\n", filename.c_str());
  }
}
