/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 *
 * See LICENSE file in the root directory for Meta's license terms.
 */

#pragma once

#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "analysis.h"

enum class WarpStatusKind { LOOPING, BARRIER, PROGRESSING };

struct InstructionInfo {
  uint64_t pc = 0;
  std::string mnemonic = "UNKNOWN";
  bool has_mem = false;
};

struct WarpStatusEntry {
  WarpKey key;
  WarpStatusKind status = WarpStatusKind::PROGRESSING;
  int loop_period = 0;
  int repeat_cnt = 0;
  time_t inactive_secs = 0;
  time_t last_seen_secs = 0;
  std::optional<time_t> exit_candidate_secs;
  std::optional<InstructionInfo> last_instruction;
  std::vector<InstructionInfo> loop_body;
};

struct WarpStatusSummary {
  uint64_t kernel_launch_id = 0;
  KernelDimensions dims = {};
  size_t total_warps = 0;
  size_t finished_warps = 0;
  size_t active_warps = 0;
  size_t never_executed = 0;
  std::set<int> finished_ids;
  std::set<int> active_ids;
  std::set<int> never_executed_ids;
  bool has_stats = false;
  std::vector<WarpStatusEntry> entries;
};

WarpStatusSummary collect_warp_status(CTXstate* ctx_state, uint64_t current_kernel_launch_id);

void print_warp_status_text(const WarpStatusSummary& summary);

void write_warp_status_json(const WarpStatusSummary& summary, const std::string& basename);
