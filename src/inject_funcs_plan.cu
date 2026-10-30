/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#include <cuda_runtime.h>
#include <stdarg.h>

#include "plan_device.h"
#include "plan_selection.h"
#include "utils/channel.hpp"
#include "utils/utils.h"

static_assert(sizeof(unsigned int) == 4, "NVBit register arguments require 32-bit unsigned words");

extern "C" __device__ __noinline__ void instrument_plan_values(int pred, uint64_t control_address,
                                                               uint64_t channel_address, uint32_t point_id,
                                                               uint32_t word_count, ...) {
  if (!control_address) return;
  auto* control = reinterpret_cast<PlanLaunchControl*>(control_address);
  if (point_id >= control->point_count || word_count > PLAN_CAPTURE_WORDS) return;
  const auto& point = control->points[point_id];
  // The callback runs before applying the application's predicate. Every lane
  // in reached participates in every collective, including all-false guards.
  const unsigned reached = __activemask();
  const unsigned executed = __ballot_sync(reached, pred != 0);
  const int lane = get_laneid();
  const int leader = __ffs(reached) - 1;
  const int4 cta = get_ctaid();
  const uint32_t warp = (threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) / 32;
  uint32_t words[PLAN_CAPTURE_WORDS];
  va_list arguments;
  va_start(arguments, word_count);
  for (uint32_t i = 0; i < word_count; ++i) words[i] = va_arg(arguments, unsigned int);
  va_end(arguments);

  for (uint32_t config_index = 0; config_index < control->config_count; ++config_index) {
    const auto bit = uint64_t{1} << (config_index % 64);
    const auto bank = config_index / 64;
    if (!(control->enabled[bank] & point.configurations[bank] & bit)) continue;
    const auto& config = control->configs[config_index];
    int geometry_selected = 0;
    if (lane == leader) {
      const uint64_t cta_key[] = {static_cast<uint32_t>(cta.x), static_cast<uint32_t>(cta.y),
                                  static_cast<uint32_t>(cta.z)};
      const uint64_t warp_key[] = {warp};
      geometry_selected = cutracer::plan_selection_contains(config.ctas, config.cta_count, cta_key) &&
                          cutracer::plan_selection_contains(config.warps, config.warp_count, warp_key);
    }
    // CTA and warp are uniform. Search once, then broadcast to every reached
    // lane before any lane selection; even unselected lanes join collectives.
    geometry_selected = __shfl_sync(reached, geometry_selected, leader);
    const unsigned selected = reached & config.lane_mask;
    if (!geometry_selected || !selected) continue;
    auto& counters = control->counters[config_index];
    unsigned long long sequence = 0, byte_offset = 0;
    if (lane == leader) {
      // Count actual entry visits even when their packets hit a budget.
      // Completion also requires zero drops and exact sent/received/written
      // counts; an entry count alone is never a delivery receipt.
      if (point.entry_marker[bank] & bit) atomicAdd(&counters.entries, 1ULL);
      sequence = atomicAdd(&counters.attempted, 1ULL);
      byte_offset = atomicAdd(&counters.attempted_bytes, static_cast<unsigned long long>(sizeof(plan_observation_t)));
    }
    sequence = __shfl_sync(reached, sequence, leader);
    byte_offset = __shfl_sync(reached, byte_offset, leader);
    if (sequence >= config.max_records || byte_offset > config.max_bytes ||
        sizeof(plan_observation_t) > config.max_bytes - byte_offset) {
      if (lane == leader) atomicAdd(&counters.dropped, 1ULL);
      continue;
    }
    const unsigned valid = (point.reached_policy[bank] & bit) ? selected : selected & executed;
    plan_observation_t packet{};
    packet.header.type = MSG_TYPE_PLAN_OBSERVATION;
    packet.version = 1;
    packet.kernel_launch_id = control->launch_id;
    packet.point_id = point_id;
    packet.config_index = config_index;
    packet.sequence = sequence;
    packet.cta[0] = cta.x;
    packet.cta[1] = cta.y;
    packet.cta[2] = cta.z;
    packet.warp_in_cta = warp;
    packet.reached_mask = reached;
    packet.executed_mask = executed;
    packet.selected_mask = selected;
    packet.valid_mask = valid;
    packet.word_count = word_count;
    packet.word_mask = point.word_masks[config_index];
    for (uint32_t word = 0; word < word_count; ++word) {
      if (!(packet.word_mask & (1U << word))) continue;
      for (int source = 0; source < 32; ++source) {
        if (!(valid & (1U << source))) continue;
        const auto value = __shfl_sync(reached, words[word], source);
        if (lane == leader) packet.values[word][source] = value;
      }
    }
    if (lane == leader) {
      reinterpret_cast<ChannelDev*>(channel_address)->push(&packet, sizeof(packet));
      atomicAdd(&counters.emitted, 1ULL);
      atomicAdd(&counters.emitted_bytes, static_cast<unsigned long long>(sizeof(packet)));
    }
  }
}
