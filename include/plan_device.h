/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "common.h"

static_assert(sizeof(plan_observation_t) == 2120, "Plan packet ABI changed; version the producer and reader together");

inline constexpr uint32_t kPlanConfigsMax = 256;
inline constexpr uint32_t kPlanMaskWords = kPlanConfigsMax / 64;
inline constexpr uint32_t kPlanLaunchSlots = 1024;
inline constexpr uint32_t kPlanPointsMax = 8192;

struct PlanDeviceConfig {
  uint64_t max_records, max_bytes;
  uint32_t lane_mask;
  uint32_t cta_count, warp_count;
  const uint64_t* ctas;
  const uint64_t* warps;
};

struct PlanDevicePoint {
  uint64_t configurations[kPlanMaskWords];
  uint64_t reached_policy[kPlanMaskWords];
  uint64_t entry_marker[kPlanMaskWords];
  uint32_t word_masks[kPlanConfigsMax];
  uint32_t word_count;
};

struct PlanDeviceCounters {
  unsigned long long attempted;
  unsigned long long attempted_bytes;
  unsigned long long emitted;
  unsigned long long emitted_bytes;
  unsigned long long dropped;
  unsigned long long entries;
};

// One immutable selection per actual launch, with independent device counters.
// The host reads counters only after verified CUDA completion and channel drain.
struct PlanLaunchControl {
  uint64_t launch_id;
  uint64_t enabled[kPlanMaskWords];
  const PlanDeviceConfig* configs;
  const PlanDevicePoint* points;
  uint32_t config_count;
  uint32_t point_count;
  PlanDeviceCounters counters[kPlanConfigsMax];
};
