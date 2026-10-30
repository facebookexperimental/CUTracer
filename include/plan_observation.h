/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include "common.h"

#define PLAN_CAPTURE_WORDS 16

// Packet version 1. Masks refer to the same warp invocation at this point.
// valid_mask may be zero when a reached point has no executed selected lanes.
struct plan_observation_t {
  message_header_t header;
  uint32_t version;
  uint64_t kernel_launch_id;
  uint32_t point_id;
  uint32_t config_index;
  uint64_t sequence;
  uint32_t cta[3];
  uint32_t warp_in_cta;
  uint32_t reached_mask;
  uint32_t executed_mask;
  uint32_t selected_mask;
  uint32_t valid_mask;
  uint32_t word_count;
  uint32_t word_mask;
  uint32_t values[PLAN_CAPTURE_WORDS][32];
};
