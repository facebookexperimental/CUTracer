/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <nlohmann/json.hpp>

#include "env_config.h"
#include "instrument_types.h"

inline nlohmann::json instrument_ipoints_to_json() {
  nlohmann::json ipoints = nlohmann::json::object();
  if (instrument_type_to_index.count(InstrumentType::REG_TRACE)) {
    const IPointType position = resolve_instrument_ipoint(InstrumentType::REG_TRACE, IPointType::BEFORE);
    ipoints["reg_trace"] = position == IPointType::BEFORE ? "BEFORE" : "AFTER";
  }
  return ipoints;
}
