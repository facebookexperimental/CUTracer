/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <rapidjson/stringbuffer.h>

#include "trace_writer.h"

namespace cutracer {

// Serialize `record` into `sb` via the rapidjson streaming writer (no trailing
// newline). Returns false only for an unknown type or null data (the caller
// skips such records). Defined in trace_rapidjson.cpp.
//
// Internal seam shared by the live writer (write_json_format) and the unit
// tests. NOT included by trace_writer.h, so rapidjson never reaches the nvcc
// compile of analysis.cu.
bool rj_serialize_to_buffer(const TraceRecord& record, rapidjson::StringBuffer& sb);

}  // namespace cutracer
