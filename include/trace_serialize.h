/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <string>

#include "trace_writer.h"

namespace cutracer {

/**
 * @brief Serialize one trace record to its NDJSON line via the rapidjson
 *        streaming writer (no trailing newline).
 *
 * Returns an empty string only for unknown types or null data — these fall
 * back to nlohmann in the live writer. This is a thin, engine-explicit seam over the same
 * serializer used by write_json_format(); it exists so unit tests (and ad-hoc
 * diagnostics) can exercise the rapidjson output directly, bypassing the
 * CUTRACER_JSON_ENGINE env dispatch and file IO.
 */
std::string serialize_record_rapidjson(const TraceRecord& record);

}  // namespace cutracer
