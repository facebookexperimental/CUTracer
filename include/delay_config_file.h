/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <string>

#include "nlohmann/json.hpp"

namespace cutracer {

// Writes one process's delay config into CUTRACER_DELAY_DUMP_PATH, which every
// process of a run shares. Under an exclusive lock on `<path>.lock`, kernels
// already in the file are kept when its seed and delay match `config` (the same
// run), and `config`'s kernels replace entries with the same key. The file is
// replaced by a rename, so a reader never sees a partial file. A config without
// kernels never replaces an existing file. Returns an error message, or an
// empty string on success.
std::string merge_delay_config_file(const std::string& path, const nlohmann::json& config);

}  // namespace cutracer
