/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <string>

// Empty on an empty/unreadable file or a hashing failure.
std::string cubin_file_sha256(const std::string& path);

// Dumped CUBINs are recorded by file name, relative to the trace output directory.
std::string cubin_artifact_path(const std::string& output_dir, const std::string& name);

// Rereads a dumped CUBIN and returns "sm_100a" when its bytes still have this
// SHA-256 and an ELF target that plan capture qualifies. Throws
// std::runtime_error naming the failed check otherwise.
std::string plan_cubin_target(const std::string& path, const std::string& sha256, int sm_family);
