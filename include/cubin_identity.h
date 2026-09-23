/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <string>

// Empty on an empty/unreadable file or a hashing failure.
std::string cubin_file_sha256(const std::string& path);
