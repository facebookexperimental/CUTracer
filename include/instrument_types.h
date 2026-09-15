/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */

#pragma once

enum class InstrumentType {
  OPCODE_ONLY,      // Lightweight: only collect opcode information
  REG_TRACE,        // Medium: collect register values
  MEM_ADDR_TRACE,   // Heavy: collect memory access information (address only)
  MEM_VALUE_TRACE,  // Heavy: collect memory access with values
  TMA_TRACE,        // TMA descriptor tracing for Tensor Memory Accelerator instructions
  RANDOM_DELAY,     // Inject random delays on synchronization instructions
};
