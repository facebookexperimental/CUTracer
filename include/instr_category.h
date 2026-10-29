/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 * See LICENSE file in the root directory for Meta's license terms.
 */

#ifndef INSTR_CATEGORY_H
#define INSTR_CATEGORY_H

#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/**
 * @brief Instruction categories for classification and tracing.
 *
 * This enum defines logical categories of SASS instructions that can be
 * detected and traced. Each category groups related instructions together
 * for easier analysis and extension.
 *
 * To add a new category:
 * 1. Add a new enum value here
 * 2. Add pattern(s) to INSTR_CATEGORY_PATTERNS below
 * 3. The detection and logging will happen automatically
 */
enum class InstrCategory {
  NONE = 0,  // No category / unknown instruction

  // Matrix Multiply-Accumulate instructions
  MMA,

  // Tensor Memory Access instructions
  TMA,

  // Synchronization instructions
  SYNC,

  // Add new categories here as needed
  // Example: LDST, ALU, CONTROL, etc.
};

/**
 * @brief Get a human-readable name for an instruction category.
 */
inline const char* get_instr_category_name(InstrCategory cat) {
  switch (cat) {
    case InstrCategory::NONE:
      return "NONE";
    case InstrCategory::MMA:
      return "MMA";
    case InstrCategory::TMA:
      return "TMA";
    case InstrCategory::SYNC:
      return "SYNC";
    default:
      return "UNKNOWN";
  }
}

/**
 * @brief Pattern definition for instruction category matching.
 */
struct InstrCategoryPattern {
  const char* pattern;      // SASS substring to match
  InstrCategory category;   // Category this pattern belongs to
  const char* description;  // Human-readable description
};

/**
 * @brief Instruction category patterns.
 *
 * Each entry maps a SASS instruction pattern to a category.
 * Patterns are matched using substring search (strstr).
 *
 * To add support for new instructions:
 * - Add a new entry with the SASS pattern, category, and description
 * - The pattern should be specific enough to avoid false matches
 */
static const std::vector<InstrCategoryPattern> INSTR_CATEGORY_PATTERNS = {
    // MMA (Matrix Multiply-Accumulate) instructions
    // Hopper GMMA
    {"HGMMA", InstrCategory::MMA, "Hopper GMMA (sm_90)"},
    // Blackwell UTC*MMA variants (sm_100+)
    {"UTCHMMA", InstrCategory::MMA, "Blackwell UTCHMMA"},
    {"UTCIMMA", InstrCategory::MMA, "Blackwell UTCIMMA"},
    {"UTCQMMA", InstrCategory::MMA, "Blackwell UTCQMMA"},
    {"UTCOMMA", InstrCategory::MMA, "Blackwell UTCOMMA"},

    // TMA (Tensor Memory Access) instructions
    {"UTMALDG", InstrCategory::TMA, "Unified TMA Load Global"},
    {"UTMASTG", InstrCategory::TMA, "Unified TMA Store Global"},
    {"UTMAREDG", InstrCategory::TMA, "Unified TMA Store Global with Reduction"},

    // SYNC (Synchronization) instructions
    {"WARPGROUP.DEPBAR", InstrCategory::SYNC, "Warpgroup dependency barrier"},
    {"SYNCS.", InstrCategory::SYNC, "Memory barrier synchronization"},
    {"BAR.SYNC", InstrCategory::SYNC, "Named barrier synchronization"},
    {"BAR.ARV", InstrCategory::SYNC, "Named barrier arrival"},
    {"DEPBAR.", InstrCategory::SYNC, "Asynchronous dependency barrier"},
};

/**
 * @brief Detect the category of an instruction from its SASS string.
 *
 * @param sass The SASS instruction string
 * @return The detected category, or InstrCategory::NONE if no match
 */
inline InstrCategory detect_instr_category(const char* sass) {
  if (sass == nullptr) {
    return InstrCategory::NONE;
  }

  for (const auto& entry : INSTR_CATEGORY_PATTERNS) {
    if (strstr(sass, entry.pattern) != nullptr) {
      return entry.category;
    }
  }

  return InstrCategory::NONE;
}

/**
 * @brief Get the pattern description for a matched instruction.
 *
 * @param sass The SASS instruction string
 * @return The description of the matched pattern, or nullptr if no match
 */
inline const char* get_instr_pattern_description(const char* sass) {
  if (sass == nullptr) {
    return nullptr;
  }

  for (const auto& entry : INSTR_CATEGORY_PATTERNS) {
    if (strstr(sass, entry.pattern) != nullptr) {
      return entry.description;
    }
  }

  return nullptr;
}

/**
 * @brief Check if an instruction belongs to a specific category.
 *
 * @param sass The SASS instruction string
 * @param category The category to check for
 * @return true if the instruction belongs to the category
 */
inline bool is_instr_category(const char* sass, InstrCategory category) {
  return detect_instr_category(sass) == category;
}

/**
 * @brief Get all patterns for a specific category.
 *
 * @param category The category to get patterns for
 * @return Vector of pattern strings for the category
 */
inline std::vector<const char*> get_patterns_for_category(InstrCategory category) {
  std::vector<const char*> patterns;
  for (const auto& entry : INSTR_CATEGORY_PATTERNS) {
    if (entry.category == category) {
      patterns.push_back(entry.pattern);
    }
  }
  return patterns;
}

#endif /* INSTR_CATEGORY_H */
