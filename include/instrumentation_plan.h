/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <array>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace cutracer::plan {

inline constexpr char kSchema[] = "cutracer.instrumentation_plan/v1";
inline constexpr uint64_t kExactIntegerMax = (uint64_t{1} << 53) - 1;
inline constexpr size_t kFileBytesMax = 16 * 1024 * 1024;

class PlanError : public std::invalid_argument {
 public:
  using std::invalid_argument::invalid_argument;
};

enum class Phase { Before, After };
enum class PredicatePolicy { Executed, Reached };
enum class RegisterBank { R, UR, P, UP };
enum class ObservationKind { Registers, MemoryAddress, MemoryValue, Opcode, Tma, Delay };

struct Register {
  RegisterBank bank;
  uint32_t index;
  std::string name() const;
};

struct Observation {
  ObservationKind kind;
  std::vector<Register> registers;
  uint64_t mref = 0;
  uint32_t delay_ns = 0;
};

struct Point {
  std::string function;
  uint64_t pc = 0;  // Function-relative byte offset, never an instruction ordinal.
  std::string expected_bytes;
  Phase phase = Phase::Before;
  PredicatePolicy predicate = PredicatePolicy::Executed;
  std::vector<Observation> observations;
};

struct LaunchSelector {
  bool all = true;
  std::vector<uint64_t> ordinals;
  bool selects(uint64_t ordinal) const;
};

// An empty vector denotes "all"; empty selection arrays are rejected.
struct CaptureSelector {
  std::vector<std::array<uint64_t, 3>> ctas;
  std::vector<uint64_t> warps;
  std::vector<uint64_t> lanes;
};

struct Budget {
  uint64_t max_records = 0;
  uint64_t max_bytes = 0;
};

struct Config {
  std::string id;
  std::vector<std::string> requests;
  LaunchSelector launches;
  CaptureSelector capture;
  Budget budget;
  std::vector<Point> points;
};

struct Kernel {
  std::string entry;
  std::string target;
  std::string cubin_sha256;
  std::vector<Config> configs;
};

struct Policy {
  bool report_unsupported_as_gap = false;
  bool skip_identity_mismatch = false;
};

// Parsing is transactional. Only a completely validated canonical plan can be
// published, and runtime consumers get a const view. Parsing does not certify
// that any observation is supported or applied by an instrumentation backend.
class CanonicalPlan {
 public:
  const std::string& id() const {
    return id_;
  }

  bool report_unsupported_as_gap() const {
    return policy_.report_unsupported_as_gap;
  }

  // A launch of a planned entry from another binary is left uninstrumented.
  bool skip_identity_mismatch() const {
    return policy_.skip_identity_mismatch;
  }

  const std::vector<Kernel>& kernels() const {
    return kernels_;
  }

  const nlohmann::json& canonical() const {
    return canonical_;
  }

  const std::string& canonical_sha256() const {
    return canonical_sha256_;
  }

  const std::string& source_sha256() const {
    return source_sha256_;
  }

 private:
  friend class PlanLoader;
  CanonicalPlan(std::string id, Policy flags, std::vector<Kernel> kernels, nlohmann::json canonical,
                std::string source_sha256);
  std::string id_;
  Policy policy_;
  std::vector<Kernel> kernels_;
  nlohmann::json canonical_;
  std::string canonical_sha256_;
  std::string source_sha256_;
};

class PlanLoader {
 public:
  static CanonicalPlan parse(const std::string& bytes);
  static CanonicalPlan load(const std::string& path);
};

nlohmann::json to_json(const Point& point);
nlohmann::json to_json(const Config& config);
nlohmann::json to_json(const Kernel& kernel);

}  // namespace cutracer::plan
