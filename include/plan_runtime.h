/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */
#pragma once

#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>

#include "common.h"
#include "instrumentation_plan.h"
#include "nvbit.h"

class TraceWriter;

namespace cutracer {

class PlanRuntime {
 public:
  using Json = nlohmann::json;
  using Sink = std::function<void(const std::string&, Json)>;
  // Function identities name their dumped CUBINs relative to output_dir.
  PlanRuntime(plan::CanonicalPlan plan, std::string output_dir, Sink sink);
  ~PlanRuntime();

  bool disabled() const {
    return disabled_.load();
  }

  void disable(const std::string& reason);
  void initialize_context(CUcontext context);
  bool prepare(CUcontext context, CUfunction function, const Json& identity, uint64_t channel);
  void validate_graph_function(CUcontext context, CUfunction function);
  Json launch(CUcontext context, CUfunction function, const Json& identity);
  void device_complete(CUcontext context, uint64_t launch, bool completed, bool drained);
  void consume(CUcontext context, const plan_observation_t& record, TraceWriter& writer);
  void finish_trace(CUcontext context, uint64_t launch, TraceWriter& writer);
  void forget_module(CUcontext context, CUmodule module);
  void destroy_context(CUcontext context);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::atomic<bool> disabled_{false};
};

std::shared_ptr<PlanRuntime> current_runtime_plan();
void publish_runtime_plan(std::shared_ptr<PlanRuntime> plan);

}  // namespace cutracer
