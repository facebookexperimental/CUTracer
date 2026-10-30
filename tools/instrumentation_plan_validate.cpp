/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */
#include <exception>
#include <iostream>

#include "instrumentation_plan.h"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: instrumentation_plan_validate PLAN.json[.ndjson]\n";
    return 2;
  }
  try {
    const auto plan = cutracer::plan::PlanLoader::load(argv[1]);
    std::cout << nlohmann::json{{"schema", "cutracer.instrumentation-plan-validation/v1"},
                                {"source_sha256", plan.source_sha256()},
                                {"canonical_sha256", plan.canonical_sha256()},
                                {"canonical", plan.canonical()},
                                {"runtime_applied", false}}
                     .dump(-1, ' ', true)
              << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
