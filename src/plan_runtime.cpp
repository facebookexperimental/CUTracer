/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */
#include "plan_runtime.h"

#include <cuda.h>

#include <algorithm>
#include <array>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "cubin_identity.h"
#include "plan_device.h"
#include "runtime_identity.h"
#include "trace_writer.h"

namespace cutracer {
namespace {
// Never destroyed: NVBit calls nvbit_at_term from a static destructor, after
// the static objects of this translation unit may already be gone.
std::shared_ptr<PlanRuntime>& published_plan() {
  static auto* plan = new std::shared_ptr<PlanRuntime>();
  return *plan;
}
}  // namespace

std::shared_ptr<PlanRuntime> current_runtime_plan() {
  return std::atomic_load(&published_plan());
}

void publish_runtime_plan(std::shared_ptr<PlanRuntime> plan) {
  std::atomic_store(&published_plan(), std::move(plan));
}

namespace {
using Json = nlohmann::json;
using PointKey = std::tuple<std::string, std::string, uint64_t, plan::Phase>;
using Mask = std::array<uint64_t, kPlanMaskWords>;

void enable(uint64_t* mask, uint32_t index) {
  mask[index / 64] |= uint64_t{1} << (index % 64);
}

bool enabled(const uint64_t* mask, uint32_t index) {
  return (mask[index / 64] >> (index % 64)) & 1;
}

std::string hex(uint64_t value) {
  std::ostringstream out;
  out << "0x" << std::hex << value;
  return out.str();
}

std::string instruction_bytes(Instr* instruction) {
  std::string result;
  instruction->getSassBinary(
      [](uint8_t byte, void* output) {
        static constexpr char digits[] = "0123456789abcdef";
        auto& bytes = *static_cast<std::string*>(output);
        bytes.push_back(digits[byte >> 4]);
        bytes.push_back(digits[byte & 15]);
      },
      &result);
  return result;
}

// Lanes that leave through a control transfer never reach an after-instruction
// callback. Keep in sync with `_RUNTIME_CONTROL` in observe/resolve.py.
bool transfers_control(const std::string& opcode) {
  static const std::set<std::string> opcodes = {"BPT", "BRA", "BRX",  "BRXU", "CALL", "EXIT",
                                                "JMP", "JMX", "JMXU", "KILL", "RET",  "RTT"};
  return opcodes.count(opcode) != 0;
}

void cuda_ok(CUresult status) {
  if (status == CUDA_SUCCESS) {
    return;
  }
  const char* message = nullptr;
  cuGetErrorString(status, &message);
  throw plan::PlanError(std::string("Plan device allocation: ") + (message ? message : "CUDA driver failure"));
}

uint64_t checked_product(uint64_t a, uint64_t b) {
  if (b && a > std::numeric_limits<uint64_t>::max() / b) {
    throw plan::PlanError("Capture geometry exceeds count range");
  }
  return a * b;
}

uint64_t expected_entries(const plan::Config& config, const Json& identity) {
  const auto grid = identity.at("grid").get<std::array<uint64_t, 3>>();
  const auto block = identity.at("block").get<std::array<uint64_t, 3>>();
  const auto threads = checked_product(checked_product(block[0], block[1]), block[2]);
  if (threads > 1024) {
    throw plan::PlanError("Invalid block geometry for plan capture");
  }
  uint64_t ctas = checked_product(checked_product(grid[0], grid[1]), grid[2]);
  if (!config.capture.ctas.empty()) {
    ctas = 0;
    for (const auto& cta : config.capture.ctas) {
      if (cta[0] < grid[0] && cta[1] < grid[1] && cta[2] < grid[2]) {
        ++ctas;
      }
    }
  }
  uint64_t warps = 0;
  for (uint64_t warp = 0; warp < (threads + 31) / 32; ++warp) {
    if (!config.capture.warps.empty() &&
        !std::binary_search(config.capture.warps.begin(), config.capture.warps.end(), warp)) {
      continue;
    }
    const auto lanes = std::min<uint64_t>(32, threads - warp * 32);
    if (config.capture.lanes.empty() || config.capture.lanes.front() < lanes) {
      ++warps;
    }
  }
  return checked_product(ctas, warps);
}
}  // namespace

struct PlanRuntime::Impl {
  struct Config {
    size_t kernel;
    const plan::Config* source;
  };

  struct Point {
    PointKey key;
    std::string expected;
    std::map<uint32_t, const plan::Point*> requests;
    std::set<uint32_t> entries;
    std::vector<plan::Register> words;
    PlanDevicePoint device{};
  };

  struct Function {
    CUfunction handle;
    CUmodule module;
    int registers;
    std::map<uint64_t, Instr*> instructions;
  };

  struct Bound {
    size_t kernel;
    CUmodule module;
    std::vector<uint32_t> configs;
    std::map<uint32_t, std::string> gaps;
    std::map<uint32_t, Json> points;
  };

  struct Counts {
    uint64_t expected_entries = 0, received = 0, written = 0, json_bytes = 0;
    bool selected = false;
    std::string error;
    PlanDeviceCounters device{};
  };

  struct Launch {
    Json identity, applied;
    size_t slot = kPlanLaunchSlots;
    bool completed = false, drained = false, counters_available = false;
    std::map<uint32_t, Counts> counts;
    std::set<std::pair<uint32_t, uint64_t>> records;
  };

  struct Context {
    std::vector<void*> allocations;
    PlanDeviceConfig* configs = nullptr;
    PlanDevicePoint* points = nullptr;
    PlanLaunchControl* slots = nullptr;
    std::array<bool, kPlanLaunchSlots> occupied{};
    std::map<CUfunction, Bound> bound;
    // Functions of a planned entry name loaded from a binary the plan does not name.
    std::map<CUfunction, CUmodule> skipped;
    std::map<CUfunction, CUmodule> owners;
    std::set<std::pair<CUfunction, uint32_t>> injected;
    std::map<uint64_t, Launch> launches;

    template <typename T>
    T* allocate(size_t count) {
      static_assert(std::is_trivially_copyable_v<T>, "Plan device tables are plain values");
      T* result = nullptr;
      cuda_ok(cuMemAllocManaged(reinterpret_cast<CUdeviceptr*>(&result), count * sizeof(T), CU_MEM_ATTACH_GLOBAL));
      if (!result) {
        throw plan::PlanError("Plan device allocation returned no memory");
      }
      allocations.push_back(result);
      std::fill_n(result, count, T{});
      return result;
    }
  };

  plan::CanonicalPlan plan;
  std::string output_dir;
  Sink sink;
  std::mutex mutex;
  std::vector<Config> configs;
  std::vector<Point> points;
  std::map<PointKey, uint32_t> point_ids;
  std::map<CUcontext, std::unique_ptr<Context>> contexts;

  Impl(plan::CanonicalPlan canonical, std::string artifacts, Sink events)
      : plan(std::move(canonical)), output_dir(std::move(artifacts)), sink(std::move(events)) {
    for (size_t kernel = 0; kernel < plan.kernels().size(); ++kernel) {
      const auto& k = plan.kernels()[kernel];
      for (const auto& config : k.configs) {
        if (configs.size() >= kPlanConfigsMax) {
          throw plan::PlanError("Plan backend configuration budget exceeded (256)");
        }
        const auto id = static_cast<uint32_t>(configs.size());
        configs.push_back({kernel, &config});
        for (const auto& requested : config.points) {
          auto& point = add_point({k.cubin_sha256, requested.function, requested.pc, requested.phase});
          point.expected = requested.expected_bytes;
          point.requests.emplace(id, &requested);
        }
        add_point({k.cubin_sha256, k.entry, 0, plan::Phase::Before}).entries.insert(id);
      }
    }
    for (auto& point : points) {
      std::map<std::string, plan::Register> words;
      for (const auto& [index, request] : point.requests) {
        for (const auto& observation : request->observations) {
          for (const auto& reg : observation.registers) {
            words.emplace(reg.name(), reg);
          }
        }
      }
      for (const auto& [name, reg] : words) {
        point.words.push_back(reg);
      }
      point.device.word_count = static_cast<uint32_t>(point.words.size());
      for (const auto& [index, request] : point.requests) {
        enable(point.device.configurations, index);
        if (request->predicate == plan::PredicatePolicy::Reached) {
          enable(point.device.reached_policy, index);
        }
        for (const auto& observation : request->observations) {
          for (const auto& reg : observation.registers) {
            const auto pos = std::find_if(point.words.begin(), point.words.end(),
                                          [&](const auto& word) { return word.name() == reg.name(); });
            const auto offset = static_cast<uint32_t>(pos - point.words.begin());
            if (offset < PLAN_CAPTURE_WORDS) {
              point.device.word_masks[index] |= 1U << offset;
            }
          }
        }
      }
      for (const auto index : point.entries) {
        enable(point.device.configurations, index);
        enable(point.device.entry_marker, index);
      }
    }
    sink("plan_loaded", {{"plan_id", plan.id()},
                         {"canonical_sha256", plan.canonical_sha256()},
                         {"file_sha256", plan.source_sha256()},
                         {"canonical", plan.canonical()},
                         {"backend", "cutracer.plan-runtime/1"},
                         {"runtime_applied", false},
                         {"packet_bytes", sizeof(plan_observation_t)},
                         {"max_live_launches", kPlanLaunchSlots}});
  }

  Point& add_point(PointKey key) {
    const auto found = point_ids.find(key);
    if (found != point_ids.end()) {
      return points.at(found->second);
    }
    if (points.size() >= kPlanPointsMax) {
      throw plan::PlanError("Plan backend point budget exceeded (8192)");
    }
    const auto id = static_cast<uint32_t>(points.size());
    point_ids.emplace(key, id);
    Point point;
    point.key = std::move(key);
    points.push_back(std::move(point));
    return points.back();
  }

  Context& context(CUcontext ctx) {
    const auto found = contexts.find(ctx);
    if (found == contexts.end()) {
      throw plan::PlanError("Plan context was not initialized by tool-init");
    }
    return *found->second;
  }

  std::map<std::string, Function> inventory(CUcontext ctx, CUfunction root, CUmodule module) {
    auto related = nvbit_get_related_functions(ctx, root);
    related.push_back(root);
    std::map<std::string, Function> result;
    for (auto function : related) {
      CUmodule owner = nullptr;
      if (cuFuncGetModule(&owner, function) != CUDA_SUCCESS || owner != module) {
        throw plan::PlanError("Related function is not bound to the same loaded module");
      }
      const std::string name = nvbit_get_func_name(ctx, function, true);
      if (result.count(name)) {
        if (result.at(name).handle != function) {
          throw plan::PlanError("Ambiguous machine function name");
        }
        continue;
      }
      int registers = 0;
      if (cuFuncGetAttribute(&registers, CU_FUNC_ATTRIBUTE_NUM_REGS, function) != CUDA_SUCCESS) {
        throw plan::PlanError("Actual register allocation unavailable");
      }
      Function entry{function, owner, registers, {}};
      for (auto* instruction : nvbit_get_instrs(ctx, function)) {
        if (!entry.instructions.emplace(instruction->getOffset(), instruction).second) {
          throw plan::PlanError("Duplicate byte PC");
        }
      }
      result.emplace(name, std::move(entry));
    }
    return result;
  }

  std::string words_capability(const Point& point, const Function& function) {
    if (point.words.size() > PLAN_CAPTURE_WORDS) {
      return "register_union_exceeds_16_words";
    }
    for (const auto& reg : point.words) {
      if ((reg.bank == plan::RegisterBank::R &&
           (reg.index > 254 || reg.index >= static_cast<uint32_t>(function.registers))) ||
          (reg.bank == plan::RegisterBank::UR && reg.index > 62) ||
          ((reg.bank == plan::RegisterBank::P || reg.bank == plan::RegisterBank::UP) && reg.index > 6)) {
        return "register_outside_actual_allocation_or_bank";
      }
    }
    return {};
  }

  std::string capability(const Point& point, const plan::Point& request, const Function& function, Instr* instruction) {
    auto word_gap = words_capability(point, function);
    if (!word_gap.empty()) {
      return word_gap;
    }
    if (request.phase == plan::Phase::After &&
        (instruction->hasPred() || request.predicate == plan::PredicatePolicy::Reached ||
         transfers_control(instruction->getOpcodeShort()))) {
      return "after_control_or_predicated_instruction_unqualified";
    }
    for (const auto& observation : request.observations) {
      if (observation.kind != plan::ObservationKind::Registers && observation.kind != plan::ObservationKind::Opcode) {
        return "observation_kind_not_implemented_by_value_producer";
      }
    }
    return {};
  }

  Json point_metadata(uint32_t id, uint32_t config, const std::string& bytes) {
    const auto& point = points.at(id);
    Json names = Json::array();
    for (size_t i = 0; i < point.words.size() && i < PLAN_CAPTURE_WORDS; ++i) {
      if (point.device.word_masks[config] & (1U << i)) {
        names.push_back(point.words[i].name());
      }
    }
    return {{"point_id", id},
            {"function", std::get<1>(point.key)},
            {"pc", hex(std::get<2>(point.key))},
            {"expected_bytes", bytes},
            {"phase", std::get<3>(point.key) == plan::Phase::Before ? "before" : "after"},
            {"predicate_policy", enabled(point.device.reached_policy, config) ? "reached_lanes" : "executed_lanes"},
            {"registers", names},
            {"entry_marker", point.entries.count(config) != 0}};
  }
};

PlanRuntime::PlanRuntime(plan::CanonicalPlan plan, std::string output_dir, Sink sink)
    : impl_(std::make_unique<Impl>(std::move(plan), std::move(output_dir), std::move(sink))) {
}

PlanRuntime::~PlanRuntime() = default;

void PlanRuntime::disable(const std::string& reason) {
  disabled_.store(true);
  std::lock_guard<std::mutex> lock(impl_->mutex);
  // Retain live launch controls and device allocations. Already patched or
  // running kernels may still reference them until CUDA context teardown.
  for (auto& [ctx, state] : impl_->contexts) {
    for (auto& [id, launch] : state->launches) {
      for (auto& [index, counts] : launch.counts) {
        if (counts.error.empty()) {
          counts.error = reason;
        }
      }
    }
  }
}

void PlanRuntime::initialize_context(CUcontext ctx) {
  if (disabled()) {
    return;
  }
  auto& impl = *impl_;
  std::lock_guard<std::mutex> lock(impl.mutex);
  if (disabled() || impl.contexts.count(ctx)) {
    return;
  }
  auto context = std::make_unique<Impl::Context>();
  auto& state = *context;
  state.configs = state.allocate<PlanDeviceConfig>(impl.configs.size());
  state.points = state.allocate<PlanDevicePoint>(impl.points.size());
  state.slots = state.allocate<PlanLaunchControl>(kPlanLaunchSlots);
  for (size_t index = 0; index < impl.configs.size(); ++index) {
    const auto& config = *impl.configs[index].source;
    auto& device = state.configs[index];
    device.max_records = config.budget.max_records;
    device.max_bytes = config.budget.max_bytes;
    device.lane_mask = config.capture.lanes.empty() ? UINT32_MAX : 0;
    for (const auto lane : config.capture.lanes) {
      device.lane_mask |= 1U << lane;
    }
    device.cta_count = static_cast<uint32_t>(config.capture.ctas.size());
    device.warp_count = static_cast<uint32_t>(config.capture.warps.size());
    if (device.cta_count) {
      auto* ctas = state.allocate<uint64_t>(config.capture.ctas.size() * 3);
      for (size_t i = 0; i < config.capture.ctas.size(); ++i) {
        std::copy(config.capture.ctas[i].begin(), config.capture.ctas[i].end(), ctas + i * 3);
      }
      device.ctas = ctas;
    }
    if (device.warp_count) {
      auto* warps = state.allocate<uint64_t>(config.capture.warps.size());
      std::copy(config.capture.warps.begin(), config.capture.warps.end(), warps);
      device.warps = warps;
    }
  }
  for (size_t index = 0; index < impl.points.size(); ++index) {
    state.points[index] = impl.points[index].device;
  }
  impl.contexts.emplace(ctx, std::move(context));
}

bool PlanRuntime::prepare(CUcontext ctx, CUfunction root, const Json& identity, uint64_t channel) {
  if (disabled()) {
    return false;
  }
  auto& impl = *impl_;
  std::lock_guard<std::mutex> lock(impl.mutex);
  if (disabled()) {
    return false;
  }
  auto& state = impl.context(ctx);
  if (state.bound.count(root)) {
    return true;
  }
  if (state.skipped.count(root)) {
    return false;
  }
  const std::string name = nvbit_get_func_name(ctx, root, true);
  std::vector<size_t> candidates;
  for (size_t i = 0; i < impl.plan.kernels().size(); ++i) {
    if (impl.plan.kernels()[i].entry == name) {
      candidates.push_back(i);
    }
  }
  if (candidates.empty()) {
    return false;
  }
  if (identity.is_null() || identity.value("entry", "") != name) {
    throw plan::PlanError("Requested kernel lacks actual runtime identity");
  }
  CUmodule module = nullptr;
  if (cuFuncGetModule(&module, root) != CUDA_SUCCESS) {
    throw plan::PlanError("Module identity unavailable at plan binding");
  }
  const auto sha256 = identity.at("cubin_sha256").get<std::string>();
  candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                                  [&](size_t i) { return impl.plan.kernels()[i].cubin_sha256 != sha256; }),
                   candidates.end());
  if (candidates.empty()) {
    if (!impl.plan.skip_identity_mismatch()) {
      throw plan::PlanError("Requested kernel binary or target identity mismatch");
    }
    state.skipped.emplace(root, module);
    return false;
  }
  const auto path = cubin_artifact_path(impl.output_dir, identity.at("cubin_path").get<std::string>());
  const auto target = plan_cubin_target(path, sha256, identity.at("sm_family").get<int>());
  auto selected = candidates.end();
  for (auto it = candidates.begin(); it != candidates.end(); ++it) {
    if (impl.plan.kernels()[*it].target == target) {
      if (selected != candidates.end()) {
        throw plan::PlanError("Ambiguous matching kernel plan");
      }
      selected = it;
    }
  }
  if (selected == candidates.end()) {
    throw plan::PlanError("Requested kernel binary or target identity mismatch");
  }
  const auto functions = impl.inventory(ctx, root, module);
  Impl::Bound bound{*selected, module, {}, {}, {}};
  for (uint32_t i = 0; i < impl.configs.size(); ++i) {
    if (impl.configs[i].kernel == *selected) {
      bound.configs.push_back(i);
    }
  }
  std::map<uint32_t, std::pair<CUfunction, Instr*>> injection;
  // Validate the full requested configuration set before inserting any calls.
  for (uint32_t id = 0; id < impl.points.size(); ++id) {
    const auto& point = impl.points[id];
    if (std::get<0>(point.key) != sha256) {
      continue;
    }
    const bool needed = std::any_of(bound.configs.begin(), bound.configs.end(),
                                    [&](uint32_t index) { return enabled(point.device.configurations, index); });
    if (!needed) {
      continue;
    }
    const auto fit = functions.find(std::get<1>(point.key));
    if (fit == functions.end()) {
      throw plan::PlanError("Requested function is not related to the loaded entry");
    }
    const auto iit = fit->second.instructions.find(std::get<2>(point.key));
    if (iit == fit->second.instructions.end()) {
      throw plan::PlanError("Requested byte PC is not an instruction boundary");
    }
    const auto bytes = instruction_bytes(iit->second);
    if (bytes.size() != 32 || (!point.expected.empty() && bytes != point.expected)) {
      throw plan::PlanError("Requested instruction bytes mismatch");
    }
    for (const auto index : bound.configs) {
      const auto request = point.requests.find(index);
      if (point.entries.count(index)) {
        const auto gap = impl.words_capability(point, fit->second);
        if (!gap.empty()) {
          if (!impl.plan.report_unsupported_as_gap()) {
            throw plan::PlanError(gap);
          }
          bound.gaps[index] = gap;
        }
      }
      if (request != point.requests.end()) {
        const auto gap = impl.capability(point, *request->second, fit->second, iit->second);
        if (!gap.empty()) {
          if (!impl.plan.report_unsupported_as_gap()) {
            throw plan::PlanError(gap);
          }
          bound.gaps[index] = gap;
        }
      }
    }
    bound.points[id] = {{"bytes", bytes}};
    injection.emplace(id, std::make_pair(fit->second.handle, iit->second));
  }
  for (const auto& [id, item] : injection) {
    auto& point = impl.points[id];
    const bool needed = std::any_of(bound.configs.begin(), bound.configs.end(), [&](uint32_t config) {
      return !bound.gaps.count(config) && enabled(point.device.configurations, config);
    });
    if (!needed) {
      continue;
    }
    if (point.words.size() > PLAN_CAPTURE_WORDS) {
      throw plan::PlanError("Unqualified register union at supplemental marker");
    }
    if (!state.injected.emplace(item.first, id).second) {
      continue;
    }
    state.owners[item.first] = module;
    auto* instruction = item.second;
    nvbit_insert_call(instruction, "instrument_plan_values",
                      std::get<3>(point.key) == plan::Phase::Before ? IPOINT_BEFORE : IPOINT_AFTER);
    nvbit_add_call_arg_guard_pred_val(instruction);
    nvbit_add_call_arg_launch_val64(instruction, 0);
    nvbit_add_call_arg_const_val64(instruction, channel);
    nvbit_add_call_arg_const_val32(instruction, id);
    nvbit_add_call_arg_const_val32(instruction, static_cast<uint32_t>(point.words.size()));
    for (const auto& reg : point.words) {
      switch (reg.bank) {
        case plan::RegisterBank::R:
          nvbit_add_call_arg_reg_val(instruction, static_cast<int>(reg.index), true);
          break;
        case plan::RegisterBank::UR:
          nvbit_add_call_arg_ureg_val(instruction, static_cast<int>(reg.index), true);
          break;
        case plan::RegisterBank::P:
          nvbit_add_call_arg_pred_val_at(instruction, static_cast<int>(reg.index), true);
          break;
        case plan::RegisterBank::UP:
          nvbit_add_call_arg_upred_val_at(instruction, static_cast<int>(reg.index), true);
          break;
      }
    }
  }
  state.bound.emplace(root, std::move(bound));
  return true;
}

void PlanRuntime::validate_graph_function(CUcontext ctx, CUfunction root) {
  if (disabled()) {
    return;
  }
  auto& impl = *impl_;
  std::lock_guard<std::mutex> lock(impl.mutex);
  if (disabled()) {
    return;
  }
  const auto& state = impl.context(ctx);
  if (state.bound.count(root) || state.skipped.count(root)) {
    return;
  }
  const std::string name = nvbit_get_func_name(ctx, root, true);
  for (const auto& kernel : impl.plan.kernels()) {
    if (kernel.entry == name) {
      throw plan::PlanError("Requested graph function was not prepared by a supported CUDA launch or graph-build API");
    }
  }
}

PlanRuntime::Json PlanRuntime::launch(CUcontext ctx, CUfunction root, const Json& identity) {
  if (disabled()) {
    return nullptr;
  }
  auto& impl = *impl_;
  std::lock_guard<std::mutex> lock(impl.mutex);
  if (disabled()) {
    return nullptr;
  }
  auto& state = impl.context(ctx);
  const auto fit = state.bound.find(root);
  if (fit == state.bound.end()) {
    throw plan::PlanError("Selected kernel lacks a prepared plan");
  }
  if (state.launches.size() >= kPlanLaunchSlots) {
    throw plan::PlanError("Unfinalized plan launch budget exceeded");
  }
  const auto& bound = fit->second;
  const uint64_t id = identity.at("launch_id");
  const uint64_t ordinal = identity.at("kernel_ordinal");
  Impl::Launch launch;
  launch.identity = identity;
  Json configurations = Json::array();
  Mask selected{};
  for (const auto index : bound.configs) {
    const auto& config = *impl.configs[index].source;
    const auto gap = bound.gaps.find(index);
    auto& counts = launch.counts[index];
    counts.selected = gap == bound.gaps.end() && config.launches.selects(ordinal);
    if (gap != bound.gaps.end()) {
      counts.error = gap->second;
    } else if (counts.selected) {
      counts.expected_entries = expected_entries(config, identity);
      if (counts.expected_entries == 0) {
        counts.error = "no_selected_warps_in_launch";
      }
      enable(selected.data(), index);
    }
    Json points = Json::array();
    for (const auto& [pid, metadata] : bound.points) {
      if (enabled(impl.points[pid].device.configurations, index)) {
        points.push_back(impl.point_metadata(pid, index, metadata.at("bytes")));
      }
    }
    configurations.push_back({{"config_index", index},
                              {"config_id", config.id},
                              {"requests", config.requests},
                              {"selected", counts.selected},
                              {"gap", gap == bound.gaps.end() ? std::string() : gap->second},
                              {"expected_entry_warps", counts.expected_entries},
                              {"configuration", plan::to_json(config)},
                              {"actual_points", points}});
  }
  const bool any = std::any_of(selected.begin(), selected.end(), [](auto bits) { return bits != 0; });
  PlanLaunchControl* control = nullptr;
  if (any) {
    const auto available = std::find(state.occupied.begin(), state.occupied.end(), false);
    if (available == state.occupied.end()) {
      throw plan::PlanError("Plan launch slots exhausted before CUDA completion");
    }
    launch.slot = available - state.occupied.begin();
    *available = true;
    control = &state.slots[launch.slot];
    *control = PlanLaunchControl{};
    control->launch_id = id;
    std::copy(selected.begin(), selected.end(), control->enabled);
    control->configs = state.configs;
    control->points = state.points;
    control->config_count = static_cast<uint32_t>(impl.configs.size());
    control->point_count = static_cast<uint32_t>(impl.points.size());
  }
  launch.applied = {{"type", "plan_applied"},
                    {"schema", "cutracer.plan-applied/v1"},
                    {"canonical_sha256", impl.plan.canonical_sha256()},
                    {"file_sha256", impl.plan.source_sha256()},
                    {"launch", identity.at("launch")},
                    {"launch_id", id},
                    {"runtime_identity", identity},
                    {"backend", "cutracer.plan-runtime/1"},
                    {"packet_version", 1},
                    {"packet_bytes", sizeof(plan_observation_t)},
                    {"budget_accounting", "device_packet_and_json_record_bytes"},
                    {"configs", configurations}};
  const auto applied = launch.applied;
  if (!state.launches.emplace(id, std::move(launch)).second) {
    throw plan::PlanError("Duplicate actual plan launch ID");
  }
  // NOLINTNEXTLINE(performance-no-int-to-ptr): the launch record keeps the actual stream handle.
  const auto stream = reinterpret_cast<CUstream>(identity.at("stream_handle").get<uint64_t>());
  nvbit_set_at_launch(ctx, root, reinterpret_cast<uint64_t>(control), stream,
                      identity.at("graph_node_handle").get<uint64_t>());
  impl.sink("plan_applied", applied);
  return applied;
}

void PlanRuntime::device_complete(CUcontext ctx, uint64_t id, bool completed, bool drained) {
  auto& impl = *impl_;
  std::lock_guard<std::mutex> lock(impl.mutex);
  // A context created after the plan was disabled has no plan state or launches.
  const auto found = impl.contexts.find(ctx);
  if (found == impl.contexts.end()) {
    return;
  }
  auto& state = *found->second;
  const auto it = state.launches.find(id);
  if (it == state.launches.end()) {
    return;
  }
  auto& launch = it->second;
  launch.completed = completed;
  launch.drained = drained;
  if (completed && drained) {
    launch.counters_available = true;
    if (launch.slot < kPlanLaunchSlots) {
      for (auto& [index, counts] : launch.counts) {
        counts.device = state.slots[launch.slot].counters[index];
      }
      state.occupied[launch.slot] = false;
      launch.slot = kPlanLaunchSlots;
    }
  }
  // Failed/uncertain launches retain their slots until context teardown.
}

void PlanRuntime::consume(CUcontext ctx, const plan_observation_t& packet, TraceWriter& writer) {
  if (disabled()) {
    writer.mark_capture_error("plan_disabled_after_error", true);
    return;
  }
  auto& impl = *impl_;
  std::lock_guard<std::mutex> lock(impl.mutex);
  if (disabled()) {
    writer.mark_capture_error("plan_disabled_after_error", true);
    return;
  }
  auto& state = impl.context(ctx);
  auto it = state.launches.find(packet.kernel_launch_id);
  if (it == state.launches.end() || packet.version != 1 || packet.point_id >= impl.points.size() ||
      packet.config_index >= impl.configs.size()) {
    writer.mark_capture_error("plan_packet_identity", true);
    return;
  }
  auto& launch = it->second;
  auto cit = launch.counts.find(packet.config_index);
  const auto& point = impl.points[packet.point_id];
  const auto& config = *impl.configs[packet.config_index].source;
  const auto grid = launch.identity.at("grid").get<std::array<uint64_t, 3>>();
  const auto block = launch.identity.at("block").get<std::array<uint64_t, 3>>();
  const uint64_t threads = checked_product(checked_product(block[0], block[1]), block[2]);
  const std::array<uint64_t, 3> cta{packet.cta[0], packet.cta[1], packet.cta[2]};
  const bool geometry =
      cta[0] < grid[0] && cta[1] < grid[1] && cta[2] < grid[2] &&
      (config.capture.ctas.empty() ||
       std::binary_search(config.capture.ctas.begin(), config.capture.ctas.end(), cta)) &&
      packet.warp_in_cta < (threads + 31) / 32 &&
      (config.capture.warps.empty() ||
       std::binary_search(config.capture.warps.begin(), config.capture.warps.end(), packet.warp_in_cta));
  const uint64_t active_lanes = geometry ? std::min<uint64_t>(32, threads - uint64_t{packet.warp_in_cta} * 32) : 0;
  const uint32_t block_mask = active_lanes == 32 ? UINT32_MAX : (uint32_t{1} << active_lanes) - 1;
  uint32_t lane_mask = config.capture.lanes.empty() ? UINT32_MAX : 0;
  for (const auto lane : config.capture.lanes) {
    lane_mask |= 1U << lane;
  }
  if (cit == launch.counts.end() || !cit->second.selected || !geometry || (packet.reached_mask & ~block_mask) ||
      packet.selected_mask != (packet.reached_mask & lane_mask) || packet.sequence >= config.budget.max_records ||
      !enabled(point.device.configurations, packet.config_index) || packet.word_count != point.words.size() ||
      packet.word_count > PLAN_CAPTURE_WORDS || packet.word_mask != point.device.word_masks[packet.config_index] ||
      !packet.reached_mask || (packet.executed_mask & ~packet.reached_mask) || !packet.selected_mask ||
      (packet.selected_mask & ~packet.reached_mask) ||
      packet.valid_mask != (enabled(point.device.reached_policy, packet.config_index)
                                ? packet.selected_mask
                                : packet.selected_mask & packet.executed_mask)) {
    writer.mark_capture_error("plan_packet_masks_or_binding", true);
    return;
  }
  auto& counts = cit->second;
  ++counts.received;
  if (launch.records.size() >= 1000000 || !launch.records.emplace(packet.config_index, packet.sequence).second) {
    counts.error = "record_identity_duplicate_or_budget";
    writer.mark_capture_error("plan_record_sequence", true);
    return;
  }
  Json registers = Json::object();
  for (uint32_t word = 0; word < packet.word_count; ++word) {
    if (!(packet.word_mask & (1U << word))) {
      continue;
    }
    Json lanes = Json::array();
    for (uint32_t lane = 0; lane < 32; ++lane) {
      if ((packet.valid_mask & (1U << lane)) && packet.values[word][lane] > 1 &&
          (point.words[word].bank == plan::RegisterBank::P || point.words[word].bank == plan::RegisterBank::UP)) {
        counts.error = "predicate_word_out_of_range";
        writer.mark_capture_error(counts.error, true);
        return;
      }
      lanes.push_back((packet.valid_mask & (1U << lane)) ? Json(hex(packet.values[word][lane])) : Json(nullptr));
    }
    registers[point.words[word].name()] = std::move(lanes);
  }
  Json record{{"type", "plan_observation"},
              {"version", 1},
              {"canonical_sha256", impl.plan.canonical_sha256()},
              {"launch", launch.identity.at("launch")},
              {"grid_launch_id", packet.kernel_launch_id},
              {"config_index", packet.config_index},
              {"config_id", impl.configs[packet.config_index].source->id},
              {"point_id", packet.point_id},
              {"sequence", packet.sequence},
              {"cta", {packet.cta[0], packet.cta[1], packet.cta[2]}},
              {"warp", packet.warp_in_cta},
              {"reached_mask", hex(packet.reached_mask)},
              {"executed_mask", hex(packet.executed_mask)},
              {"selected_mask", hex(packet.selected_mask)},
              {"valid_mask", hex(packet.valid_mask)},
              {"registers", registers},
              {"wire_bytes", sizeof(packet)}};
  const auto encoded = record.dump();
  const auto size = encoded.size() + 1;
  const auto& budget = impl.configs[packet.config_index].source->budget;
  if (counts.written >= budget.max_records || size > budget.max_bytes - std::min(counts.json_bytes, budget.max_bytes)) {
    counts.error = "json_record_budget_exhausted";
    writer.mark_capture_error("plan_json_budget", true);
    return;
  }
  if (!writer.write_encoded_record(encoded)) {
    counts.error = "trace_writer_rejected_record";
    return;
  }
  ++counts.written;
  counts.json_bytes += size;
}

void PlanRuntime::finish_trace(CUcontext ctx, uint64_t id, TraceWriter& writer) {
  auto& impl = *impl_;
  std::lock_guard<std::mutex> lock(impl.mutex);
  const auto found = impl.contexts.find(ctx);
  if (found == impl.contexts.end() || !found->second->launches.count(id)) {
    writer.mark_capture_error("missing_plan_launch_completion");
    return;
  }
  auto& state = *found->second;
  const auto it = state.launches.find(id);
  const auto& launch = it->second;
  Json configurations = Json::array();
  bool complete = !disabled() && launch.completed && launch.drained && launch.counters_available;
  for (const auto& [index, counts] : launch.counts) {
    const auto& d = counts.device;
    const bool ok = counts.selected && counts.error.empty() && launch.counters_available &&
                    counts.expected_entries > 0 && d.entries == counts.expected_entries && d.dropped == 0 &&
                    d.attempted == d.emitted && d.emitted == counts.received && counts.received == counts.written &&
                    d.emitted_bytes == counts.received * sizeof(plan_observation_t);
    const bool incomplete = !counts.error.empty() || (counts.selected && !ok);
    const auto* status = incomplete ? "incomplete" : counts.selected ? "complete" : "not_selected";
    if (incomplete) {
      complete = false;
    }
    configurations.push_back({{"config_index", index},
                              {"config_id", impl.configs[index].source->id},
                              {"status", status},
                              {"error", counts.error},
                              {"selected", counts.selected},
                              {"expected_entry_warps", counts.expected_entries},
                              {"entry_warps", d.entries},
                              {"attempted_records", d.attempted},
                              {"device_records", d.emitted},
                              {"device_dropped", d.dropped},
                              {"attempted_bytes", d.attempted_bytes},
                              {"device_bytes", d.emitted_bytes},
                              {"received_records", counts.received},
                              {"written_records", counts.written},
                              {"json_bytes", counts.json_bytes}});
  }
  Json completion{{"type", "plan_capture_completion"},
                  {"schema", "cutracer.plan-completion/v1"},
                  {"canonical_sha256", impl.plan.canonical_sha256()},
                  {"launch", launch.identity.at("launch")},
                  {"grid_launch_id", id},
                  {"kernel_completed", launch.completed},
                  {"channel_drained", launch.drained},
                  {"counters_available", launch.counters_available},
                  {"complete", complete},
                  {"configs", configurations},
                  {"trace_footer_required", true}};
  if (!complete) {
    writer.mark_capture_error("plan_capture_incomplete");
  }
  writer.write_metadata(completion);
  impl.sink("plan_capture_completion", completion);
  state.launches.erase(it);
}

void PlanRuntime::forget_module(CUcontext ctx, CUmodule module) {
  auto& impl = *impl_;
  std::lock_guard<std::mutex> lock(impl.mutex);
  const auto found = impl.contexts.find(ctx);
  if (found == impl.contexts.end()) {
    return;
  }
  auto& state = *found->second;
  const auto unloaded = [&](CUmodule owner) { return !module || owner == module; };
  for (auto it = state.bound.begin(); it != state.bound.end();) {
    it = unloaded(it->second.module) ? state.bound.erase(it) : std::next(it);
  }
  for (auto it = state.skipped.begin(); it != state.skipped.end();) {
    it = unloaded(it->second) ? state.skipped.erase(it) : std::next(it);
  }
  for (auto it = state.injected.begin(); it != state.injected.end();) {
    it = unloaded(state.owners.at(it->first)) ? state.injected.erase(it) : std::next(it);
  }
  for (auto it = state.owners.begin(); it != state.owners.end();) {
    it = unloaded(it->second) ? state.owners.erase(it) : std::next(it);
  }
}

void PlanRuntime::destroy_context(CUcontext ctx) {
  auto& impl = *impl_;
  std::lock_guard<std::mutex> lock(impl.mutex);
  const auto found = impl.contexts.find(ctx);
  if (found == impl.contexts.end()) {
    return;
  }
  if (!found->second->launches.empty()) {
    impl.sink("plan_gap",
              {{"reason", "context_ended_with_unfinalized_captures"}, {"count", found->second->launches.size()}});
  }
  // NVBit can notify after the driver has destroyed this context. Its managed
  // allocations are then already reclaimed by CUDA; do not create a new
  // context through the runtime or turn this normal teardown into a failure.
  CUcontext current = nullptr;
  if (cuCtxGetCurrent(&current) == CUDA_SUCCESS && current == ctx) {
    for (auto* allocation : found->second->allocations) {
      const auto status = cuMemFree(reinterpret_cast<CUdeviceptr>(allocation));
      if (status != CUDA_ERROR_INVALID_CONTEXT && status != CUDA_ERROR_CONTEXT_IS_DESTROYED) {
        cuda_ok(status);
      }
    }
  }
  impl.contexts.erase(found);
}
}  // namespace cutracer
