/*
 * SPDX-FileCopyrightText: Copyright (c) Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: MIT
 */
#include "instrumentation_plan.h"

#include <picosha2.h>

#include <algorithm>
#include <charconv>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <tuple>
#include <utility>

namespace cutracer::plan {
namespace {
using Json = nlohmann::json;
constexpr size_t kNodesMax = 500000;
constexpr size_t kKernelsMax = 256;
constexpr size_t kConfigsMax = 4096;
constexpr size_t kPointsMax = 65536;
constexpr size_t kArrayMax = 65536;
constexpr size_t kObservationsMax = 512;

[[noreturn]] void fail(const std::string& detail) {
  throw PlanError("instrumentation plan: " + detail);
}

void keys(const Json& j, std::initializer_list<const char*> expected) {
  if (!j.is_object() || j.size() != expected.size()) {
    fail("object has missing or unknown fields");
  }
  for (const auto* key : expected) {
    if (!j.contains(key)) {
      fail(std::string("missing field ") + key);
    }
  }
}

std::string text(const Json& j) {
  if (!j.is_string()) {
    fail("expected string");
  }
  const auto& value = j.get_ref<const std::string&>();
  if (value.empty() || value.size() > 1024 || value.find('\0') != std::string::npos) {
    fail("empty or oversized string or embedded NUL");
  }
  return value;
}

void constant(const Json& j, const char* expected) {
  if (text(j) != expected) {
    fail(std::string("expected ") + expected);
  }
}

uint64_t integer(const Json& j, uint64_t low = 0, uint64_t high = kExactIntegerMax) {
  if (!j.is_number_integer() || (j.is_number_integer() && !j.is_number_unsigned() && j.get<int64_t>() < 0)) {
    fail("expected nonnegative exact integer");
  }
  const auto value = j.get<uint64_t>();
  if (value < low || value > high) {
    fail("integer outside permitted range");
  }
  return value;
}

void array(const Json& j, size_t maximum = kArrayMax) {
  if (!j.is_array() || j.empty() || j.size() > maximum) {
    fail("expected nonempty bounded array");
  }
}

template <class T>
void unique_sort(std::vector<T>& values) {
  std::sort(values.begin(), values.end());
  if (std::adjacent_find(values.begin(), values.end()) != values.end()) {
    fail("duplicate selection or request ID");
  }
}

std::vector<uint64_t> integers(const Json& j, uint64_t maximum = kExactIntegerMax) {
  array(j);
  std::vector<uint64_t> result;
  for (const auto& item : j) {
    result.push_back(integer(item, 0, maximum));
  }
  unique_sort(result);
  return result;
}

std::vector<std::string> strings(const Json& j) {
  array(j);
  std::vector<std::string> result;
  for (const auto& item : j) {
    result.push_back(text(item));
  }
  unique_sort(result);
  return result;
}

bool hex(const std::string& value) {
  return std::all_of(value.begin(), value.end(),
                     [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

std::string hex_pc(uint64_t value) {
  std::ostringstream out;
  out << "0x" << std::hex << value;
  return out.str();
}

uint64_t pc(const Json& j) {
  const auto value = text(j);
  if (value.size() < 3 || value.size() > 18 || value.substr(0, 2) != "0x" || !hex(value.substr(2))) {
    fail("PC must be a uint64 lowercase hex byte offset");
  }
  uint64_t result = 0;
  const auto parsed = std::from_chars(value.data() + 2, value.data() + value.size(), result, 16);
  if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
    fail("invalid PC");
  }
  return result;
}

Register reg(const Json& j) {
  const auto value = text(j);
  size_t prefix = 1;
  RegisterBank bank;
  if (value.substr(0, 2) == "UR") {
    bank = RegisterBank::UR;
    prefix = 2;
  } else if (value.substr(0, 2) == "UP") {
    bank = RegisterBank::UP;
    prefix = 2;
  } else if (value[0] == 'R') {
    bank = RegisterBank::R;
  } else if (value[0] == 'P') {
    bank = RegisterBank::P;
  } else {
    fail("unknown register bank");
  }
  if (value.size() == prefix ||
      !std::all_of(value.begin() + prefix, value.end(), [](char c) { return c >= '0' && c <= '9'; })) {
    fail("invalid register index");
  }
  uint32_t index = 0;
  const auto parsed = std::from_chars(value.data() + prefix, value.data() + value.size(), index);
  if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size()) {
    fail("register index exceeds uint32");
  }
  return {bank, index};
}

const char* kind_name(ObservationKind kind) {
  switch (kind) {
    case ObservationKind::Registers:
      return "reg_trace";
    case ObservationKind::MemoryAddress:
      return "mem_addr_trace";
    case ObservationKind::MemoryValue:
      return "mem_value_trace";
    case ObservationKind::Opcode:
      return "opcode_only";
    case ObservationKind::Tma:
      return "tma_trace";
    case ObservationKind::Delay:
      return "random_delay";
  }
  fail("invalid observation kind");
}

Json encode(const Observation& observation) {
  Json result{{"kind", kind_name(observation.kind)}};
  switch (observation.kind) {
    case ObservationKind::Registers: {
      std::vector<std::string> names;
      names.reserve(observation.registers.size());
      for (const auto& r : observation.registers) {
        names.push_back(r.name());
      }
      std::sort(names.begin(), names.end());
      result["registers"] = names;
      break;
    }
    case ObservationKind::MemoryAddress:
    case ObservationKind::MemoryValue:
      result["mref"] = observation.mref;
      break;
    case ObservationKind::Delay:
      result["delay_ns"] = observation.delay_ns;
      break;
    case ObservationKind::Opcode:
    case ObservationKind::Tma:
      break;
  }
  return result;
}

Observation observation(const Json& j) {
  if (!j.is_object() || !j.contains("kind")) {
    fail("observation needs a kind");
  }
  const auto kind = text(j.at("kind"));
  if (kind == "reg_trace") {
    keys(j, {"kind", "registers"});
    array(j.at("registers"), 512);
    std::map<std::string, Register> registers;
    for (const auto& item : j.at("registers")) {
      auto r = reg(item);
      if (!registers.emplace(r.name(), r).second) {
        fail("duplicate register");
      }
    }
    Observation result{ObservationKind::Registers, {}};
    for (const auto& [name, r] : registers) {
      result.registers.push_back(r);
    }
    return result;
  }
  if (kind == "mem_addr_trace" || kind == "mem_value_trace") {
    keys(j, {"kind", "mref"});
    return {kind == "mem_addr_trace" ? ObservationKind::MemoryAddress : ObservationKind::MemoryValue,
            {},
            integer(j.at("mref")),
            0};
  }
  if (kind == "random_delay") {
    keys(j, {"kind", "delay_ns"});
    return {ObservationKind::Delay,
            {},
            0,
            static_cast<uint32_t>(integer(j.at("delay_ns"), 1, std::numeric_limits<uint32_t>::max()))};
  }
  keys(j, {"kind"});
  if (kind == "opcode_only") {
    return {ObservationKind::Opcode, {}};
  }
  if (kind == "tma_trace") {
    return {ObservationKind::Tma, {}};
  }
  fail("unknown observation kind " + kind);
}

// Unions preserve each configuration's consumers and budget. A delay cannot
// silently change because another request happens to share this point.
void merge_observations(std::vector<Observation>& dest, const std::vector<Observation>& incoming) {
  using Key = std::pair<ObservationKind, uint64_t>;
  const auto key = [](const Observation& value) -> Key {
    return {value.kind, value.kind == ObservationKind::MemoryAddress || value.kind == ObservationKind::MemoryValue
                            ? value.mref
                            : 0};
  };
  std::map<Key, size_t> positions;
  for (size_t i = 0; i < dest.size(); ++i) {
    positions.emplace(key(dest[i]), i);
  }
  for (const auto& candidate : incoming) {
    auto [position, inserted] = positions.emplace(key(candidate), dest.size());
    if (inserted) {
      if (dest.size() >= kObservationsMax) {
        fail("observation union exceeds limit");
      }
      dest.push_back(candidate);
      continue;
    }
    auto* it = &dest.at(position->second);
    if (it->kind == ObservationKind::Registers) {
      std::map<std::string, Register> regs;
      for (const auto& r : it->registers) {
        regs.emplace(r.name(), r);
      }
      for (const auto& r : candidate.registers) {
        regs.emplace(r.name(), r);
      }
      if (regs.size() > 512) {
        fail("register union exceeds limit");
      }
      it->registers.clear();
      for (const auto& [name, r] : regs) {
        it->registers.push_back(r);
      }
    } else if (it->kind == ObservationKind::Delay && it->delay_ns != candidate.delay_ns) {
      fail("conflicting delay at one point");
    }
  }
  std::vector<std::pair<std::string, Observation>> ordered;
  ordered.reserve(dest.size());
  for (auto& value : dest) {
    auto encoded = encode(value).dump();
    ordered.emplace_back(std::move(encoded), std::move(value));
  }
  std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
  dest.clear();
  for (auto& value : ordered) {
    dest.push_back(std::move(value.second));
  }
}

Point point(const Json& j) {
  keys(j, {"function", "pc", "expected_bytes", "phase", "predicate_policy", "observations"});
  Point result;
  result.function = text(j.at("function"));
  result.pc = pc(j.at("pc"));
  result.expected_bytes = text(j.at("expected_bytes"));
  if (result.expected_bytes.size() < 16 || result.expected_bytes.size() > 64 || result.expected_bytes.size() % 2 ||
      !hex(result.expected_bytes)) {
    fail("instruction bytes must be 8 to 32 lowercase hex bytes");
  }
  const auto phase = text(j.at("phase"));
  if (phase != "before" && phase != "after") {
    fail("unknown phase");
  }
  result.phase = phase == "before" ? Phase::Before : Phase::After;
  const auto predicate = text(j.at("predicate_policy"));
  if (predicate != "executed_lanes" && predicate != "reached_lanes") {
    fail("unknown predicate policy");
  }
  result.predicate = predicate == "executed_lanes" ? PredicatePolicy::Executed : PredicatePolicy::Reached;
  array(j.at("observations"), kObservationsMax);
  std::vector<Observation> incoming;
  incoming.reserve(j.at("observations").size());
  for (const auto& item : j.at("observations")) {
    incoming.push_back(observation(item));
  }
  merge_observations(result.observations, incoming);
  return result;
}

Config config(const Json& j, size_t& point_count) {
  keys(j, {"config_id", "requests", "launches", "capture", "budget", "points"});
  Config result;
  result.id = text(j.at("config_id"));
  result.requests = strings(j.at("requests"));
  const auto& launches = j.at("launches");
  if (!launches.is_object() || !launches.contains("mode")) {
    fail("launch selector needs mode");
  }
  const auto mode = text(launches.at("mode"));
  if (mode == "all") {
    keys(launches, {"mode"});
  } else if (mode == "per_context_kernel_ordinal") {
    keys(launches, {"mode", "indices"});
    result.launches.all = false;
    result.launches.ordinals = integers(launches.at("indices"));
  } else {
    fail("unknown launch selector mode");
  }
  const auto& capture = j.at("capture");
  keys(capture, {"ctas", "warps", "lanes"});
  const auto& ctas = capture.at("ctas");
  if (ctas.is_string()) {
    constant(ctas, "all");
  } else {
    array(ctas);
    for (const auto& cta : ctas) {
      if (!cta.is_array() || cta.size() != 3) {
        fail("CTA selection needs three coordinates");
      }
      result.capture.ctas.push_back({integer(cta[0]), integer(cta[1]), integer(cta[2])});
    }
    unique_sort(result.capture.ctas);
  }
  for (const auto* name : {"warps", "lanes"}) {
    const auto& values = capture.at(name);
    if (values.is_string()) {
      constant(values, "all");
    } else if (std::string(name) == "warps") {
      result.capture.warps = integers(values);
    } else {
      result.capture.lanes = integers(values, 31);
    }
  }
  const auto& budget = j.at("budget");
  keys(budget, {"max_records", "max_bytes", "on_limit"});
  constant(budget.at("on_limit"), "mark_incomplete");
  result.budget = {integer(budget.at("max_records"), 1), integer(budget.at("max_bytes"), 1)};
  array(j.at("points"), kPointsMax);
  std::map<std::tuple<std::string, uint64_t, Phase>, Point> points;
  for (const auto& item : j.at("points")) {
    if (++point_count > kPointsMax) {
      fail("total point limit exceeded");
    }
    auto candidate = point(item);
    const auto key = std::make_tuple(candidate.function, candidate.pc, candidate.phase);
    auto [it, added] = points.emplace(key, candidate);
    if (!added) {
      if (it->second.expected_bytes != candidate.expected_bytes || it->second.predicate != candidate.predicate) {
        fail("conflicting bytes or predicate policy at one point");
      }
      merge_observations(it->second.observations, candidate.observations);
    }
  }
  for (auto& [key, value] : points) {
    result.points.push_back(std::move(value));
  }
  return result;
}

Kernel kernel(const Json& j, size_t& config_count, size_t& point_count) {
  keys(j, {"match", "pc_namespace", "configs"});
  constant(j.at("pc_namespace"), "function_relative_bytes");
  const auto& match = j.at("match");
  keys(match, {"entry", "target", "identity"});
  Kernel result;
  result.entry = text(match.at("entry"));
  result.target = text(match.at("target"));
  if (result.target.size() < 4) {
    fail("invalid target");
  }
  auto arch = result.target.substr(3);
  if (!arch.empty() && (arch.back() == 'a' || arch.back() == 'f')) {
    arch.pop_back();
  }
  if (result.target.substr(0, 3) != "sm_" || arch.empty() ||
      !std::all_of(arch.begin(), arch.end(), [](char c) { return c >= '0' && c <= '9'; })) {
    fail("invalid target");
  }
  const auto& identity = match.at("identity");
  keys(identity, {"kind", "value"});
  constant(identity.at("kind"), "cubin_sha256");
  result.cubin_sha256 = text(identity.at("value"));
  if (result.cubin_sha256.size() != 64 || !hex(result.cubin_sha256)) {
    fail("invalid cubin SHA256");
  }
  array(j.at("configs"), kConfigsMax);
  for (const auto& item : j.at("configs")) {
    if (++config_count > kConfigsMax) {
      fail("total config limit exceeded");
    }
    result.configs.push_back(config(item, point_count));
  }
  return result;
}

bool policy(const Json& j) {
  keys(j, {"unmatched_kernel", "unsupported_request", "identity_mismatch", "conflicts"});
  constant(j.at("unmatched_kernel"), "skip");
  constant(j.at("identity_mismatch"), "error");
  constant(j.at("conflicts"), "error");
  const auto unsupported = text(j.at("unsupported_request"));
  if (unsupported != "error" && unsupported != "report_gap") {
    fail("unknown unsupported request policy");
  }
  return unsupported == "report_gap";
}

Json parse_json(const std::string& bytes, size_t& nodes) {
  std::vector<std::set<std::string>> objects;
  auto callback = [&](int depth, Json::parse_event_t event, Json& parsed) {
    if (depth > 64 || ++nodes > kNodesMax) {
      fail("JSON nesting or node budget exceeded");
    }
    if (event == Json::parse_event_t::object_start) {
      objects.emplace_back();
    } else if (event == Json::parse_event_t::object_end) {
      objects.pop_back();
    } else if (event == Json::parse_event_t::key) {
      if (!objects.back().insert(parsed.get<std::string>()).second) {
        fail("duplicate JSON key");
      }
    }
    return true;
  };
  return Json::parse(bytes, callback);
}

std::vector<Json> documents(const std::string& bytes) {
  size_t nodes = 0;
  try {
    return {parse_json(bytes, nodes)};
  } catch (const Json::parse_error&) {
    // A multi-record NDJSON file is not one JSON document. Every individual
    // line must still parse strictly, and no partial plan escapes this function.
  }
  nodes = 0;
  std::vector<Json> result;
  std::istringstream input(bytes);
  for (std::string line; std::getline(input, line);) {
    if (line.find_first_not_of(" \r\t") == std::string::npos) {
      continue;
    }
    if (result.size() == kKernelsMax) {
      fail("too many NDJSON records");
    }
    result.push_back(parse_json(line, nodes));
  }
  if (result.empty()) {
    fail("empty plan");
  }
  return result;
}

std::vector<Kernel> normalize(std::vector<Kernel> kernels) {
  using KernelKey = std::tuple<std::string, std::string, std::string>;
  std::map<KernelKey, Kernel> grouped;
  // Instruction bytes belong to the binary/function, even when several
  // entries or configurations refer to it.
  std::map<std::tuple<std::string, std::string, uint64_t>, std::string> bytes;
  for (auto& k : kernels) {
    const auto key = std::make_tuple(k.cubin_sha256, k.entry, k.target);
    auto [it, inserted] = grouped.emplace(key, Kernel{k.entry, k.target, k.cubin_sha256, {}});
    for (auto& c : k.configs) {
      for (const auto& p : c.points) {
        auto [bit, new_bytes] = bytes.emplace(std::make_tuple(k.cubin_sha256, p.function, p.pc), p.expected_bytes);
        if (!new_bytes && bit->second != p.expected_bytes) {
          fail("conflicting instruction bytes across configurations");
        }
      }
      it->second.configs.push_back(std::move(c));
    }
  }
  std::vector<Kernel> result;
  for (auto& [key, k] : grouped) {
    std::map<std::string, Config> configs;
    for (auto& c : k.configs) {
      auto [it, inserted] = configs.emplace(c.id, c);
      if (!inserted && to_json(it->second) != to_json(c)) {
        fail("conflicting config ID");
      }
    }
    k.configs.clear();
    for (auto& [id, c] : configs) {
      k.configs.push_back(std::move(c));
    }
    result.push_back(std::move(k));
  }
  return result;
}

}  // namespace

std::string Register::name() const {
  const char* prefix = bank == RegisterBank::R    ? "R"
                       : bank == RegisterBank::UR ? "UR"
                       : bank == RegisterBank::P  ? "P"
                                                  : "UP";
  return prefix + std::to_string(index);
}

bool LaunchSelector::selects(uint64_t ordinal) const {
  return all || std::binary_search(ordinals.begin(), ordinals.end(), ordinal);
}

Json to_json(const Point& point) {
  Json observations = Json::array();
  for (const auto& observation : point.observations) {
    observations.push_back(encode(observation));
  }
  return {{"function", point.function},
          {"pc", hex_pc(point.pc)},
          {"expected_bytes", point.expected_bytes},
          {"phase", point.phase == Phase::Before ? "before" : "after"},
          {"predicate_policy", point.predicate == PredicatePolicy::Executed ? "executed_lanes" : "reached_lanes"},
          {"observations", observations}};
}

Json to_json(const Config& config) {
  Json launches{{"mode", config.launches.all ? "all" : "per_context_kernel_ordinal"}};
  if (!config.launches.all) {
    launches["indices"] = config.launches.ordinals;
  }
  Json capture{{"ctas", "all"}, {"warps", "all"}, {"lanes", "all"}};
  if (!config.capture.ctas.empty()) {
    capture["ctas"] = config.capture.ctas;
  }
  if (!config.capture.warps.empty()) {
    capture["warps"] = config.capture.warps;
  }
  if (!config.capture.lanes.empty()) {
    capture["lanes"] = config.capture.lanes;
  }
  Json points = Json::array();
  for (const auto& point : config.points) {
    points.push_back(to_json(point));
  }
  return {{"config_id", config.id},
          {"requests", config.requests},
          {"launches", launches},
          {"capture", capture},
          {"budget",
           {{"max_records", config.budget.max_records},
            {"max_bytes", config.budget.max_bytes},
            {"on_limit", "mark_incomplete"}}},
          {"points", points}};
}

Json to_json(const Kernel& kernel) {
  Json configs = Json::array();
  for (const auto& config : kernel.configs) {
    configs.push_back(to_json(config));
  }
  return {{"match",
           {{"entry", kernel.entry},
            {"target", kernel.target},
            {"identity", {{"kind", "cubin_sha256"}, {"value", kernel.cubin_sha256}}}}},
          {"pc_namespace", "function_relative_bytes"},
          {"configs", configs}};
}

CanonicalPlan::CanonicalPlan(std::string id, bool report_gap, std::vector<Kernel> kernels, Json canonical,
                             std::string source_sha256)
    : id_(std::move(id)),
      report_gap_(report_gap),
      kernels_(std::move(kernels)),
      canonical_(std::move(canonical)),
      canonical_sha256_(picosha2::hash256_hex_string(canonical_.dump(-1, ' ', true))),
      source_sha256_(std::move(source_sha256)) {
}

CanonicalPlan PlanLoader::parse(const std::string& bytes) {
  if (bytes.empty() || bytes.size() > kFileBytesMax) {
    fail("empty or oversized plan file");
  }
  // The JSON lexer treats raw NUL as EOF and would silently ignore the tail.
  if (bytes.find('\0') != std::string::npos) {
    fail("raw NUL in plan file");
  }
  try {
    const auto docs = documents(bytes);
    std::string id;
    Json header;
    std::vector<Kernel> kernels;
    size_t config_count = 0, point_count = 0;
    bool report_gap = false;
    for (const auto& doc : docs) {
      if (!doc.is_object() || !doc.contains("schema")) {
        fail("plan needs schema");
      }
      const auto schema = text(doc.at("schema"));
      if (schema == kSchema) {
        keys(doc, {"schema", "plan_id", "policy", "kernels"});
        if (docs.size() != 1) {
          fail("document cannot be mixed with NDJSON records");
        }
        array(doc.at("kernels"), kKernelsMax);
      } else if (schema == "cutracer.kernel_plan/v1") {
        keys(doc, {"schema", "plan_id", "policy", "kernel"});
      } else {
        fail("unknown schema version");
      }
      const auto candidate_id = text(doc.at("plan_id"));
      const bool candidate_policy = policy(doc.at("policy"));
      if (id.empty()) {
        id = candidate_id;
        header = doc.at("policy");
        report_gap = candidate_policy;
      } else if (id != candidate_id || header != doc.at("policy")) {
        fail("inconsistent NDJSON plan ID or policy");
      }
      if (schema == kSchema) {
        for (const auto& item : doc.at("kernels")) {
          kernels.push_back(kernel(item, config_count, point_count));
        }
      } else {
        kernels.push_back(kernel(doc.at("kernel"), config_count, point_count));
      }
    }
    kernels = normalize(std::move(kernels));
    Json serialized = Json::array();
    for (const auto& k : kernels) {
      serialized.push_back(to_json(k));
    }
    Json canonical{{"schema", kSchema}, {"plan_id", id}, {"policy", header}, {"kernels", serialized}};
    return CanonicalPlan(std::move(id), report_gap, std::move(kernels), std::move(canonical),
                         picosha2::hash256_hex_string(bytes));
  } catch (const Json::exception& error) {
    fail(std::string("invalid JSON: ") + error.what());
  }
}

CanonicalPlan PlanLoader::load(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    fail("cannot open plan file " + path);
  }
  std::string bytes;
  std::array<char, 65536> buffer{};
  while (file) {
    file.read(buffer.data(), buffer.size());
    if (bytes.size() + file.gcount() > kFileBytesMax) {
      fail("oversized plan file");
    }
    bytes.append(buffer.data(), static_cast<size_t>(file.gcount()));
  }
  if (file.bad() || !file.eof()) {
    fail("cannot read complete plan file");
  }
  return parse(bytes);
}

}  // namespace cutracer::plan
