// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
//
// NOT SYNCED TO OSS - tests for fb/implicit_regs_fb.h.

/**
 * @file test_implicit_regs_fb.cpp
 * @brief Unit tests for collect_utc_mma_implicit_regs / collect_implicit_regs.
 *
 * Compile and run:
 *   g++ -std=c++17 -I../../include -o test_implicit_regs_fb test_implicit_regs_fb.cpp
 *   ./test_implicit_regs_fb
 *
 * Key regression coverage: A=tmem path in UTC*MMA must still push URx+1 for
 * the B-matrix gdesc. The earlier `generic_strs[1] must be gdesc` guard
 * regressed this path -- see header doc in fb/implicit_regs_fb.h.
 */

#include <cassert>
#include <iostream>
#include <string>
#include <vector>

// We want this test to stay standalone (no buck2, no CUDA toolkit, no
// nlohmann::json). instrument.h transitively pulls in analysis.h ->
// nlohmann/json.hpp, which is not on the bare-g++ include path here, so we
// inline a minimal OperandLists definition that must mirror the one in
// include/instrument.h. fb/implicit_regs_fb.h only accesses ureg_nums, but the
// full struct definition is required at the point the inline function uses it.
struct OperandLists {
  std::vector<int> reg_nums;
  std::vector<int> ureg_nums;
  std::vector<int> desc_urs;
};

#include "implicit_regs.h"  // OperandContext, collect_implicit_regs, collect_utc_mma_implicit_regs

// ============================================================================
// Mini test harness (matches style of test_instr_category.cpp /
// test_kernel_types.cpp in this directory).
// ============================================================================

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name)                             \
  std::cout << "Testing: " << #name << "... "; \
  if (test_##name()) {                         \
    std::cout << "PASSED" << std::endl;        \
    tests_passed++;                            \
  } else {                                     \
    std::cout << "FAILED" << std::endl;        \
    tests_failed++;                            \
  }

static bool contains(const std::vector<int>& v, int x) {
  for (int e : v) {
    if (e == x) return true;
  }
  return false;
}

// ============================================================================
// Test Cases
// ============================================================================

// REGRESSION: A=tmem path (B is the only gdesc, idesc is also GENERIC).
// Before the fix, the `generic_strs[1].find("gdesc") == npos` guard returned
// false here because [1] == "idesc[UR69]", so UR_B+1 was never pushed and the
// Python BlackwellDecoder skipped the event.
bool test_a_is_tmem_b_is_gdesc() {
  OperandContext ctx;
  // ctx.generic_* mirrors what NVBit reports for
  //   "UTCHMMA tmem[UR62], gdesc[UR48], tmem[UR15], tmem[UR68], idesc[UR69], UPT ;"
  // tmem[] -> UREG (captured by main loop, not present in ctx.generic_*).
  ctx.generic_strs = {"gdesc[UR48]", "idesc[UR69]"};
  ctx.generic_urs = {48, 69};

  OperandLists operands;
  bool ok = collect_utc_mma_implicit_regs("UTCHMMA tmem[UR62], gdesc[UR48], tmem[UR15], tmem[UR68], idesc[UR69], UPT ;",
                                          ctx, operands);
  if (!ok) return false;
  // Must include UR49 (high half of gdesc[UR48]).
  if (!contains(operands.ureg_nums, 49)) return false;
  // Must NOT include UR70 (idesc +1 is deferred; see TODO in fb header).
  if (contains(operands.ureg_nums, 70)) return false;
  return true;
}

// REGRESSION: A=gdesc path (the originally-working case).
// Two gdesc operands plus one idesc; ctx.generic_strs is length 3.
bool test_a_is_gdesc_b_is_gdesc() {
  OperandContext ctx;
  // From observed SASS:
  //   "UTCHMMA gdesc[UR32], gdesc[UR48], tmem[UR17], tmem[UR4], idesc[UR5], !UPT ;"
  ctx.generic_strs = {"gdesc[UR32]", "gdesc[UR48]", "idesc[UR5]"};
  ctx.generic_urs = {32, 48, 5};

  OperandLists operands;
  bool ok = collect_utc_mma_implicit_regs("UTCHMMA gdesc[UR32], gdesc[UR48], tmem[UR17], tmem[UR4], idesc[UR5], !UPT ;",
                                          ctx, operands);
  if (!ok) return false;
  if (!contains(operands.ureg_nums, 33)) return false;  // gdesc_A high
  if (!contains(operands.ureg_nums, 49)) return false;  // gdesc_B high
  if (contains(operands.ureg_nums, 6)) return false;    // idesc +1 not yet collected
  return true;
}

// Defensive: empty operand context (e.g., parse failure upstream).
// Must not crash and must not append anything.
bool test_empty_ctx() {
  OperandContext ctx;
  OperandLists operands;
  bool ok = collect_utc_mma_implicit_regs("UTCHMMA <garbage>", ctx, operands);
  return ok && operands.ureg_nums.empty();
}

// Defensive: only an idesc GENERIC operand (no gdesc at all).
// Must not crash; no +1 register should be appended.
bool test_no_gdesc_at_all() {
  OperandContext ctx;
  ctx.generic_strs = {"idesc[UR5]"};
  ctx.generic_urs = {5};

  OperandLists operands;
  bool ok = collect_utc_mma_implicit_regs("UTCHMMA <hypothetical>", ctx, operands);
  return ok && operands.ureg_nums.empty();
}

// REGRESSION: HGMMA path goes through collect_implicit_regs() and must keep
// pushing the 3 implicit URs (n+1, n+2, n+3) for the 128-bit Hopper descriptor.
// Diff 4 only touched UTC*MMA -- this guards against accidental refactor damage.
bool test_hgmma_path_untouched() {
  OperandContext ctx;
  ctx.generic_strs = {"gdesc[UR5].tnspB"};
  ctx.generic_urs = {5};

  OperandLists operands;
  collect_implicit_regs("HGMMA.64x128x16.F32 R0, R4, R8, !UR0", ctx, operands);
  return contains(operands.ureg_nums, 6) && contains(operands.ureg_nums, 7) && contains(operands.ureg_nums, 8);
}

// Top-level dispatcher must route UTC*MMA correctly through
// collect_utc_mma_implicit_regs (covers the strstr match against each alias).
bool test_dispatcher_routes_all_utc_mma_aliases() {
  const char* aliases[] = {
      "UTCHMMA tmem[UR62], gdesc[UR48], tmem[UR15], tmem[UR68], idesc[UR69], UPT ;",
      "UTCIMMA tmem[UR62], gdesc[UR48], tmem[UR15], tmem[UR68], idesc[UR69], UPT ;",
      "UTCQMMA tmem[UR62], gdesc[UR48], tmem[UR15], tmem[UR68], idesc[UR69], UPT ;",
      "UTCOMMA tmem[UR62], gdesc[UR48], tmem[UR15], tmem[UR68], idesc[UR69], UPT ;",
  };
  for (const char* sass : aliases) {
    OperandContext ctx;
    ctx.generic_strs = {"gdesc[UR48]", "idesc[UR69]"};
    ctx.generic_urs = {48, 69};

    OperandLists operands;
    collect_implicit_regs(sass, ctx, operands);
    if (!contains(operands.ureg_nums, 49)) return false;
  }
  return true;
}

// ============================================================================
// Entrypoint
// ============================================================================

int main() {
  std::cout << "Running test_implicit_regs_fb..." << std::endl;
  std::cout << "===============================" << std::endl;

  TEST(a_is_tmem_b_is_gdesc);
  TEST(a_is_gdesc_b_is_gdesc);
  TEST(empty_ctx);
  TEST(no_gdesc_at_all);
  TEST(hgmma_path_untouched);
  TEST(dispatcher_routes_all_utc_mma_aliases);

  std::cout << "===============================" << std::endl;
  std::cout << "Passed: " << tests_passed << ", Failed: " << tests_failed << std::endl;
  return tests_failed == 0 ? 0 : 1;
}
