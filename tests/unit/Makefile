# Copyright (c) Meta Platforms, Inc. and affiliates.
# Makefile for CUTracer unit tests

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -I../../include

# Test executables
TESTS = test_instr_category test_kernel_types test_implicit_regs_fb

.PHONY: all clean test

all: $(TESTS)

test_instr_category: test_instr_category.cpp ../../include/instr_category.h
	$(CXX) $(CXXFLAGS) -o $@ $<

test_kernel_types: test_kernel_types.cpp ../../include/kernel_types.h
	$(CXX) $(CXXFLAGS) -o $@ $<

# implicit_regs.h transitively pulls in log.h which pulls in cuda.h. Use the
# nvbit-bundled cuda.h so this test stays standalone (no CUDA toolkit needed).
NVBIT_INCLUDE = ../../../../../../third-party/nvbit/core
test_implicit_regs_fb: test_implicit_regs_fb.cpp ../../include/implicit_regs.h ../../include/fb/implicit_regs_fb.h ../../include/instrument.h ../../include/common.h
	$(CXX) $(CXXFLAGS) -I$(NVBIT_INCLUDE) -o $@ $<

test: $(TESTS)
	@echo "Running unit tests..."
	@for t in $(TESTS); do \
		echo ""; \
		echo ">>> Running $$t"; \
		./$$t || exit 1; \
	done
	@echo ""
	@echo "All tests passed!"

clean:
	rm -f $(TESTS)
