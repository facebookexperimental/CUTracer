# Copyright (c) Meta Platforms, Inc. and affiliates.
# Makefile for CUTracer unit tests

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -I../../include

# Test executables
TESTS = test_instr_category test_kernel_types

.PHONY: all clean test

all: $(TESTS)

test_instr_category: test_instr_category.cpp ../../include/instr_category.h
	$(CXX) $(CXXFLAGS) -o $@ $<

test_kernel_types: test_kernel_types.cpp ../../include/kernel_types.h
	$(CXX) $(CXXFLAGS) -o $@ $<

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
