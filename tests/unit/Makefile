# Copyright (c) Meta Platforms, Inc. and affiliates.
# Makefile for CUTracer unit tests

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -I../../include

# GoogleTest source tests require libgtest-dev on Debian/Ubuntu.
GTEST_CPPFLAGS ?=
GTEST_LIBS ?= -lgtest_main -lgtest -pthread
THIRD_PARTY := ../../third_party
CUDA_HOME ?= /usr/local/cuda

# Test executables
TESTS = test_instr_category test_kernel_types test_cubin_identity test_capture_completion

.PHONY: all clean test

all: $(TESTS)

test_instr_category: test_instr_category.cpp ../../include/instr_category.h
	$(CXX) $(CXXFLAGS) -o $@ $<

test_kernel_types: test_kernel_types.cpp ../../include/kernel_types.h
	$(CXX) $(CXXFLAGS) -o $@ $<

test_cubin_identity: ../../src/test/CubinIdentityTest.cpp ../../src/cubin_identity.cpp \
                    ../../src/test/TemporaryDirectoryTest.h ../../include/cubin_identity.h \
                    $(THIRD_PARTY)/picosha2/picosha2.h
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(GTEST_CPPFLAGS) -I$(THIRD_PARTY)/picosha2 \
		$(filter %.cpp,$^) $(LDFLAGS) $(GTEST_LIBS) -o $@

test_capture_completion: ../../src/test/CaptureCompletionTest.cpp ../../src/trace_writer.cpp \
                         ../../src/trace_rapidjson.cpp ../../src/test/TemporaryDirectoryTest.h \
                         $(wildcard ../../include/*.h)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(GTEST_CPPFLAGS) -I$(THIRD_PARTY) \
		-isystem $(THIRD_PARTY)/nvbit/core -isystem $(CUDA_HOME)/include \
		$(filter %.cpp,$^) $(LDFLAGS) $(GTEST_LIBS) -lzstd -o $@

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
