# MIT License
# Copyright (c) Meta Platforms, Inc. and affiliates.
# See LICENSE file for details.
# Project name
PROJECT := cutracer

# Compiler settings
CXX ?=
NVCC=nvcc -ccbin=$(CXX) -D_FORCE_INLINES
PTXAS=ptxas
BIN2C=bin2c

# Version checks
NVCC_VER_REQ=10.1
NVCC_VER=$(shell $(NVCC) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
NVCC_VER_CHECK=$(shell echo "${NVCC_VER} >= $(NVCC_VER_REQ)" | bc)

ifeq ($(NVCC_VER_CHECK),0)
$(error ERROR: nvcc version >= $(NVCC_VER_REQ) required to compile an nvbit tool! Instrumented applications can still use lower versions of nvcc.)
endif

PTXAS_VER_ADD_FLAG=12.3
PTXAS_VER=$(shell $(PTXAS) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
PTXAS_VER_CHECK=$(shell echo "${PTXAS_VER} >= $(PTXAS_VER_ADD_FLAG)" | bc)

ifeq ($(PTXAS_VER_CHECK), 0)
MAXRREGCOUNT_FLAG=-maxrregcount=24
else
MAXRREGCOUNT_FLAG=
endif

# Debug settings
ifeq ($(DEBUG),1)
DEBUG_FLAGS := -g -O0
else
DEBUG_FLAGS := -O3 -g
endif

# Directory structure
SRC_DIR := src
OBJ_DIR := obj
LIB_DIR := lib
INCLUDE_DIR := include

# NVBIT settings
NVBIT_PATH=./third_party/nvbit/core
PICOSHA2_PATH=./third_party/picosha2
INCLUDES=-I$(NVBIT_PATH) -I./$(INCLUDE_DIR) -I./third_party -I$(PICOSHA2_PATH)

# Libraries
# zstd linking strategy:
#   - RHEL/CentOS/Fedora: static linking (their libzstd.a is PIC-compatible)
#   - Ubuntu/Debian/others: dynamic linking (their libzstd.a lacks -fPIC)
#   - Override with STATIC_ZSTD=1 or DYNAMIC_ZSTD=1
# Note: -lpthread is required because zstd uses POSIX threads internally

# Detect OS type from /etc/os-release
OS_ID := $(shell . /etc/os-release 2>/dev/null && echo $$ID)
OS_ID_LIKE := $(shell . /etc/os-release 2>/dev/null && echo $$ID_LIKE)
IS_RHEL_LIKE := $(if $(or $(findstring rhel,$(OS_ID) $(OS_ID_LIKE)),\
                          $(findstring centos,$(OS_ID)),\
                          $(findstring fedora,$(OS_ID)),\
                          $(findstring rocky,$(OS_ID)),\
                          $(findstring almalinux,$(OS_ID))),1,)

# Helper function to find static zstd library
define find_static_zstd
$(or $(wildcard $(shell pkg-config --variable=libdir libzstd 2>/dev/null)/libzstd.a),\
     $(if $(filter-out libzstd.a,$(shell $(CC) -print-file-name=libzstd.a 2>/dev/null)),\
          $(shell $(CC) -print-file-name=libzstd.a 2>/dev/null),))
endef

ifdef DYNAMIC_ZSTD
    # User explicitly requested dynamic linking
    ZSTD_LIB := -lzstd
else ifdef STATIC_ZSTD
    # User explicitly requested static linking
    ZSTD_LIB := $(call find_static_zstd)
    ifeq ($(ZSTD_LIB),)
        $(error ERROR: libzstd.a not found. Install with: dnf install libzstd-static (RHEL/Fedora) or apt install libzstd-dev (Ubuntu/Debian))
    endif
else ifdef IS_RHEL_LIKE
    # RHEL-like OS: default to static linking (their static lib is PIC-compatible)
    ZSTD_LIB := $(call find_static_zstd)
    ifeq ($(ZSTD_LIB),)
        $(error ERROR: libzstd.a not found. Install with: dnf install libzstd-static)
    endif
else
    # Other OS (Ubuntu, Debian, etc.): default to dynamic linking
    # Their libzstd.a is not compiled with -fPIC, so it can't be linked into a .so
    ZSTD_LIB := -lzstd
endif

LIBS=-L$(NVBIT_PATH) -lnvbit $(ZSTD_LIB) -lpthread -ldl
NVCC_PATH=-L $(subst bin/nvcc,lib64,$(shell which nvcc | tr -s /))

# Identify inject_funcs.cu specifically
INJECT_FUNCS_SRC := $(SRC_DIR)/inject_funcs.cu
INJECT_FUNCS_OBJ := $(OBJ_DIR)/inject_funcs.o

# Source files (excluding inject_funcs.cu)
CU_SRCS := $(filter-out $(INJECT_FUNCS_SRC),$(wildcard $(SRC_DIR)/*.cu))
CPP_SRCS := $(wildcard $(SRC_DIR)/*.cpp)

# Internal fb/ source files (only compiled if fb/ directory exists)
FB_SRC_DIR := $(SRC_DIR)/fb
FB_CU_SRCS := $(wildcard $(FB_SRC_DIR)/*.cu)
# Separate inject_funcs_fb.cu from other fb/ files (needs special flags)
FB_INJECT_FUNCS_SRC := $(FB_SRC_DIR)/inject_funcs_fb.cu
FB_INJECT_FUNCS_OBJ := $(if $(wildcard $(FB_INJECT_FUNCS_SRC)),$(OBJ_DIR)/fb_inject_funcs_fb.o,)
FB_REGULAR_CU_SRCS := $(filter-out $(FB_INJECT_FUNCS_SRC),$(FB_CU_SRCS))
FB_REGULAR_OBJS := $(patsubst $(FB_SRC_DIR)/%.cu,$(OBJ_DIR)/fb_%.o,$(FB_REGULAR_CU_SRCS))
FB_OBJS := $(FB_REGULAR_OBJS) $(FB_INJECT_FUNCS_OBJ)

# Object files
REGULAR_OBJS := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(CU_SRCS))
CPP_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SRCS))

# All objects (regular + inject_funcs + cpp + fb)
OBJS := $(REGULAR_OBJS) $(INJECT_FUNCS_OBJ) $(CPP_OBJS) $(FB_OBJS)

# Tool function sources (precompiled fatbin for flush_channel)
TOOL_FUNC_DIR := $(SRC_DIR)/tool_func
TOOL_FUNC_SOURCES := $(wildcard $(TOOL_FUNC_DIR)/*.cu)
TOOL_FUNC_FATBINS := $(TOOL_FUNC_SOURCES:.cu=.fatbin)
TOOL_FUNC_BIN2CS := $(TOOL_FUNC_FATBINS:.fatbin=.c)

# Architecture
ARCH?=all

# Output file
NVBIT_TOOL=$(LIB_DIR)/$(PROJECT).so

# Main targets
all: dirs $(NVBIT_TOOL)
	@echo ""
	@echo "✅ Build successful! Output: $(NVBIT_TOOL)"
	@echo ""

dirs: $(OBJ_DIR) $(LIB_DIR)

$(OBJ_DIR):
	mkdir -p $@

$(LIB_DIR):
	mkdir -p $@

# Linking rule
$(NVBIT_TOOL): $(OBJS) $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) $(DEBUG_FLAGS) $(OBJS) $(LIBS) $(NVCC_PATH) -Wno-deprecated-gpu-targets -lcuda -lcudart_static -shared -Xcompiler -rdynamic -o $@

# Compile tool_func/*.cu → *.fatbin
$(TOOL_FUNC_DIR)/%.fatbin: $(TOOL_FUNC_DIR)/%.cu
	$(NVCC) -arch=$(ARCH) $(INCLUDES) $(DEBUG_FLAGS) -Wno-deprecated-gpu-targets -fatbin $< -o $@

# Convert *.fatbin → *.c (embedded as C byte array via bin2c)
$(TOOL_FUNC_DIR)/%.c: $(TOOL_FUNC_DIR)/%.fatbin
	$(BIN2C) -c --name $(basename $(notdir $<))_bin $< > $@

# Compilation rule for regular CUDA files (excluding inject_funcs.cu)
# Depends on bin2c outputs so #include "tool_func/flush_channel.c" is available
$(REGULAR_OBJS): $(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu $(TOOL_FUNC_BIN2CS)
	$(NVCC) -dc -c -std=c++17 $(INCLUDES) -Xptxas -cloning=no -Wno-deprecated-gpu-targets -Xcompiler -Wall -arch=$(ARCH) $(DEBUG_FLAGS) -Xcompiler -fPIC $< -o $@

# Special rule for inject_funcs.cu
$(INJECT_FUNCS_OBJ): $(INJECT_FUNCS_SRC)
	$(NVCC) $(INCLUDES) $(MAXRREGCOUNT_FLAG) -Wno-deprecated-gpu-targets -Xptxas -astoolspatch --keep-device-functions -arch=$(ARCH) -Xcompiler -Wall -Xcompiler -fPIC -c $< -o $@

# Compilation rule for C++ files
$(OBJ_DIR)/cubin_identity.o: $(PICOSHA2_PATH)/picosha2.h

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) -std=c++17 $(INCLUDES) -Wall $(DEBUG_FLAGS) -fPIC -c $< -o $@

# Compilation rule for internal fb/ CUDA files (only if fb/ directory exists)
$(OBJ_DIR)/fb_%.o: $(FB_SRC_DIR)/%.cu
	$(NVCC) -dc -c -std=c++17 $(INCLUDES) -Xptxas -cloning=no -Wno-deprecated-gpu-targets -Xcompiler -Wall -arch=$(ARCH) $(DEBUG_FLAGS) -Xcompiler -fPIC $< -o $@

# Special rule for inject_funcs_fb.cu (same flags as inject_funcs.cu for NVBit device functions)
$(OBJ_DIR)/fb_inject_funcs_fb.o: $(FB_INJECT_FUNCS_SRC)
	$(NVCC) $(INCLUDES) $(MAXRREGCOUNT_FLAG) -Wno-deprecated-gpu-targets -Xptxas -astoolspatch --keep-device-functions -arch=$(ARCH) -Xcompiler -Wall -Xcompiler -fPIC -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR) $(TOOL_FUNC_DIR)/*.fatbin $(TOOL_FUNC_DIR)/*.c

.PHONY: all clean dirs
