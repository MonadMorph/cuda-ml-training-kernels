CC = gcc
NVCC = nvcc
PKG_CONFIG ?= pkg-config

# Detect OS for OpenMP linking
UNAME := $(shell uname)

# OpenBLAS flags via pkg-config, with path for macOS
OPENBLAS_CFLAGS := $(shell PKG_CONFIG_PATH="$(shell brew --prefix openblas 2>/dev/null)/lib/pkgconfig:$(PKG_CONFIG_PATH)" $(PKG_CONFIG) --cflags openblas 2>/dev/null)
OPENBLAS_LDFLAGS := $(shell PKG_CONFIG_PATH="$(shell brew --prefix openblas 2>/dev/null)/lib/pkgconfig:$(PKG_CONFIG_PATH)" $(PKG_CONFIG) --libs openblas 2>/dev/null)

CFLAGS = -Wall -Wextra -O3 -fopenmp -ffast-math -Iinclude $(OPENBLAS_CFLAGS)
NVFLAGS = -O3 --use_fast_math -Iinclude
# -fopenmp is required at link time on GCC to pull in libgomp
LDFLAGS = $(OPENBLAS_LDFLAGS) -lm
CUDA_LDFLAGS = -lcublas -lgomp
CUDA_ARCH = -gencode arch=compute_89,code=sm_89 # Try to use 4090

C_SRCS_BASE := $(filter-out src/cpu/util.c, $(wildcard src/cpu/*.c))
CU_SRCS_BASE := $(filter-out src/cuda/util.cu, $(wildcard src/cuda/*.cu))

C_OBJS_BASE := $(C_SRCS_BASE:.c=.o)
CU_OBJS_BASE := $(CU_SRCS_BASE:.cu=.o)

UTIL_C_OBJ = src/cpu/util.o
UTIL_CU_OBJ = src/cuda/util.o

test: $(C_OBJS_BASE) $(CU_OBJS_BASE) $(UTIL_CU_OBJ)
	$(NVCC) $(CUDA_ARCH) -Xcompiler="-fopenmp" $^ $(LDFLAGS) $(CUDA_LDFLAGS) -o test

cpu: $(C_OBJS_BASE) $(UTIL_C_OBJ)
	$(CC) $^ $(LDFLAGS) -fopenmp -o test

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(CUDA_ARCH) $(NVFLAGS) -c $< -o $@

clean:
	rm -f $(C_OBJS_BASE) $(CU_OBJS_BASE) $(UTIL_C_OBJ) $(UTIL_CU_OBJ)

# To build CPU only: make cpu
# To build with CUDA: make