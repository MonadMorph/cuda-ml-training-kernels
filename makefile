CC = gcc-15
NVCC = nvcc

# Allow overriding OpenBLAS path via environment variables (useful on different systems)
OPENBLAS ?= /opt/homebrew/opt/openblas
OPENBLAS_INCLUDE ?= $(OPENBLAS)/include
OPENBLAS_LIB ?= $(OPENBLAS)/lib

CFLAGS = -Wall -Wextra -O3 -fopenmp -ffast-math -Iinclude -I$(OPENBLAS_INCLUDE)
NVFLAGS = -O3 --use_fast_math -Xcompiler="-fopenmp -Wall -Wextra -Iinclude"
LDFLAGS = -L$(OPENBLAS_LIB) -lopenblas -lgomp -lpthread -lm
CUDA_LDFLAGS = -lcublas
CUDA_ARCH = -gencode arch=compute_89,code=sm_89 # Try to use a100

CPU_ONLY ?= 0

C_SRCS    := $(wildcard src/cpu/*.c)
CU_SRCS := $(wildcard src/cuda/*.cu)

C_OBJS  := $(C_SRCS:.c=.o)
CU_OBJS := $(CU_SRCS:.cu=.o)

ifeq ($(CPU_ONLY),1)
SRCS = $(C_SRCS)
OBJS = $(C_OBJS)
LINKER = $(CC)
LINKFLAGS = $(LDFLAGS)
else
SRCS = $(C_SRCS) $(CU_SRCS)
OBJS = $(C_OBJS) $(CU_OBJS)
LINKER = $(NVCC)
LINKFLAGS = $(LDFLAGS) $(CUDA_LDFLAGS)
endif

test: $(OBJS)
ifeq ($(CPU_ONLY),1)
	$(LINKER) $^ $(LINKFLAGS) -o $@
else
	$(LINKER) $(CUDA_ARCH) -Xcompiler="-fopenmp" $^ $(LINKFLAGS) -o $@
endif

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(CUDA_ARCH) $(NVFLAGS) -c $< -o $@

clean:
	rm -f $(C_OBJS) $(CU_OBJS)

cpu:
	$(MAKE) CPU_ONLY=1 test

cuda:
	$(MAKE) CPU_ONLY=0 test

# To build CPU only: make cpu
# To build with CUDA: make

# sinteractive --partition=gpu --account=mpcs51087 --gres=gpu:1 --time=00:15:00 --constraint=v100