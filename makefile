CC = gcc
NVCC = nvcc

# Allow overriding OpenBLAS path via environment variables (useful on different systems)
OPENBLAS_INCLUDE ?= /usr/include/x86_64-linux-gnu
OPENBLAS_LIB ?= /usr/lib/x86_64-linux-gnu

CFLAGS = -Wall -Wextra -O3 -fopenmp -ffast-math -Iinclude -I$(OPENBLAS_INCLUDE)
NVFLAGS = -O3 --use_fast_math -Xcompiler="-fopenmp -Wall -Wextra -Iinclude"
LDFLAGS = -L$(OPENBLAS_LIB) -lopenblas -lgomp -lpthread -lm
CUDA_LDFLAGS = -lcublas
CUDA_ARCH = -gencode arch=compute_89,code=sm_89 # Try to use a100

C_SRCS_BASE := $(filter-out src/cpu/util.c, $(wildcard src/cpu/*.c))
CU_SRCS_BASE := $(filter-out src/cuda/util.cu, $(wildcard src/cuda/*.cu))

C_OBJS_BASE := $(C_SRCS_BASE:.c=.o)
CU_OBJS_BASE := $(CU_SRCS_BASE:.cu=.o)

UTIL_C_OBJ = src/cpu/util.o
UTIL_CU_OBJ = src/cuda/util.o

test: $(C_OBJS_BASE) $(CU_OBJS_BASE) $(UTIL_CU_OBJ)
	$(NVCC) $(CUDA_ARCH) -Xcompiler="-fopenmp" $^ $(LDFLAGS) $(CUDA_LDFLAGS) -o test

cpu: $(C_OBJS_BASE) $(UTIL_C_OBJ)
	$(CC) $^ $(LDFLAGS) -o test

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(CUDA_ARCH) $(NVFLAGS) -c $< -o $@

clean:
	rm -f $(C_OBJS_BASE) $(CU_OBJS_BASE) $(UTIL_C_OBJ) $(UTIL_CU_OBJ)

# To build CPU only: make cpu
# To build with CUDA: make

# sinteractive --partition=gpu --account=mpcs51087 --gres=gpu:1 --time=00:15:00 --constraint=v100