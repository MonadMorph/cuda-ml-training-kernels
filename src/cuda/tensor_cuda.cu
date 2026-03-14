#include "tensor.h"
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>


void kaimin_init_tensor_cuda(Tensor* tensor) {
    int size = tensor->size;
    float* data_h = (float*) malloc(sizeof(float) * size); 
    int in_dim = tensor->shape[1];
    float rand_bound = 6.0f / sqrtf(in_dim);

    // I don't want to touch curand for now
    for (int i = 0; i < size; i++) {
        data_h[i] = 2 * rand_bound * rand() / (RAND_MAX + 1.0)  - rand_bound;
    }

    cudaMalloc(&tensor->data, size * sizeof(float));
    cudaMemcpy(tensor->data, data_h, size * sizeof(float), cudaMemcpyHostToDevice);
    free(data_h);
}

void zero_init_tensor_cuda(Tensor* tensor) {
    int size = tensor->size;
    cudaMalloc(&tensor->data, size * sizeof(float));
    cudaMemset(tensor->data, 0, size * sizeof(float));
}

void init_tensor_cuda(Tensor* tensor) {
    int size = tensor->size;
    cudaMalloc(&tensor->data, size * sizeof(float));
}

__global__ void init_tensor_from_d(float* tensor_data, int tensor_size, float* basis_data, int basis_size) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;

    for (int i = thread_id; i < tensor_size; i += total_threads) {
        tensor_data[i] = basis_data[i % basis_size];
    }
}

void init_tensor_from_cuda(Tensor* tensor, Tensor basis) {
    int size = tensor->size;
    cudaMalloc(&tensor->data, size * sizeof(float));
    
    int tpb, blocks;
    tpb = size >= 256 ? 256 : 32;
    blocks = (size + tpb - 1) / tpb;

    init_tensor_from_d<<<blocks, tpb>>>(tensor->data, size, basis.data, basis.size);
}

void free_tensor_cuda(Tensor* tensor) {
    cudaFree(tensor->data);
}