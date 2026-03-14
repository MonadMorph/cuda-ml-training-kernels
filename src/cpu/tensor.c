#include "tensor.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

void kaimin_init_tensor(Tensor* tensor) {
    int size = tensor->size;
    tensor->data = (float*) malloc(sizeof(float) * size);
    float* data = tensor->data;
    int in_dim = tensor->shape[1];
    float rand_bound = 6.0f / sqrtf(in_dim);

    for (int i = 0; i < size; i++) {
        data[i] = 2 * rand_bound * rand() / (RAND_MAX + 1.0)  - rand_bound;
    }
}

void zero_init_tensor(Tensor* tensor) {
    int size = tensor->size;
    tensor->data = (float*) calloc(size, sizeof(float));
}

void init_tensor(Tensor* tensor) {
    int size = tensor->size;
    tensor->data = (float*) malloc(sizeof(float) * size);
}

void init_tensor_from(Tensor* tensor, Tensor basis) {
    int size = tensor->size;
    tensor->data = (float*) malloc(sizeof(float) * size);
    for (int i = 0; i < size; i++) {
        tensor->data[i] = basis.data[i % basis.size];
    }
}

void free_tensor(Tensor* tensor) {
    free(tensor->data);
}