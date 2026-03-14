#include "main.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

void prep_dataset_cuda(MNISTSet* dataset, int shuffle) {
    if (shuffle == 0) {
        if (dataset->on_device) return;
        float *X_h, *X_d;

        cudaMalloc(&X_d, dataset->N * dataset->data_size * sizeof(float));
        cudaMalloc(&dataset->y_rand, dataset->N * sizeof(int));

        cudaMemcpy(X_d, dataset->X, dataset->N * dataset->data_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dataset->y_rand, dataset->y, dataset->N * sizeof(int), cudaMemcpyHostToDevice);

        X_h = dataset->X;
        dataset->X = X_d;
        free(X_h);
        dataset->on_device = 1;
        return;
    }

    free(dataset->position);
    dataset->position = make_shuffled_index(dataset->N);
    if (dataset->y_rand == NULL) cudaMalloc(&dataset->y_rand, dataset->N * sizeof(int));
    int* new_y = (int*) malloc(dataset->N * sizeof(int));

    for (int i = 0; i < dataset->N; i++) {
        new_y[i] = dataset->y[dataset->position[i]];
    }

    cudaMemcpy(dataset->y_rand, new_y, dataset->N * sizeof(int), cudaMemcpyHostToDevice);
    free(new_y);
}

void pack_batch_data_cuda(Tensor* batch, MNISTSet dataset, int start) {
    int stride = dataset.data_size;
    int batch_size = batch->shape[0];

    float* data_h = (float*) malloc(sizeof(float) * batch_size * stride);
    for (int i = 0; i < batch_size; i++) {
        memcpy(data_h + i * stride, dataset.X + dataset.position[i+start] * stride, stride * sizeof(float));
    }

    cudaMemcpy(batch->data, data_h, sizeof(float) * batch_size * stride, cudaMemcpyHostToDevice);
    free(data_h);
}