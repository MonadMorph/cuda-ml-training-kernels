#include "main.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define CHECK(call)                                                    \
{                                                                           \
   const cudaError_t error = call;                                          \
   if (error != cudaSuccess)                                                \
   {                                                                        \
       printf("Error: %s:%d, ", __FILE__, __LINE__);                        \
       printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));   \
       exit(1);                                                             \
   }                                                                        \
}

typedef struct {
    cudaStream_t copy_stream;
    cudaEvent_t batch_ready;
    float* transfer_buffer;
    float *data_d[2];
    int initialized;
    int work_on;
} Data_stream;

static Data_stream data_stream = (Data_stream) {.initialized = 0, .work_on = 0};

void prep_dataset_cuda(MNISTSet* dataset, int shuffle) {
    if (shuffle == 0) {
        if (dataset->on_device) return;
        float *X_h, *X_d;

        CHECK(cudaMalloc(&X_d, dataset->N * dataset->data_size * sizeof(float)));
        CHECK(cudaMalloc(&dataset->y_rand, dataset->N * sizeof(int)));

        CHECK(cudaMemcpy(X_d, dataset->X, dataset->N * dataset->data_size * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dataset->y_rand, dataset->y, dataset->N * sizeof(int), cudaMemcpyHostToDevice));

        X_h = dataset->X;
        dataset->X = X_d;
        free(X_h);
        dataset->on_device = 1;
        return;
    }

    if (!data_stream.initialized) {
        size_t batch_size_byte = sizeof(float) * dataset->batch_size * dataset->data_size;
        CHECK(cudaStreamCreateWithFlags(&data_stream.copy_stream, cudaStreamNonBlocking)); // Kernel still on Null stream, so non blocking
        CHECK(cudaEventCreate(&data_stream.batch_ready));
        CHECK(cudaMallocHost((void**)&data_stream.transfer_buffer, batch_size_byte));
        CHECK(cudaMalloc((void**)&data_stream.data_d[0], batch_size_byte));
        CHECK(cudaMalloc((void**)&data_stream.data_d[1], batch_size_byte));
        data_stream.initialized = 1;
    }

    free(dataset->position);
    dataset->position = make_shuffled_index(dataset->N);
    if (dataset->y_rand == NULL) CHECK(cudaMalloc(&dataset->y_rand, dataset->N * sizeof(int)));
    int* new_y;
    CHECK(cudaMallocHost((void**)&new_y, dataset->N * sizeof(int)));

    for (int i = 0; i < dataset->N; i++) {
        new_y[i] = dataset->y[dataset->position[i]];
    }

    CHECK(cudaMemcpy(dataset->y_rand, new_y, dataset->N * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaFreeHost(new_y));
}

void fetch_data_cuda(Tensor* batch, MNISTSet dataset, int start) {
    int stride = dataset.data_size;
    int batch_size = batch->shape[0];

    for (int i = 0; i < batch_size; i++) {
        memcpy(data_stream.transfer_buffer + i * stride, dataset.X + dataset.position[i+start] * stride, stride * sizeof(float));
    }

    CHECK(cudaMemcpyAsync(data_stream.data_d[data_stream.work_on], data_stream.transfer_buffer, sizeof(float) * batch_size * stride, cudaMemcpyHostToDevice, data_stream.copy_stream));
    CHECK(cudaEventRecord(data_stream.batch_ready, data_stream.copy_stream)); 
}

void pack_batch_data_cuda(Tensor* batch, MNISTSet dataset, int start) {
    (void)dataset; // Unused
    (void)start;

    CHECK(cudaEventSynchronize(data_stream.batch_ready));
    batch->data = data_stream.data_d[data_stream.work_on];
    data_stream.work_on ^= 1; // Switch to the other slot (for next fetching)
}

void destory_data_cuda(MNISTSet dataset) {
    if (data_stream.initialized) {
        CHECK(cudaStreamDestroy(data_stream.copy_stream));
        CHECK(cudaEventDestroy(data_stream.batch_ready));
        CHECK(cudaFreeHost(data_stream.transfer_buffer));
        CHECK(cudaFree(data_stream.data_d[0]));
        CHECK(cudaFree(data_stream.data_d[1]));
        data_stream.initialized = 0;
    }

    if (dataset.on_device) {
        CHECK(cudaFree(dataset.X));
    } else {
        free(dataset.X);
    }

    if (dataset.y_rand) CHECK(cudaFree(dataset.y_rand));
    if (dataset.position) free(dataset.position);
    free(dataset.y);
}