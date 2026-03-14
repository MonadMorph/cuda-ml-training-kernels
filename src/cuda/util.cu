#include "main.h"
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

struct timeval timerStart;

void StartTimer(){
    cudaDeviceSynchronize();
    gettimeofday(&timerStart, NULL);
}

float GetTimer(){
    struct timeval timerStop, timerElapsed;
    cudaDeviceSynchronize();
    gettimeofday(&timerStop, NULL);
    timersub(&timerStop, &timerStart, &timerElapsed);

    return (float) timerElapsed.tv_sec+timerElapsed.tv_usec/1e6;
}

Mat_Operators mat_op;
Tensor_Ops t_op;
void (*prep_dataset) (MNISTSet* dataset, int shuffle) = NULL;
void (*pack_batch_data) (Tensor* batch, MNISTSet dataset, int start) = NULL;
int (*greedy_accuracy) (Tensor probs, int* gt) = NULL;

int roll_dice(Tensor probs, int* gt) {
    int num_correct = 0;
    int batch_size = probs.shape[0];
    int prob_space = probs.shape[1];

    for (int i = 0; i < batch_size; i++) {
        float rand_float = rand() / (RAND_MAX + 1.0);
        int j = 0;
        for (; j < prob_space; j++) {
            rand_float -= probs.data[i*prob_space+j];
            if (rand_float <= 0) break;
        }
        if (j >= prob_space) j = prob_space - 1;  // clamp if row sum < 1
        if (j == gt[i]) num_correct++;
    }
    return num_correct;
}

int greedy_accuracy_cpu(Tensor probs, int* gt) {
    int num_correct = 0;
    int batch_size = probs.shape[0];
    int prob_space = probs.shape[1];

    for (int i = 0; i < batch_size; i++) {
        float max = 0.0f; int argmax = 0;
        for (int j = 0; j < prob_space; j++) {
            if (probs.data[i*prob_space+j] > max) {
                max = probs.data[i*prob_space+j];
                argmax = j;
            }
        }
        if (argmax == gt[i]) num_correct++;
    }
    return num_correct;
}

void print_tensor(Tensor t) {
    printf("Shape: [%d", t.shape[0]);
    for (int i = 1; i < t.ndim; i++) {
        printf(", %d", t.shape[i]);
    }
    printf("]\n");

    int total = 1;
    for (int i = 0; i < t.ndim; i++) total *= t.shape[i];

    for (int i = 0; i < total && i < MAX_PRINT; i++) {
        printf("%.3f ", t.data[i]);
    }
    if (total > MAX_PRINT) printf("...");
    printf("\n");
}

int* make_shuffled_index(int N) {
    int* indices = (int*) malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) indices[i] = i;

    for (int i = N-1; i > 0; i--) {
        int j = rand() % (i+1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
    return indices;
}

void set_global_operators(int which) {
    switch (which) {
        case 1:
            mat_op.linear_forward  = linear_layer_forward_tiled;
            mat_op.linear_back = relu_layer_backward_tiled;
            mat_op.update_weight    = update_weight_tiled;
            break;
        case 2:
            mat_op.linear_forward  = linear_layer_forward_blas;
            mat_op.linear_back = relu_layer_backward_blas;
            mat_op.update_weight    = update_weight_blas;
            break;
        case 11:
            mat_op.linear_forward  = linear_layer_forward_cuda;
            mat_op.linear_back = relu_layer_backward_cuda;
            mat_op.update_weight    = update_weight_cuda;
            break;
        case 12:
            mat_op.linear_forward  = linear_layer_forward_cublas;
            mat_op.linear_back = relu_layer_backward_cublas;
            mat_op.update_weight    = update_weight_cublas;
            break;
        default:
            mat_op.linear_forward  = linear_layer_forward;
            mat_op.linear_back = relu_layer_backward;
            mat_op.update_weight    = update_weight;
            break;
    }

    if (which >= 10) {
        t_op.kaimin_init_tensor = kaimin_init_tensor_cuda;
        t_op.zero_init_tensor = zero_init_tensor_cuda;
        t_op.init_tensor = init_tensor_cuda;
        t_op.init_tensor_from = init_tensor_from_cuda;
        t_op.free_tensor = free_tensor_cuda;

        t_op.softmax = softmax_cuda;
        t_op.softmax_backward = softmax_backward_cuda;
        t_op.update_bias = update_bias_cuda;
        t_op.merged_update_bias = merged_update_bias_cuda;
        t_op.cross_entropy_loss = cross_entropy_loss_cuda;

        prep_dataset = prep_dataset_cuda;
        pack_batch_data = pack_batch_data_cuda;
        greedy_accuracy = greedy_accuracy_cuda;
    } else {
        t_op.kaimin_init_tensor = kaimin_init_tensor;
        t_op.zero_init_tensor = zero_init_tensor;
        t_op.init_tensor = init_tensor;
        t_op.init_tensor_from = init_tensor_from;
        t_op.free_tensor = free_tensor;

        t_op.softmax = softmax;
        t_op.softmax_backward = softmax_backward;
        t_op.update_bias = update_bias;
        t_op.merged_update_bias = merged_update_bias;
        t_op.cross_entropy_loss = cross_entropy_loss;

        prep_dataset = prep_dataset_cpu;
        pack_batch_data = pack_batch_data_cpu;
        greedy_accuracy = greedy_accuracy_cpu;
    }
}

__global__ static void greedy_accuracy_d(float* data, int batch_size, int prob_size, int* gt, int* num_correct){
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;

    for (int i = thread_id; i < batch_size; i += total_threads) {
        float max = data[i * prob_size];
        int argmax = 0;

        for (int j = 1; j < prob_size; j++) {
            float cur = data[i * prob_size + j];
            if (cur > max) {
                max = cur;
                argmax = j;
            }
        }
        if (argmax == gt[i]) atomicAdd(num_correct, 1);
    }
}

int greedy_accuracy_cuda(Tensor probs, int* gt){
    int task_size = probs.shape[0];
    int prob_space = probs.shape[1];

    int tpb = (task_size >= 256) ? 256 : 32;
    int blocks = (task_size + tpb - 1) / tpb;

    int *num_correct = NULL;
    cudaMalloc((void**)&num_correct, sizeof(int));
    cudaMemset(num_correct, 0, sizeof(int));

    greedy_accuracy_d<<<blocks, tpb>>>(probs.data, task_size, prob_space, gt, num_correct);

    int num_correct_h = 0;
    cudaMemcpy(&num_correct_h, num_correct, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(num_correct);

    return num_correct_h;
}

void print_tensor_cuda(Tensor t) { // debugging only
    float* data_h = (float *) malloc(sizeof(float) * t.size);
    cudaMemcpy(data_h, t.data, sizeof(float) * t.size, cudaMemcpyDeviceToHost);

    printf("Tensor has dimensions [%d, %d], with size %d\n", t.shape[0], t.shape[1], t.size);
    int print_count = t.size > MAX_PRINT ? MAX_PRINT : t.size;
    printf("The first few entries are:\n");
    for (int i = 0; i < print_count; i++) {
        printf("%.6f   ", data_h[i]);
    }
    printf("\n\n");
    free(data_h);
}