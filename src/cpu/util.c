#include "main.h"
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>

int MAX_PRINT = 50;

struct timeval timerStart;

void StartTimer(){
    gettimeofday(&timerStart, NULL);
}

float GetTimer(){
    struct timeval timerStop, timerElapsed;
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
        default:
            mat_op.linear_forward  = linear_layer_forward;
            mat_op.linear_back = relu_layer_backward;
            mat_op.update_weight    = update_weight;
            break;
    }

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