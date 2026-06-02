#include "main.h"
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
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

struct timeval timerStart;

void StartTimer(){
    CHECK(cudaDeviceSynchronize());
    gettimeofday(&timerStart, NULL);
}

float GetTimer(){
    struct timeval timerStop, timerElapsed;
    CHECK(cudaDeviceSynchronize());
    gettimeofday(&timerStop, NULL);
    timersub(&timerStop, &timerStart, &timerElapsed);

    return (float) timerElapsed.tv_sec+timerElapsed.tv_usec/1e6;
}

Mat_Operators mat_op;
Tensor_Ops t_op;
Dataset_Operators data_ops;
void (*greedy_accuracy) (Tensor probs, int* gt) = NULL;
int (*get_accuracy) (void);

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

static int acc_initialized = 0;
static int* agg_acc_d;
static int agg_acc;

void greedy_accuracy_cpu(Tensor probs, int* gt) {
    int num_correct = 0;
    int batch_size = probs.shape[0];
    int prob_space = probs.shape[1];

    if (!acc_initialized) {
        agg_acc = 0;
        acc_initialized = 1;
    }

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
    agg_acc += num_correct;
}

int get_accuracy_cpu(void) {
    acc_initialized = 0;
    return agg_acc;
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
            mat_op.destory_weight_sync = destory_weight_sync_cuda;
            break;
        case 12:
            mat_op.linear_forward  = linear_layer_forward_cublas;
            mat_op.linear_back = relu_layer_backward_cublas;
            mat_op.update_weight    = update_weight_cublas;
            mat_op.destory_weight_sync = destory_weight_sync_cublas;
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
        t_op.destory_bias_sync = destory_bias_sync_cuda;

        t_op.softmax = softmax_cuda;
        t_op.softmax_backward = softmax_backward_cuda;
        t_op.update_bias = update_bias_cuda;
        t_op.merged_update_bias = merged_update_bias_cuda;
        t_op.cross_entropy_loss = cross_entropy_loss_cuda;
        t_op.get_loss = get_loss_cuda;

        data_ops.prep_dataset = prep_dataset_cuda;
        data_ops.pack_batch_data = pack_batch_data_cuda;
        data_ops.fetch_data = fetch_data_cuda;
        data_ops.destory_data = destory_data_cuda;
        greedy_accuracy = greedy_accuracy_cuda;
        get_accuracy = get_accuracy_cuda;
    } else {
        t_op.kaimin_init_tensor = kaimin_init_tensor;
        t_op.zero_init_tensor = zero_init_tensor;
        t_op.init_tensor = init_tensor;
        t_op.init_tensor_from = init_tensor_from;
        t_op.free_tensor = free_tensor;

        mat_op.destory_weight_sync = destory_weight_sync_cpu;
        t_op.destory_bias_sync = destory_bias_sync_cpu;

        t_op.softmax = softmax;
        t_op.softmax_backward = softmax_backward;
        t_op.update_bias = update_bias;
        t_op.merged_update_bias = merged_update_bias;
        t_op.cross_entropy_loss = cross_entropy_loss;
        t_op.get_loss = get_loss_cpu;

        data_ops.prep_dataset = prep_dataset_cpu;
        data_ops.pack_batch_data = pack_batch_data_cpu;
        data_ops.fetch_data = fetch_data_cpu;
        data_ops.destory_data = destory_data_cpu;
        greedy_accuracy = greedy_accuracy_cpu;
        get_accuracy = get_accuracy_cpu;
    }
}

__global__ static void greedy_accuracy_d(float* data, int batch_size, int prob_size, int* gt, int* num_correct){
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    bool correct = 0;
    __shared__ int num_correct_local;

    if (threadIdx.x == 0) num_correct_local = 0;
    __syncthreads();

    if (thread_id < batch_size) {
        float max = data[thread_id * prob_size];
        int argmax = 0;

        for (int j = 1; j < prob_size; j++) {
            float cur = data[thread_id * prob_size + j];
            if (cur > max) {
                max = cur;
                argmax = j;
            }
        }
        if (argmax == gt[thread_id]) correct = 1;
    }

    // Use warp level preditive to do reduction
    unsigned mask = __ballot_sync(0xffffffff, correct);
    int count = __popc(mask);

    if (thread_id % 32 == 0) atomicAdd(&num_correct_local, count);
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(num_correct, num_correct_local);
}

void greedy_accuracy_cuda(Tensor probs, int* gt){
    int task_size = probs.shape[0];
    int prob_space = probs.shape[1];

    int tpb = (task_size >= 256) ? 256 : 32;
    int blocks = (task_size + tpb - 1) / tpb;

    if (!acc_initialized) {
        CHECK(cudaMalloc((void**)&agg_acc_d, sizeof(int)));
        CHECK(cudaMemset(agg_acc_d, 0, sizeof(int)));
        acc_initialized = 1;
    }

    greedy_accuracy_d<<<blocks, tpb>>>(probs.data, task_size, prob_space, gt, agg_acc_d);
    CHECK(cudaGetLastError());
}

int get_accuracy_cuda(void){
    int num_correct_h = 0;
    CHECK(cudaMemcpy(&num_correct_h, agg_acc_d, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(agg_acc_d));
    acc_initialized = 0;

    return num_correct_h;
}

void print_tensor_cuda(Tensor t) { // debugging only
    float* data_h = (float *) malloc(sizeof(float) * t.size);
    CHECK(cudaMemcpy(data_h, t.data, sizeof(float) * t.size, cudaMemcpyDeviceToHost));

    printf("Tensor has dimensions [%d, %d], with size %d\n", t.shape[0], t.shape[1], t.size);
    int print_count = t.size > MAX_PRINT ? MAX_PRINT : t.size;
    printf("The first few entries are:\n");
    for (int i = 0; i < print_count; i++) {
        printf("%.6f   ", data_h[i]);
    }
    printf("\n\n");
    free(data_h);
}