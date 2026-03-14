#include "tensor.h"
#include <math.h>
#include <cuda_runtime.h>

// In place softmax
__global__ static void softmax_d(float* data, int shape0, int shape1) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;

    for (int i = thread_id; i < shape0; i += total_threads) {
        float max = data[i*shape1];
        for (int j = 1; j < shape1; j++) {
            max = data[i*shape1 + j] > max ? data[i*shape1 + j] : max;
        }
        float sum = 0;
        for (int j = 0; j < shape1; j++) {
            data[i*shape1 + j] = exp(data[i*shape1 + j] - max);
            sum += data[i*shape1 + j];
        }
        for (int j = 0; j < shape1; j++) {
            data[i*shape1 + j] /= sum;
        }
    }
}

void softmax_cuda(Tensor* logits) {
    int tpb, blocks;
    int task_size = logits->shape[0];
    tpb = task_size >= 256 ? 256 : 32;
    blocks = (task_size + tpb - 1) / tpb;

    softmax_d<<<blocks, tpb>>>(logits->data, task_size, logits->shape[1]);
}

__global__ static void softmax_backward_d(float* data, int batch_size, int prob_space, int* gt) {
    // gt need to be on device
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;

    for (int i = thread_id; i < batch_size; i += total_threads) {
        data[i * prob_space + gt[i]] -= 1;
    }
}

void softmax_backward_cuda(Tensor* error, int* gt) {    // gt need to be on device
    int tpb, blocks;
    int task_size = error->shape[0];
    tpb = task_size >= 256 ? 256 : 32;
    blocks = (task_size + tpb - 1) / tpb;

    softmax_backward_d<<<blocks, tpb>>>(error->data, task_size, error->shape[1], gt); 
}

__global__ static void update_bias_d(float* bias_data, int param_size, float* error_data, int batch_size, float factor) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;

    for (int i = thread_id; i < param_size; i += total_threads) {
        float acc = 0;
        for (int j = 0; j < batch_size; j++) {
            acc += error_data[j*param_size+i];
        }
        bias_data[i] -= acc * factor;
    }
}

void update_bias_cuda(Tensor error, float lr, Tensor* bias) {
    int tpb, blocks;
    int task_size = bias->shape[1];
    tpb = task_size >= 256 ? 256 : 32;
    blocks = (task_size + tpb - 1) / tpb;

    int batch_size = error.shape[0];
    float factor = lr / batch_size;

    update_bias_d<<<blocks, tpb>>>(bias->data, task_size, error.data, batch_size, factor);
}

__global__ static void merged_update_bias_d(float* bias_data, float* D1_data, int b1_size, float* D2_data, int b2_size, float* D3_data, int b3_size, int batch_size, float factor) {

    int D2_threshold = b1_size;
    int D3_threshold = D2_threshold + b2_size;
    int total = D3_threshold + b3_size;
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;

    for (int i = thread_id; i < total; i += total_threads) {
        if (i < D3_threshold) {
            if (i < D2_threshold) {
                // D1 code
                int stride = b1_size;
                for (int j = 0; j < batch_size; j++) {
                    bias_data[i] += D1_data[j*stride+i] * factor;
                }
            } else {
                // D2 code
                int stride = b2_size;
                int local_i = i - D2_threshold;
                for (int j = 0; j < batch_size; j++) {
                    bias_data[i] += D2_data[j*stride+local_i] * factor;
                }
            }
        } else {
            // D3 code
            int stride = b3_size;
            int local_i = i - D3_threshold;
            for (int j = 0; j < batch_size; j++) {
                bias_data[i] += D3_data[j*stride+local_i] * factor;
            }
        }
    }
}

void merged_update_bias_cuda(Tensor D1, Tensor D2, Tensor D3, float lr, Tensor* bias) {

    int batch_size = D1.shape[0];
    int b1_size = D1.shape[1];
    int b2_size = D2.shape[1];
    int b3_size = D3.shape[1];
    float factor = - lr / batch_size;

    int tpb, blocks;
    int task_size = b1_size + b2_size + b3_size;
    tpb = task_size >= 256 ? 256 : 32;
    blocks = (task_size + tpb - 1) / tpb;

    merged_update_bias_d<<<blocks, tpb>>>(bias->data, D1.data, b1_size, D2.data, b2_size, D3.data, b3_size, batch_size, factor);
}

__global__ static void cross_entropy_loss_d(float* data, int batch_size, int param_size, int* gt, float* loss_d) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;

    for (int i = thread_id; i < batch_size; i += total_threads) {
        atomicAdd(loss_d, -logf(data[i * param_size + gt[i]]));
    }
}

float cross_entropy_loss_cuda(Tensor probs, int* gt) {
    int tpb, blocks;
    int task_size = probs.shape[0];
    tpb = task_size >= 256 ? 256 : 32;
    blocks = (task_size + tpb - 1) / tpb;

    int param_size = probs.shape[1];
    float *loss_d; // This is so stupid
    cudaMalloc(&loss_d, sizeof(float));
    cudaMemset(loss_d, 0, sizeof(float));

    cross_entropy_loss_d<<<blocks, tpb>>>(probs.data, task_size, param_size, gt, loss_d);

    float loss_h = 0.0f;
    cudaMemcpy(&loss_h, loss_d, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(loss_d);

    return loss_h;
}