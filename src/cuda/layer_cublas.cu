#include "layer.h"
#include <stdio.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*cublasSgemm(handle, transa, transb, I, J, K,
            &alpha,
            A, lda,
            B, ldb,
            &beta,
            C, ldc);*/

static cublasHandle_t handle;
static int handle_ready = 0;

static void inline ensure_cublas_handle(void) {
    if (!handle_ready) {
        cublasStatus_t stat = cublasCreate(&handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            printf("cublasCreate failed\n");
        }
        handle_ready = 1;
    }
}

__global__ static void add_bias_relu_d(float* data, float* bias, int I, int J, int do_relu){
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;

    for (int index = thread_id; index < I*J; index += total_threads) {
        float acc = data[index] + bias[index%J];
        data[index] = do_relu ? (acc > 0 ? acc : 0) : acc;
    }
}

void linear_layer_forward_cublas (Tensor input, Tensor weight, Tensor bias, Tensor* output, int do_relu) {
    // shapes
    const int I = input.shape[0];      // input outer dim (batch)
    const int J = weight.shape[0];     // weight outer dim (out)
    const int K = input.shape[1];      // "hot dimension" (in)

    *output = (Tensor){.data = NULL, .ndim = 2, .shape = {I, J}, .size = I * J};
    init_tensor_cuda(output);

    // This is always true
    const int out_stride = output->shape[1];
    const int in_stride  = input.shape[1];
    const int weight_stride = weight.shape[1];

    float alpha = 1.0f;
    float beta  = 0.0f;

    ensure_cublas_handle();
    cublasStatus_t stat = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, J, I, K,
        &alpha, weight.data, weight_stride,
        input.data,  in_stride,
        &beta, output->data, out_stride);

    // To be removed
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("cublasSgemm failed in linear_layer_forward_cublas\n");
    }

    // Separate kernel for bias + optional relu
    int tpb = 256;
    int blocks = (output->size + tpb - 1) / tpb;
    add_bias_relu_d<<<blocks, tpb>>>(output->data, bias.data, I, J, do_relu);
}

__global__ static void relu_derivative_d(float* out_data, float* logits, int size){
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;

    for (int index = thread_id; index < size; index += total_threads) {
        out_data[index] = logits[index] > 0 ? out_data[index] : 0;
    }
}

void relu_layer_backward_cublas(Tensor input, Tensor weight, Tensor cur_logits, Tensor* output) {
    // shapes
    const int I = input.shape[0];      // input outer dim (batch)
    const int J = weight.shape[1];     // weight outer dim (in)
    const int K = input.shape[1];      // "hot dimension" (out)

    *output = (Tensor){.data = NULL, .ndim = 2, .shape = {I, J}, .size = I * J};
    init_tensor_cuda(output);

    // This is always true
    const int out_stride = output->shape[1];
    const int in_stride  = input.shape[1];
    const int weight_stride = weight.shape[1];

    float alpha = 1.0f;
    float beta  = 0.0f;

    ensure_cublas_handle();
    cublasStatus_t stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, J, I, K,
        &alpha, weight.data, weight_stride,
        input.data,  in_stride,
        &beta, output->data, out_stride);

    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("cublasSgemm failed in relu_layer_backward_cublas\n");
    }

    // Separate kernel for relu derivative
    int tpb = 256;
    int blocks = (output->size + tpb - 1) / tpb;
    relu_derivative_d<<<blocks, tpb>>>(output->data, cur_logits.data, output->size);
}

void update_weight_cublas(Tensor input, Tensor weight, float lr, Tensor* output) {
    // shapes
    // input is D_i, weight is A_{i-1} and out is W_i
    const int I = input.shape[1];      // input outer dim (out)
    const int J = weight.shape[1];     // weight outer dim (in)
    const int K = input.shape[0];      // "hot dimension" (batch)

    // This is always true
    const int out_stride = output->shape[1];
    const int in_stride  = input.shape[1];
    const int weight_stride = weight.shape[1];

    float alpha = -lr / K;
    float beta  = 1.0f;

    ensure_cublas_handle();
    cublasStatus_t stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, J, I, K,
        &alpha, weight.data, weight_stride,
        input.data,  in_stride,
        &beta, output->data, out_stride);

    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("cublasSgemm failed in update_weight_cublas\n");
    }
}