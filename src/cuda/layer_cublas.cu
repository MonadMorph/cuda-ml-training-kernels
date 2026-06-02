#include "layer.h"
#include <stdio.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define NUM_STREAMS 3
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

#define CHECK_CUBLAS(call)                                      \
{                                                               \
    const cublasStatus_t status = call;                         \
    if (status != CUBLAS_STATUS_SUCCESS) {                      \
        printf("cuBLAS Error: %s:%d, code:%d\n",                \
               __FILE__, __LINE__, status);                     \
        exit(1);                                                \
    }                                                           \
}

/*cublasSgemm(handle, transa, transb, I, J, K,
            &alpha,
            A, lda,
            B, ldb,
            &beta,
            C, ldc);*/

typedef struct {
    int initialized;
    int work_on;
    cublasHandle_t null_handle;
    cudaStream_t weight_streams[NUM_STREAMS];
    cublasHandle_t w_handles[NUM_STREAMS];
} Cublas_handles;

static Cublas_handles handles = (Cublas_handles) {.initialized = 0};

static void inline init_handles(void) {
    if (!handles.initialized) {
        for (int i = 0; i < NUM_STREAMS; i++) {
            CHECK(cudaStreamCreate(&handles.weight_streams[i]));
            CHECK_CUBLAS(cublasCreate(&handles.w_handles[i]));
            CHECK_CUBLAS(cublasSetStream(handles.w_handles[i], handles.weight_streams[i]));
        }
        CHECK_CUBLAS(cublasCreate(&handles.null_handle));
        handles.initialized = 1;
        handles.work_on = 0;
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

    // This is always true
    const int out_stride = output->shape[1];
    const int in_stride  = input.shape[1];
    const int weight_stride = weight.shape[1];

    float alpha = 1.0f;
    float beta  = 0.0f;

    init_handles();
    CHECK_CUBLAS(cublasSgemm(handles.null_handle, CUBLAS_OP_T, CUBLAS_OP_N, J, I, K,
        &alpha, weight.data, weight_stride,
        input.data,  in_stride,
        &beta, output->data, out_stride));

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

    // This is always true
    const int out_stride = output->shape[1];
    const int in_stride  = input.shape[1];
    const int weight_stride = weight.shape[1];

    float alpha = 1.0f;
    float beta  = 0.0f;

    init_handles();
    CHECK_CUBLAS(cublasSgemm(handles.null_handle, CUBLAS_OP_N, CUBLAS_OP_N, J, I, K,
        &alpha, weight.data, weight_stride,
        input.data,  in_stride,
        &beta, output->data, out_stride));

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

    init_handles();
    int work_on = handles.work_on;
    handles.work_on = (work_on + 1) % NUM_STREAMS;
    CHECK_CUBLAS(cublasSgemm(handles.w_handles[work_on], CUBLAS_OP_N, CUBLAS_OP_T, J, I, K,
        &alpha, weight.data, weight_stride,
        input.data,  in_stride,
        &beta, output->data, out_stride));
}

void destory_weight_sync_cublas (void) {
    if (handles.initialized) {
        for (int i = 0; i < NUM_STREAMS; i++) {
            CHECK_CUBLAS(cublasDestroy(handles.w_handles[i]));
            CHECK(cudaStreamDestroy(handles.weight_streams[i]));
        }
        CHECK_CUBLAS(cublasDestroy(handles.null_handle));
        handles.initialized = 0;
    }
}