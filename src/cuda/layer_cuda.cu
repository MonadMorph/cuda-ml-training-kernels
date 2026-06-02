#include "layer.h"
#include <math.h>
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
#define TILE 16
#define NUM_STREAMS 3

typedef struct {
    int initialized;
    int work_on;
    cudaStream_t weight_streams[NUM_STREAMS];
} Cuda_streams;

static Cuda_streams streams = (Cuda_streams) {.initialized = 0};

static void inline init_streams(void) {
    if (!streams.initialized) {
        for (int i = 0; i < NUM_STREAMS; i++) {
            CHECK(cudaStreamCreate(&streams.weight_streams[i]));
        }
        streams.initialized = 1;
        streams.work_on = 0;
    }
}

// I don't want to do the mental gymnastic of dealing with transpose. And I wish to integrate relu in as well. So 3 kernel functions!
__global__ static void linear_layer_forward_d(float* __restrict__ A, float* __restrict__ W, float* __restrict__ b, float* __restrict__ C, int I, int J, int K, int do_relu) {
    __shared__ float As[TILE][TILE+1];
    __shared__ float Ws[TILE][TILE+1]; // pad to reduce bank conflict

    // In tile index
    int tx = threadIdx.x;   // Directly index in 2 dim!
    int ty = threadIdx.y;   // 0..15
    // Global index
    int row = blockIdx.y * TILE + ty; // i
    int col   = blockIdx.x * TILE + tx; // output j
    int w_row = blockIdx.x * TILE + ty; // cooperative W load row

    float acc = 0.0f;

    // loop over K tiles
    for (int k0 = 0; k0 < K; k0 += TILE) {
        int k = k0 + tx;

        // Load A tile: A[row, k0+tx]
        if (row < I && k < K) {
            As[tx][ty] = A[row * K + k];
        } else As[tx][ty] = 0.0f; // switch order to avoid bank conflict

        // Load W tile: W[w_row, k0+tx]   
        if (w_row < J && k < K) {
            Ws[tx][ty] = W[w_row * K + k]; // I will load As and Ws in the good way, so we can iterate over
        } else Ws[tx][ty] = 0.0f;
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            acc += As[kk][ty] * Ws[kk][tx]; // few bank conflict
        }
        __syncthreads();
    }

    if (row < I && col < J) {
        acc += b[col];
        C[row * J + col] = do_relu ? (acc > 0.0f ? acc : 0.0f) : acc;
    }
}

void linear_layer_forward_cuda (Tensor input, Tensor weight, Tensor bias, Tensor* output, int do_relu) {
    // shapes
    const int I = input.shape[0];      // input outer dim (batch)
    const int J = weight.shape[0];     // weight outer dim (out)
    const int K = input.shape[1];      // "hot dimension" (in)

    dim3 block(TILE, TILE); // ptb
    dim3 grid((J + TILE - 1) / TILE, (I + TILE - 1) / TILE); // nblocks

    linear_layer_forward_d<<<grid, block>>>(input.data, weight.data, bias.data, output->data, I, J, K, do_relu);
    CHECK(cudaGetLastError());
}

__global__ static void relu_layer_backward_d(float* __restrict__ A, float* __restrict__ W, float* __restrict__ Logits, float* __restrict__ C, int I, int J, int K) {
    __shared__ float As[TILE][TILE+1];
    __shared__ float Ws[TILE][TILE+1];

    // In tile index
    int tx = threadIdx.x;   // Directly index in 2 dim!
    int ty = threadIdx.y;   // 0..15
    // Global index
    int row = blockIdx.y * TILE + ty; // i
    int col = blockIdx.x * TILE + tx; // j

    float acc = 0.0f;

    // loop over K tiles
    for (int k0 = 0; k0 < K; k0 += TILE) {
        // Load A tile: A[row, k0+tx]
        if (row < I && (k0 + tx) < K) {
            As[tx][ty] = A[row * K + (k0 + tx)];
        } else As[tx][ty] = 0.0f;

        // Load W tile: W[col, k0+ty]   
        if (col < J && (k0 + ty) < K) {
            Ws[ty][tx] = W[(k0 + ty) * J + col]; // I will load As and Ws in the good way, so we can iterate over
        } else Ws[ty][tx] = 0.0f;
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            acc += As[kk][ty] * Ws[kk][tx];
        }
        __syncthreads();
    }

    // Hadamard by z (or a) thing
    if (row < I && col < J) C[row * J + col] = Logits[row * J + col] > 0.0f ? acc : 0.0f;
}

void relu_layer_backward_cuda(Tensor input, Tensor weight, Tensor cur_logits, Tensor* output) {
    // shapes
    const int I = input.shape[0];      // input outer dim (batch)
    const int J = weight.shape[1];     // weight outer dim (in)
    const int K = input.shape[1];      // "hot dimension" (out)

    dim3 block(TILE, TILE); // ptb
    dim3 grid((J + TILE - 1) / TILE, (I + TILE - 1) / TILE); // nblocks

    relu_layer_backward_d<<<grid, block>>>(input.data, weight.data, cur_logits.data, output->data, I, J, K);
    CHECK(cudaGetLastError());
}

__global__ static void update_weight_d(float* __restrict__ A, float* __restrict__ W, float* __restrict__ C, float factor, int I, int J, int K) {
    __shared__ float As[TILE][TILE+1];
    __shared__ float Ws[TILE][TILE+1];

    // In tile index
    int tx = threadIdx.x;   // Directly index in 2 dim!
    int ty = threadIdx.y;   // 0..15
    // Global index
    int row = blockIdx.y * TILE + ty; // i
    int col = blockIdx.x * TILE + tx; // j
    int a_col = blockIdx.y * TILE + tx;

    float acc = 0.0f;

    // loop over K tiles
    for (int k0 = 0; k0 < K; k0 += TILE) {
        // Load A tile: A[k0+ty, a_col]
        if (a_col < I && (k0 + ty) < K) {
            As[ty][tx] = A[(k0 + ty) * I + a_col];
        } else As[ty][tx] = 0.0f;

        // Load W tile: W[col, k0+ty]   
        if (col < J && (k0 + ty) < K) {
            Ws[ty][tx] = W[(k0 + ty) * J + col]; // I will load As and Ws in the good way, so we can iterate over
        } else Ws[ty][tx] = 0.0f;
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            acc += As[kk][ty] * Ws[kk][tx];
        }
        __syncthreads();
    }

    // Hadamard by z (or a) thing
    if (row < I && col < J) C[row * J + col] -= factor * acc;
}

void update_weight_cuda(Tensor input, Tensor weight, float lr, Tensor* output) {
    // shapes
    // input is D_i, weight is A_{i-1} and out is W_i
    const int I = input.shape[1];      // input outer dim (out)
    const int J = weight.shape[1];     // weight outer dim (in)
    const int K = input.shape[0];      // "hot dimension" (batch)

    dim3 block(TILE, TILE); // ptb
    dim3 grid((J + TILE - 1) / TILE, (I + TILE - 1) / TILE); // nblocks
    float factor = lr / K;

    init_streams();
    int work_on = streams.work_on;
    streams.work_on = (work_on + 1) % NUM_STREAMS;
    update_weight_d<<<grid, block, 0, streams.weight_streams[work_on]>>>(input.data, weight.data, output->data, factor, I, J, K);
    CHECK(cudaGetLastError());
}

void destory_weight_sync_cuda (void) {
    if (streams.initialized) {
        for (int i = 0; i < NUM_STREAMS; i++) {
            CHECK(cudaStreamDestroy(streams.weight_streams[i]));
        }
        streams.initialized = 0;
    }
}