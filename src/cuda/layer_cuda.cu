#include "layer.h"
#include <math.h>
#include <cuda_runtime.h>

#define TILE 16

// I don't want to do the mental gymnastic of dealing with transpose. And I wish to integrate relu in as well. So 3 kernel functions!
__global__ static void linear_layer_forward_d(float* __restrict__ A, float* __restrict__ W, float* __restrict__ b, float* __restrict__ C, int I, int J, int K, int do_relu) {
    __shared__ float As[TILE][TILE];
    __shared__ float Ws[TILE][TILE];

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
            As[ty][tx] = A[row * K + (k0 + tx)];
        } else As[ty][tx] = 0.0f;

        // Load W tile: W[col, k0+ty]   
        if (col < J && (k0 + ty) < K) {
            Ws[tx][ty] = W[col * K + (k0 + ty)]; // I will load As and Ws in the good way, so we can iterate over
        } else Ws[tx][ty] = 0.0f;
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            acc += As[ty][kk] * Ws[tx][kk];
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
}

__global__ static void relu_layer_backward_d(float* __restrict__ A, float* __restrict__ W, float* __restrict__ Logits, float* __restrict__ C, int I, int J, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Ws[TILE][TILE];

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
            As[ty][tx] = A[row * K + (k0 + tx)];
        } else As[ty][tx] = 0.0f;

        // Load W tile: W[col, k0+ty]   
        if (col < J && (k0 + ty) < K) {
            Ws[tx][ty] = W[(k0 + ty) * J + col]; // I will load As and Ws in the good way, so we can iterate over
        } else Ws[tx][ty] = 0.0f;
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            acc += As[ty][kk] * Ws[tx][kk];
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
}

__global__ static void update_weight_d(float* __restrict__ A, float* __restrict__ W, float* __restrict__ C, float factor, int I, int J, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Ws[TILE][TILE];

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
            As[ty][tx] = A[(k0 + tx) * I + row];
        } else As[ty][tx] = 0.0f;

        // Load W tile: W[col, k0+ty]   
        if (col < J && (k0 + ty) < K) {
            Ws[tx][ty] = W[(k0 + ty) * J + col]; // I will load As and Ws in the good way, so we can iterate over
        } else Ws[tx][ty] = 0.0f;
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < TILE; ++kk) {
            acc += As[ty][kk] * Ws[tx][kk];
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

    update_weight_d<<<grid, block>>>(input.data, weight.data, output->data, factor, I, J, K);
}