#include "layer.h"
#include <math.h>
#include <omp.h>

void linear_layer_forward_tiled(Tensor input, Tensor weight, Tensor bias, Tensor* output, int do_relu) {
    // shapes
    const int I = input.shape[0];      // input outer dim (batch)
    const int J = weight.shape[0];     // weight outer dim (out)
    const int K = input.shape[1];      // "hot dimension" (in)

    // This is always true
    const int out_stride = output->shape[1];
    const int in_stride  = input.shape[1];
    const int weight_stride = weight.shape[1];

    // Tile sizes (tune later)
    const int TI = 8;   // tile over i
    const int TJ = 16;   // tile over j
    const int TK = 128;   // tile over k (reduction)

    // Parallelize over (i0, j0) tiles
    // Collapse gives more parallel chunks
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i0 = 0; i0 < I; i0 += TI) {
        for (int j0 = 0; j0 < J; j0 += TJ) {
            const int i_max = (i0 + TI < I) ? (i0 + TI) : I;
            const int j_max = (j0 + TJ < J) ? (j0 + TJ) : J;

            // Initialize this output tile with bias
            for (int i = i0; i < i_max; i++) {
                float *out_row = output->data + i * out_stride;
                for (int j = j0; j < j_max; j++) {
                    float acc = bias.data[j % bias.size]; // bias size N
                    out_row[j] = acc;
                }
            }

            // Accumulate over K in TK chunks.
            for (int k0 = 0; k0 < K; k0 += TK) {
                const int k_max = (k0 + TK < K) ? (k0 + TK) : K;
                for (int i = i0; i < i_max; i++) {
                    const float *in_row = input.data + i * in_stride;
                    float *out_row = output->data + i * out_stride;
                    for (int j = j0; j < j_max; j++) {
                        const float *w_row = weight.data + j * weight_stride; // weight[j, :]
                        float acc = out_row[j]; // continue accumulating

                        // inner reduction
                        for (int k = k0; k < k_max; k++) acc += in_row[k] * w_row[k]; // 6 layer loop!
                        out_row[j] = acc;
                    }
                }
            }

            // Optional ReLU on this tile.
            if (do_relu) {
                for (int i = i0; i < i_max; ++i) {
                    float *out_row = output->data + i * out_stride;
                    for (int j = j0; j < j_max; ++j) out_row[j] = (out_row[j] > 0.0f) ? out_row[j] : 0.0f;
                }
            }
        }
    }
}

void relu_layer_backward_tiled(Tensor input, Tensor weight, Tensor cur_logits, Tensor* output) {
    // shapes
    const int I = input.shape[0];      // input outer dim (batch)
    const int J = weight.shape[1];     // weight outer dim (in)
    const int K = input.shape[1];      // "hot dimension" (out)

    *output = (Tensor){ .ndim = 2, .shape = {I, J}, .size = I * J};
    zero_init_tensor(output);

    // This is always true
    const int out_stride = output->shape[1];
    const int in_stride  = input.shape[1];
    const int weight_stride = weight.shape[1];
    const int logits_stride = cur_logits.shape[1];

    // Tile sizes (tune later)
    const int TI = 8;   // tile over i
    const int TJ = 16;   // tile over j
    const int TK = 64;   // tile over k (reduction)

    // Parallelize over (i0, j0) tiles
    // Collapse gives more parallel chunks
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i0 = 0; i0 < I; i0 += TI) {
        for (int j0 = 0; j0 < J; j0 += TJ) {
            const int i_max = (i0 + TI < I) ? (i0 + TI) : I;
            const int j_max = (j0 + TJ < J) ? (j0 + TJ) : J;

            // Initialize this output tile with 0
            for (int i = i0; i < i_max; i++) {
                float *out_row = output->data + i * out_stride;
                for (int j = j0; j < j_max; j++) out_row[j] = 0;
            }

            // Accumulate over K in TK chunks.
            for (int k0 = 0; k0 < K; k0 += TK) {
                const int k_max = (k0 + TK < K) ? (k0 + TK) : K;
                for (int i = i0; i < i_max; i++) {
                    const float *in_row = input.data + i * in_stride;
                    float *out_row = output->data + i * out_stride;
                    float *cur_logits_row = cur_logits.data + i * logits_stride;
                    for (int j = j0; j < j_max; j++) {
                        if (cur_logits_row[j] <= 0.0f) continue; // Hadamard by z (or a) thing
                        const float *w_col = weight.data + j; // weight[:, j]
                        float acc = 0;
                        // inner reduction
                        for (int k = k0; k < k_max; k++) acc += in_row[k] * w_col[k*weight_stride]; // 6 layer loop!
                        out_row[j] += acc;
                    }
                }
            }
        }
    }
}

void update_weight_tiled(Tensor input, Tensor weight, float lr, Tensor* output) {
    // shapes
    // input is D_i, weight is A_{i-1} and out is W_i
    const int I = input.shape[1];      // input outer dim (out)
    const int J = weight.shape[1];     // weight outer dim (in)
    const int K = input.shape[0];      // "hot dimension" (batch)

    // This is always true
    const int out_stride = output->shape[1];
    const int in_stride  = input.shape[1];
    const int weight_stride = weight.shape[1];

    // Tile sizes (tune later)
    const int TI = 32;   // tile over i
    const int TJ = 32;   // tile over j
    const int TK = 16;   // tile over k (reduction)

    // Parallelize over (i0, j0) tiles
    // Collapse gives more parallel chunks
    float factor = lr / K;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i0 = 0; i0 < I; i0 += TI) {
        for (int j0 = 0; j0 < J; j0 += TJ) {
            const int i_max = (i0 + TI < I) ? (i0 + TI) : I;
            const int j_max = (j0 + TJ < J) ? (j0 + TJ) : J;

            // Accumulate over K in TK chunks.
            for (int k0 = 0; k0 < K; k0 += TK) {
                const int k_max = (k0 + TK < K) ? (k0 + TK) : K;
                for (int i = i0; i < i_max; i++) {
                    const float *in_col = input.data + i;
                    float *out_row = output->data + i * out_stride;
                    for (int j = j0; j < j_max; j++) {
                        const float *w_col = weight.data + j; // weight[:, j]
                        float acc = 0;
                        // inner reduction
                        for (int k = k0; k < k_max; k++) acc += in_col[k*in_stride] * w_col[k*weight_stride]; // 6 layer loop!
                        out_row[j] -= acc * factor;
                    }
                }
            }
        }
    }
}