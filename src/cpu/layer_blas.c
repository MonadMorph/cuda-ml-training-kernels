#include "layer.h"
#include <math.h>
#include <cblas.h>

/*
void cblas_sgemm(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB,
                 const int M, const int N, const int K,
                 const float alpha, const float *A, const int lda,
                 const float *B, const int ldb, const float beta,
                 float *C, const int ldc);
*/

// For some weird reason I got error: expected identifier or '(' before '__extension__' if I use the variable name "I". Changed everything to II.
void linear_layer_forward_blas(Tensor input, Tensor weight, Tensor bias, Tensor* output, int do_relu) {
    // shapes
    const int II = input.shape[0];      // input outer dim (batch)
    const int J = weight.shape[0];     // weight outer dim (out)
    const int K = input.shape[1];      // "hot dimension" (in)

    // This is always true
    const int out_stride = output->shape[1];
    const int in_stride  = input.shape[1];
    const int weight_stride = weight.shape[1];

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
        II, J, K,
        1.0f, input.data, in_stride,
        weight.data, weight_stride, 0.0f,
        output->data, out_stride);

    // add bias and do relu
    float* data = output->data;
    int stride = bias.size;
    //#pragma omp parallel for
    for (int i = 0; i < output->size; i++) {
        float acc = data[i] + bias.data[i%stride];
        data[i] = do_relu ? (acc > 0 ? acc : 0) : acc;
    }
}

void relu_layer_backward_blas(Tensor input, Tensor weight, Tensor cur_logits, Tensor* output) {
    // shapes
    const int II = input.shape[0];      // input outer dim (batch)
    const int J = weight.shape[1];     // weight outer dim (in)
    const int K = input.shape[1];      // "hot dimension" (out)

    // This is always true
    const int out_stride = output->shape[1];
    const int in_stride  = input.shape[1];
    const int weight_stride = weight.shape[1];

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
        II, J, K,
        1.0f, input.data, in_stride,
        weight.data, weight_stride, 0.0f,
        output->data, out_stride);

    // Do Hadamard
    float* data = output->data;
    //#pragma omp parallel for
    for (int i = 0; i < output->size; i++) data[i] = cur_logits.data[i] > 0 ? data[i] : 0;
}

void update_weight_blas(Tensor input, Tensor weight, float lr, Tensor* output) {
    // shapes
    // input is D_i, weight is A_{i-1} and out is W_i
    const int II = input.shape[1];      // input outer dim (out)
    const int J = weight.shape[1];     // weight outer dim (in)
    const int K = input.shape[0];      // "hot dimension" (batch)

    // This is always true
    const int out_stride = output->shape[1];
    const int in_stride  = input.shape[1];
    const int weight_stride = weight.shape[1];

    float factor = lr / K;
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
        II, J, K,
        -factor, input.data, in_stride,
        weight.data, weight_stride, 1.0f,
        output->data, out_stride);
}