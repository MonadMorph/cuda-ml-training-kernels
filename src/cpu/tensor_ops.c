#include "main.h"
#include "tensor.h"
#include <math.h>

// In place softmax
void softmax(Tensor* logits) {
    int shape[2] = {logits->shape[0], logits->shape[1]};
    float* data = logits->data;
    
    // #pragma omp parallel for schedule(static)
    for (int i = 0; i < shape[0]; i++) {
        float max = data[i*shape[1]];
        for (int j = 1; j < shape[1]; j++) {
            max = data[i*shape[1] + j] > max ? data[i*shape[1] + j] : max;
        }
        float sum = 0;
        for (int j = 0; j < shape[1]; j++) {
            data[i*shape[1] + j] = exp(data[i*shape[1] + j] - max);
            sum += data[i*shape[1] + j];
        }
        for (int j = 0; j < shape[1]; j++) {
            data[i*shape[1] + j] /= sum;
        }
    }
}

void softmax_backward(Tensor* error, int* gt) {

    float* data = error->data;
    int stride = error->shape[1];
    int batch_size = error->shape[0];
    for (int i = 0; i < batch_size; i ++) {
        data[i * stride + gt[i]] -= 1;
    }
}

void update_bias(Tensor error, float lr, Tensor* bias) {

    int batch_size = error.shape[0];
    int stride = error.shape[1];
    float factor = lr / batch_size;

    // #pragma omp parallel for schedule(static)
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < error.shape[1]; j++) {
            bias->data[j] -= error.data[i*stride+j] * factor;
        }
    }
}

void merged_update_bias(Tensor D1, Tensor D2, Tensor D3, float lr, Tensor* bias) {

    int batch_size = D1.shape[0];
    int D2_threshold = D1.shape[1];
    int D3_threshold = D2_threshold + D2.shape[1];
    int total = D3_threshold + D3.shape[1];
    float factor = - lr / batch_size;

    // #pragma omp parallel for schedule(static)
    for (int i = 0; i < total; i++) {
        if (i < D3_threshold) {
            if (i < D2_threshold) {
                // D1 code
                int stride = D1.shape[1];
                for (int j = 0; j < batch_size; j++) {
                    bias->data[i] += D1.data[j*stride+i] * factor;
                }
            } else {
                // D2 code
                int stride = D2.shape[1];
                int local_i = i - D2_threshold;
                for (int j = 0; j < batch_size; j++) {
                    bias->data[i] += D2.data[j*stride+local_i] * factor;
                }
            }
        } else {
            // D3 code
            int stride = D3.shape[1];
            int local_i = i - D3_threshold;
            for (int j = 0; j < batch_size; j++) {
                bias->data[i] += D3.data[j*stride+local_i] * factor;
            }
        }
    }
}

float cross_entropy_loss(Tensor probs, int* gt) {
    float loss = 0;
    int stride = probs.shape[1];
    int batch_size = probs.shape[0];

    // #pragma omp parallel for schedule(static)
    for (int i = 0; i < batch_size; i ++) {
        loss -= log(probs.data[i * stride + gt[i]]);
    }
    return loss;
}