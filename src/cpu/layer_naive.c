#include "layer.h"
#include <math.h>
#include <omp.h>

// output = input*weight + bias (relu)
void linear_layer_forward(Tensor input, Tensor weight, Tensor bias, Tensor* output, int do_relu) {

    int input_transposed = 0, weight_transposed = 1;

    //This is the hard part, I am doing input*weight, because output dimension is easier to track. I am using row major matrix, so if input is not transposed while output is transposed, we are in the sweet spot.
    *output = (Tensor){.ndim = 2, .shape = {input.shape[input_transposed], weight.shape[!weight_transposed]}, .size = input.shape[input_transposed] * weight.shape[!weight_transposed]};
    init_tensor(output); 
    
    int out_stride = output->shape[1];
    int in_stride = input.shape[1];
    int weight_stride = weight.shape[1];
    // Book keeping: 3 dimensions
    // i dimension: input[0], output[0] 
    // j dimension: weight[0], output[1] 
    // k dimension: input[1], weight[1]

    // Big old 3 layer for loop
    #pragma omp parallel for
    for (int index = 0; index < output->size; index++) {
        int i = index / out_stride;
        int j = index % out_stride;
        float acc = bias.data[(i*out_stride + j) % bias.size];
        for (int k = 0; k < in_stride; k++) {
            acc += 
            input.data[i * in_stride + k] * 
            weight.data[j * weight_stride + k]; // I wish I did not used a flat matrix
        }
        // Do relu here
        output->data[i*out_stride + j] = do_relu ? (acc > 0 ? acc : 0) : acc;
    }
}

void relu_layer_backward(Tensor input, Tensor weight, Tensor cur_logits, Tensor* output) {
    int input_transposed = 0, weight_transposed = 0;

    *output = (Tensor){.ndim = 2, .shape = {input.shape[input_transposed], weight.shape[!weight_transposed]}, .size = input.shape[input_transposed] * weight.shape[!weight_transposed]};
    init_tensor(output); 
    
    int out_stride = output->shape[1];
    int in_stride = input.shape[1];
    int weight_stride = weight.shape[1];
    int logits_stride = cur_logits.shape[1];
    // Book keeping: 3 dimensions
    // i dimension: input[0], output[0] 
    // j dimension: weight[1], output[1] 
    // k dimension: input[1], weight[0]

    // Big old 3 layer for loop
    #pragma omp parallel for
    for (int index = 0; index < output->size; index++) {
        int i = index / out_stride;
        int j = index % out_stride;
        // The Hadamard by z (or a) thing
        if (cur_logits.data[i*logits_stride + j] <= 0) {
            output->data[i*out_stride + j] = 0;
            continue;
        }
        float acc = 0;
        for (int k = 0; k < in_stride; k++) {
            acc += 
            input.data[i * in_stride + k] * 
            weight.data[k * weight_stride + j]; // I wish I did not used a flat matrix
        }
        output->data[i*out_stride + j] = acc;
    }
}

void update_weight(Tensor input, Tensor weight, float lr, Tensor* output) {

    int out_stride = output->shape[1];
    int in_stride = input.shape[1];
    int weight_stride = weight.shape[1];
    // Book keeping: 3 dimensions
    // i dimension: input[1], output[0] 
    // j dimension: weight[1], output[1] 
    // k dimension: input[0], weight[0]
    int max_k = input.shape[0];

    // Big old 3 layer for loop
    float factor = lr / input.shape[0];
    #pragma omp parallel for
    for (int index = 0; index < output->size; index++) {
        int i = index / out_stride;
        int j = index % out_stride;
        float acc = 0;
        for (int k = 0; k < max_k; k++) {
            acc += 
            input.data[k * in_stride + i] * 
            weight.data[k * weight_stride + j]; // I wish I did not used a flat matrix.
        }
        // printf("%f\n", acc);
        output->data[i*out_stride + j] -= factor * acc;
    }
}