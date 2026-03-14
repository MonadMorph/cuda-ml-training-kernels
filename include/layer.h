#ifndef LAYER_H
#define LAYER_H
#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

/*----------Matrix Operator Section----------*/
typedef void (*linear_forward_fn)(Tensor input, Tensor weight, Tensor bias, Tensor* output, int do_relu);
typedef void (*linear_back_fn)(Tensor input, Tensor weight, Tensor cur_logits, Tensor* output);
typedef void (*update_weight_fn)(Tensor input, Tensor weight, float lr, Tensor* output);

typedef struct {
    linear_forward_fn  linear_forward;
    linear_back_fn linear_back;
    update_weight_fn    update_weight;
} Mat_Operators;

extern Mat_Operators mat_op;

/*----------Naive Matrix Operation Section----------*/
void linear_layer_forward(Tensor input, Tensor weight, Tensor bias, Tensor* output, int do_relu);
void relu_layer_backward(Tensor next_weight, Tensor next_error, Tensor cur_logits, Tensor* cur_error);
void update_weight(Tensor error, Tensor cached, float lr, Tensor* weight);

/*----------Tiled Matrix Operation Section----------*/
void linear_layer_forward_tiled(Tensor input, Tensor weight, Tensor bias, Tensor* output, int do_relu);
void relu_layer_backward_tiled(Tensor input, Tensor weight, Tensor cur_logits, Tensor* output);
void update_weight_tiled(Tensor input, Tensor weight, float lr, Tensor* output);

/*----------BLAS Matrix Operation Section----------*/
void linear_layer_forward_blas(Tensor input, Tensor weight, Tensor bias, Tensor* output, int do_relu);
void relu_layer_backward_blas(Tensor input, Tensor weight, Tensor cur_logits, Tensor* output);
void update_weight_blas(Tensor input, Tensor weight, float lr, Tensor* output);

/*----------Cuda Matrix Operation Section----------*/
void linear_layer_forward_cuda(Tensor input, Tensor weight, Tensor bias, Tensor* output, int do_relu);
void relu_layer_backward_cuda(Tensor input, Tensor weight, Tensor cur_logits, Tensor* output);
void update_weight_cuda(Tensor input, Tensor weight, float lr, Tensor* output);

/*----------Cublas Matrix Operation Section----------*/
void linear_layer_forward_cublas(Tensor input, Tensor weight, Tensor bias, Tensor* output, int do_relu);
void relu_layer_backward_cublas(Tensor input, Tensor weight, Tensor cur_logits, Tensor* output);
void update_weight_cublas(Tensor input, Tensor weight, float lr, Tensor* output);

#ifdef __cplusplus
}
#endif
#endif