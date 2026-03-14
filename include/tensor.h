#ifndef TENSOR_H
#define TENSOR_H

#ifdef __cplusplus
extern "C" {
#endif

/*----------Tensor Operator Section----------*/
typedef struct {
    float *data;      // contiguous
    int ndim;
    int shape[2];     // Batch size/out dim, in dim
    int size;         // total elements
} Tensor;

typedef void (*kaimin_init_fn)(Tensor* tensor);
typedef void (*zero_init_fn)(Tensor* tensor);
typedef void (*init_tensor_fn)(Tensor* tensor);
typedef void (*init_tensor_from_fn)(Tensor* tensor, Tensor basis);
typedef void (*free_tensor_fn)(Tensor* tensor);

typedef void (*softmax_fn)(Tensor* logits);
typedef void (*softmax_backward_fn)(Tensor* error, int* gt);
typedef void (*update_bias_fn)(Tensor error, float lr, Tensor* bias);
typedef void (*merged_update_bias_fn)(Tensor D1, Tensor D2, Tensor D3, float lr, Tensor* bias);
typedef float (*cross_entropy_loss_fn)(Tensor probs, int* gt);

typedef struct {
    kaimin_init_fn  kaimin_init_tensor;
    zero_init_fn zero_init_tensor;
    init_tensor_fn    init_tensor;
    init_tensor_from_fn init_tensor_from;
    free_tensor_fn free_tensor;

    softmax_fn softmax;
    softmax_backward_fn softmax_backward;
    update_bias_fn update_bias;
    merged_update_bias_fn merged_update_bias;
    cross_entropy_loss_fn cross_entropy_loss;
} Tensor_Ops;

extern Tensor_Ops t_op;

/*----------Tensor Init Section----------*/
void kaimin_init_tensor(Tensor* tensor);
void zero_init_tensor(Tensor* tensor);
void init_tensor(Tensor* tensor);
void init_tensor_from(Tensor* tensor, Tensor basis);
void free_tensor(Tensor* tensor);

/*----------Cuda Tensor Init Section----------*/
void kaimin_init_tensor_cuda(Tensor* tensor);
void zero_init_tensor_cuda(Tensor* tensor);
void init_tensor_cuda(Tensor* tensor);
void init_tensor_from_cuda(Tensor* tensor, Tensor basis);
void free_tensor_cuda(Tensor* tensor); 

/*----------Tensor Operation Section----------*/
void softmax(Tensor* logits);
void softmax_backward(Tensor* error, int* gt);
void update_bias(Tensor error, float lr, Tensor* bias);
void merged_update_bias(Tensor D1, Tensor D2, Tensor D3, float lr, Tensor* bias);
float cross_entropy_loss(Tensor probs, int* gt);

/*----------Cuda Tensor Operation Section----------*/
void softmax_cuda(Tensor* logits);
void softmax_backward_cuda(Tensor* error, int* gt);
void update_bias_cuda(Tensor error, float lr, Tensor* bias);
void merged_update_bias_cuda(Tensor D1, Tensor D2, Tensor D3, float lr, Tensor* bias);
float cross_entropy_loss_cuda(Tensor probs, int* gt);

#ifdef __cplusplus
}
#endif
#endif