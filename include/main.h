#include <stdint.h>
#ifndef MAIN_H
#define MAIN_H
#include "tensor.h"
#include "layer.h"

#define MAX_PRINT 50

#ifdef __cplusplus
extern "C" {
#endif

void set_global_operators(int which); // 0 naive, 1 tiled, 2 blas, 11 cuda

/*----------Dataloader Section----------*/
typedef struct {
    float *X;   // N * 784 contiguous
    int *y;   // N
    int* position;
    int* y_rand;
    int    N;
    int    data_size;
    int on_device;
    int batch_size;
} MNISTSet;

int load_mnist_set(const char *img_path, const char *lbl_path, MNISTSet *set);
void free_mnist_set(MNISTSet *set);

typedef void (*prep_dataset_fn) (MNISTSet* dataset, int shuffle);
typedef void (*pack_batch_data_fn) (Tensor* batch, MNISTSet dataset, int start);
typedef void (*fetch_data_fn) (Tensor* batch, MNISTSet dataset, int start);
typedef void (*destory_data_fn) (MNISTSet dataset);

typedef struct {
    prep_dataset_fn  prep_dataset;
    pack_batch_data_fn pack_batch_data;
    fetch_data_fn    fetch_data;
    destory_data_fn destory_data;
} Dataset_Operators;

extern Dataset_Operators data_ops;

void prep_dataset_cpu(MNISTSet* dataset, int shuffle);
void prep_dataset_cuda(MNISTSet* dataset, int shuffle);

void pack_batch_data_cpu(Tensor* batch, MNISTSet dataset, int start);
void pack_batch_data_cuda(Tensor* batch, MNISTSet dataset, int start);

void fetch_data_cuda(Tensor* batch, MNISTSet dataset, int start);
void fetch_data_cpu(Tensor* batch, MNISTSet dataset, int start);

void destory_data_cuda(MNISTSet dataset);
void destory_data_cpu(MNISTSet dataset);

/*----------Model Section----------*/
typedef struct {
    Tensor W1, b1, W2, b2, W3, b3;
} Model;

Model model_init();
float train_one_epoch(Model*m, MNISTSet dataset, float lr, int batch_size, int stochastic);
float accuracy_testing(Model*m, MNISTSet dataset, int batch_size, float* testing_loss);
void destory_model(Model m);

/*----------Util Section----------*/

void StartTimer(void);
float GetTimer(void);
void print_tensor(Tensor t);
void print_tensor_cuda(Tensor t);
int* make_shuffled_index(int N);

extern void (*greedy_accuracy) (Tensor probs, int* gt);
void greedy_accuracy_cpu(Tensor probs, int* gt);
void greedy_accuracy_cuda(Tensor probs, int* gt);

extern int (*get_accuracy) (void);
int get_accuracy_cpu(void);
int get_accuracy_cuda(void);

#ifdef __cplusplus
}
#endif
#endif