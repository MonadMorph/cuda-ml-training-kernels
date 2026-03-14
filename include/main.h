#include <stdint.h>
#ifndef MAIN_H
#define MAIN_H
#include "tensor.h"
#include "layer.h"

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
} MNISTSet;

int load_mnist_set(const char *img_path, const char *lbl_path, MNISTSet *set);
void free_mnist_set(MNISTSet *set);

extern void (*prep_dataset) (MNISTSet* dataset, int shuffle);
void prep_dataset_cpu(MNISTSet* dataset, int shuffle);
void prep_dataset_cuda(MNISTSet* dataset, int shuffle);

extern void (*pack_batch_data) (Tensor* batch, MNISTSet dataset, int start);
void pack_batch_data_cpu(Tensor* batch, MNISTSet dataset, int start);
void pack_batch_data_cuda(Tensor* batch, MNISTSet dataset, int start);

/*----------Model Section----------*/
typedef struct {
    Tensor W1, b1, W2, b2, W3, b3;
} Model;

Model model_init();
float train_one_epoch(Model*m, MNISTSet dataset, float lr, int batch_size, int stochastic);
float accuracy_testing(Model*m, MNISTSet dataset, int batch_size, float* testing_loss);

/*----------Util Section----------*/

void StartTimer(void);
float GetTimer(void);
void print_tensor(Tensor t);
void print_tensor_cuda(Tensor t);
int* make_shuffled_index(int N);

extern int (*greedy_accuracy) (Tensor probs, int* gt);
int greedy_accuracy_cpu(Tensor probs, int* gt);
int greedy_accuracy_cuda(Tensor probs, int* gt);

#ifdef __cplusplus
}
#endif
#endif