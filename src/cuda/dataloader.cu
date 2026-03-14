#include "main.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

static int read_be_u32(FILE *f){
    unsigned char b[4];
    fread(b, 1, 4, f);
    return (b[0] << 24) | (b[1] << 16) | (b[2] <<  8) | b[3];
}

static int load_mnist_images(const char *path, float **X, int *N) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror("fopen");
        exit(1);
    }

    read_be_u32(f);          // magic

    int n = read_be_u32(f);
    read_be_u32(f);          // rows
    read_be_u32(f);          // cols

    float *images = (float*) malloc(n * 784 * sizeof(float));
    unsigned char *buf = (unsigned char*) malloc(n * 784);
    fread(buf, 1, n * 784, f);
    fclose(f);

    for (int i = 0; i < n * 784; ++i)
        images[i] = buf[i] / 255.0f;

    free(buf);
    *X = images;
    *N = n;
    return 0;
}

static int load_mnist_labels(const char *path, int **y, int *N) {
    FILE *f = fopen(path, "rb");
    read_be_u32(f);          // magic
    int n = read_be_u32(f);

    unsigned char *buf = (unsigned char *) malloc(n);
    fread(buf, 1, n, f);
    fclose(f);

    int *labels = (int *) malloc(n * sizeof(int));
    for (int i = 0; i < n; ++i)
        labels[i] = (int)buf[i];

    free(buf);
    *y = labels;
    *N = n;
    return 0;
}

int load_mnist_set(const char *img_path, const char *lbl_path, MNISTSet *set) {
    int n1, n2;
    load_mnist_images(img_path, &set->X, &n1);
    load_mnist_labels(lbl_path, &set->y, &n2);
    set->N = n1;
    set->data_size = 784;
    set->position = NULL;
    set->y_rand = NULL;
    set->on_device = 0;
    return 0;
}

void free_mnist_set(MNISTSet *set) {
    free(set->X);
    free(set->y);
}

void prep_dataset_cpu(MNISTSet* dataset, int shuffle) {
    if (shuffle == 0) {
        dataset->y_rand = dataset->y;    
        return;
    }

    free(dataset->position);
    dataset->position = make_shuffled_index(dataset->N);
    if (dataset->y_rand == NULL) dataset->y_rand = (int*) malloc(dataset->N * sizeof(int));

    for (int i = 0; i < dataset->N; i++) {
        dataset->y_rand[i] = dataset->y[dataset->position[i]];
    }
}

void prep_dataset_cuda(MNISTSet* dataset, int shuffle) {
    if (shuffle == 0) {
        if (dataset->on_device) return; 
        float *X_h, *X_d;

        cudaMalloc(&X_d, dataset->N * dataset->data_size * sizeof(float));
        cudaMalloc(&dataset->y_rand, dataset->N * sizeof(int));

        cudaMemcpy(X_d, dataset->X, dataset->N * dataset->data_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dataset->y_rand, dataset->y, dataset->N * sizeof(int), cudaMemcpyHostToDevice);

        X_h = dataset->X;
        dataset->X = X_d;
        //free(X_h);
        dataset->on_device = 1;
        return;
    }

    free(dataset->position);
    dataset->position = make_shuffled_index(dataset->N);
    if (dataset->y_rand == NULL) cudaMalloc(&dataset->y_rand, dataset->N * sizeof(int));
    int* new_y = (int*) malloc(dataset->N * sizeof(int));

    for (int i = 0; i < dataset->N; i++) {
        new_y[i] = dataset->y[dataset->position[i]];
    }

    cudaMemcpy(dataset->y_rand, new_y, dataset->N * sizeof(int), cudaMemcpyHostToDevice);
    free(new_y);
}

void pack_batch_data_cpu(Tensor* batch, MNISTSet dataset, int start) {
    float* batch_data = batch->data;
    int stride = dataset.data_size;
    int batch_size = batch->shape[0];

    for (int i = 0; i < batch_size; i++) {
        memcpy(batch_data + i * stride, dataset.X + dataset.position[i+start] * stride, stride * sizeof(float));
    }
}

void pack_batch_data_cuda(Tensor* batch, MNISTSet dataset, int start) {
    int stride = dataset.data_size;
    int batch_size = batch->shape[0];

    float* data_h = (float*) malloc(sizeof(float) * batch_size * stride);
    for (int i = 0; i < batch_size; i++) {
        memcpy(data_h + i * stride, dataset.X + dataset.position[i+start] * stride, stride * sizeof(float));
    }

    cudaMemcpy(batch->data, data_h, sizeof(float) * batch_size * stride, cudaMemcpyHostToDevice);
    free(data_h);
}