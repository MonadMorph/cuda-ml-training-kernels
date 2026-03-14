#include "main.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

void pack_batch_data_cpu(Tensor* batch, MNISTSet dataset, int start) {
    int batch_size = batch->shape[0];
    int stride = batch->shape[1];

    for (int i = 0; i < batch_size; i++) {
        int idx = (start + i) % dataset.N;
        if (dataset.position) {
            idx = dataset.position[idx];
        }
        memcpy(&batch->data[i * stride], &dataset.X[idx * 784], sizeof(float) * 784);
    }
}