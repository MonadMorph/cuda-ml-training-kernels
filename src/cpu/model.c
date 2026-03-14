#include "main.h"
#include <stdlib.h>
#include <assert.h>

static int IN_DIM = 784, HIDDEN_DIM_1 = 128, HIDDEN_DIM_2 = 256, OUT_DIM = 10;

typedef struct {
    Tensor A0, A1, A2, P;
    int initialized;
} Cache;

Model model_init() {
    Model mlp_model;

    int total_size = HIDDEN_DIM_1 + HIDDEN_DIM_2 + OUT_DIM;
    mlp_model.W1 = (Tensor){.ndim = 2, .shape = {HIDDEN_DIM_1, IN_DIM}, .size = IN_DIM * HIDDEN_DIM_1};
    mlp_model.b1 = (Tensor){.ndim = 1, .shape = {1, total_size}, .size = total_size};
    mlp_model.W2 = (Tensor){.ndim = 2, .shape = {HIDDEN_DIM_2, HIDDEN_DIM_1}, .size = HIDDEN_DIM_1 * HIDDEN_DIM_2};
    mlp_model.b2 = (Tensor){.ndim = 1, .shape = {1, HIDDEN_DIM_2}, .size = HIDDEN_DIM_2};    
    mlp_model.W3 = (Tensor){.ndim = 2, .shape = {OUT_DIM, HIDDEN_DIM_2}, .size = HIDDEN_DIM_2 * OUT_DIM};
    mlp_model.b3 = (Tensor){.ndim = 1, .shape = {1, OUT_DIM}, .size = OUT_DIM};

    srand(42);
    t_op.kaimin_init_tensor(&mlp_model.W1);
    t_op.kaimin_init_tensor(&mlp_model.W2);
    t_op.kaimin_init_tensor(&mlp_model.W3);

    t_op.zero_init_tensor(&mlp_model.b1);
    mlp_model.b2.data = mlp_model.b1.data + HIDDEN_DIM_1;
    mlp_model.b3.data = mlp_model.b2.data + HIDDEN_DIM_2;
    mlp_model.b1.size = HIDDEN_DIM_1;
    mlp_model.b1.shape[1] = HIDDEN_DIM_1;

    return mlp_model;
}

static Cache model_forward(Model *m, Tensor A0) {
    Cache cache;
    assert(A0.ndim == 2); // It should be batched
    assert(A0.shape[1] == IN_DIM);
    cache.A0 = A0;

    mat_op.linear_forward(A0, m->W1, m->b1, &cache.A1, 1);
    mat_op.linear_forward(cache.A1, m->W2, m->b2, &cache.A2, 1);
    mat_op.linear_forward(cache.A2, m->W3, m->b3, &cache.P, 0);
    t_op.softmax(&cache.P);

    return cache;
}

static void free_cache(Cache* cache) {
    t_op.free_tensor(&cache->A1);
    t_op.free_tensor(&cache->A2);
    t_op.free_tensor(&cache->P);
}

static void model_backward(Model *m, Cache cache, int* gt, float lr) {

    Tensor D3, D2, D1;
    D3 = cache.P;    
    t_op.softmax_backward(&D3, gt); // I will now do it in place
    mat_op.linear_back(D3, m->W3, cache.A2, &D2); // I should be using logits to calculate relu derivative, but that is the same as using A (after relu).
    mat_op.linear_back(D2, m->W2, cache.A1, &D1);

    // Do updates
    mat_op.update_weight(D1, cache.A0, lr, &m->W1);
    mat_op.update_weight(D2, cache.A1, lr, &m->W2);
    mat_op.update_weight(D3, cache.A2, lr, &m->W3);
    t_op.merged_update_bias(D1, D2, D3, lr, &m->b1);

    // free cache
    free_cache(&cache);
    t_op.free_tensor(&D2);
    t_op.free_tensor(&D1);
}

float train_one_epoch(Model*m, MNISTSet dataset, float lr, int batch_size, int stochastic) {
    float loss = 0;
    for (int i = 0; i < dataset.N; i += batch_size) {

        // Pack dataset into tensor, starting from i
        int batch = (i + batch_size < dataset.N) ? batch_size : (dataset.N - i);
        
        Tensor A0;
        if (stochastic) {
            A0 = (Tensor){.ndim = 2, .shape = {batch, dataset.data_size}, .size = batch * dataset.data_size};
            t_op.init_tensor(&A0);
            pack_batch_data(&A0, dataset, i);
        } else {
            A0 = (Tensor){.ndim = 2, .shape = {batch, dataset.data_size}, .size = batch * dataset.data_size};
            A0.data = dataset.X + i*dataset.data_size;
        }

        Cache cache = model_forward(m, A0);
        loss += t_op.cross_entropy_loss(cache.P, dataset.y_rand + i);
        model_backward(m, cache, dataset.y_rand + i, lr);
        if (stochastic) t_op.free_tensor(&A0);
    }
    return loss / dataset.N;
}

float accuracy_testing(Model*m, MNISTSet dataset, int batch_size, float* testing_loss) {
    int total_num_correct = 0;
    *testing_loss = 0;
    for (int i = 0; i < dataset.N; i += batch_size) {
        // Pack dataset into tensor, starting from i
        int batch = (i + batch_size < dataset.N) ? batch_size : (dataset.N - i);

        Tensor A0 = (Tensor){.ndim = 2, .shape = {batch, dataset.data_size}, .size = batch * dataset.data_size};
        A0.data = dataset.X + i*dataset.data_size;

        Cache cache = model_forward (m, A0);
        total_num_correct += greedy_accuracy(cache.P, dataset.y + i);
        *testing_loss += t_op.cross_entropy_loss(cache.P, dataset.y + i);
        free_cache(&cache);
    }
    *testing_loss /= (float) dataset.N;
    return (float) total_num_correct / (float) dataset.N;
}