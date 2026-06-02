#include "main.h"
#include <stdlib.h>
#include <assert.h>

static int IN_DIM = 784, HIDDEN_DIM_1 = 128, HIDDEN_DIM_2 = 256, OUT_DIM = 10;

typedef struct {
    Tensor A0, A1, A2, P, D2, D1;
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

static void model_forward(Model *m, Cache* cache) {
    Tensor A0 = cache->A0;
    assert(A0.ndim == 2); // It should be batched
    assert(A0.shape[1] == IN_DIM);

    mat_op.linear_forward(A0, m->W1, m->b1, &cache->A1, 1);
    mat_op.linear_forward(cache->A1, m->W2, m->b2, &cache->A2, 1);
    mat_op.linear_forward(cache->A2, m->W3, m->b3, &cache->P, 0);
    t_op.softmax(&cache->P);
}

static void init_cache(Cache* cache, int batch, Tensor A0) {
    cache->A0 = A0; // Mount data
    if (cache->initialized) {
        if (batch == cache->A1.shape[0]) return;

        cache->A1.shape[0] = batch;
        cache->A2.shape[0] = batch;
        cache->P.shape[0] = batch;
        cache->D2.shape[0] = batch;
        cache->D1.shape[0] = batch;
        return;
    }

    cache->A1 = (Tensor){.ndim = 2, .shape = {batch, HIDDEN_DIM_1}, .size = batch * HIDDEN_DIM_1};
    cache->A2 = (Tensor){.ndim = 2, .shape = {batch, HIDDEN_DIM_2}, .size = batch * HIDDEN_DIM_2};
    cache->P = (Tensor){.ndim = 2, .shape = {batch, OUT_DIM}, .size = batch * OUT_DIM};
    cache->D2 = (Tensor){.ndim = 2, .shape = {batch, HIDDEN_DIM_2}, .size = batch * HIDDEN_DIM_2};
    cache->D1 = (Tensor){.ndim = 2, .shape = {batch, HIDDEN_DIM_1}, .size = batch * HIDDEN_DIM_1};

    t_op.init_tensor(&cache->A1);
    t_op.init_tensor(&cache->A2);
    t_op.init_tensor(&cache->P);
    t_op.init_tensor(&cache->D2);
    t_op.init_tensor(&cache->D1);
    cache->initialized = 1;
}

static void free_cache(Cache* cache) {
    t_op.free_tensor(&cache->A1);
    t_op.free_tensor(&cache->A2);
    t_op.free_tensor(&cache->P);
    t_op.free_tensor(&cache->D2);
    t_op.free_tensor(&cache->D1);
}

static void model_backward(Model *m, Cache cache, int* gt, float lr) {

    Tensor D3;
    D3 = cache.P;    
    t_op.softmax_backward(&D3, gt); // I will now do it in place
    mat_op.linear_back(D3, m->W3, cache.A2, &cache.D2); // I should be using logits to calculate relu derivative, but that is the same as using A (after relu).
    mat_op.linear_back(cache.D2, m->W2, cache.A1, &cache.D1);

    // Do updates
    mat_op.update_weight(cache.D1, cache.A0, lr, &m->W1);
    mat_op.update_weight(cache.D2, cache.A1, lr, &m->W2);
    mat_op.update_weight(D3, cache.A2, lr, &m->W3);
    t_op.update_bias(cache.D1, lr, &m->b1);
    t_op.update_bias(cache.D2, lr, &m->b2);
    t_op.update_bias(D3, lr, &m->b3);
}

float train_one_epoch(Model*m, MNISTSet dataset, float lr, int batch_size, int stochastic) {
    Cache cache = (Cache) {.initialized = 0};
    Tensor A0 = (Tensor){.ndim = 2, .shape = {batch_size, dataset.data_size}, .size = batch_size * dataset.data_size};
    if (stochastic) data_ops.fetch_data(&A0, dataset, 0);

    for (int i = 0; i < dataset.N; i += batch_size) {
        // Pack dataset into tensor, starting from i
        int batch = (i + batch_size < dataset.N) ? batch_size : (dataset.N - i);
        Tensor A0 = (Tensor){.ndim = 2, .shape = {batch, dataset.data_size}, .size = batch * dataset.data_size};
        
        if (stochastic) {
            data_ops.pack_batch_data(&A0, dataset, i); // retrieve data from current batch
            int next_i = i + batch_size; // (async) fetch data for next batch
            if (next_i < dataset.N) {
                int next_batch = (next_i + batch_size < dataset.N) ? batch_size : (dataset.N - next_i);
                Tensor next_A0 = (Tensor){.ndim = 2, .shape = {next_batch, dataset.data_size}, .size = next_batch * dataset.data_size};
                data_ops.fetch_data(&next_A0, dataset, next_i);
            }
        } else A0.data = dataset.X + i*dataset.data_size; // directly get batch i data

        init_cache(&cache, batch, A0);
        model_forward(m, &cache);
        t_op.cross_entropy_loss(cache.P, dataset.y_rand + i);
        model_backward(m, cache, dataset.y_rand + i, lr);
    }

    free_cache(&cache);
    return t_op.get_loss() / dataset.N;
}

float accuracy_testing(Model*m, MNISTSet dataset, int batch_size, float* testing_loss) {
    Cache cache = (Cache) {.initialized = 0};

    for (int i = 0; i < dataset.N; i += batch_size) {
        // Pack dataset into tensor, starting from i
        int batch = (i + batch_size < dataset.N) ? batch_size : (dataset.N - i);

        Tensor A0 = (Tensor){.ndim = 2, .shape = {batch, dataset.data_size}, .size = batch * dataset.data_size};
        A0.data = dataset.X + i*dataset.data_size;
        init_cache(&cache, batch, A0);

        model_forward(m, &cache);
        greedy_accuracy(cache.P, dataset.y_rand + i);
        t_op.cross_entropy_loss(cache.P, dataset.y_rand + i);
    }

    free_cache(&cache);
    *testing_loss = t_op.get_loss();
    *testing_loss /= (float) dataset.N;
    return (float) get_accuracy() / (float) dataset.N;
}

void destory_model(Model m) {
    t_op.free_tensor(&m.W1);
    t_op.free_tensor(&m.b1);
    t_op.free_tensor(&m.W2);
    t_op.free_tensor(&m.W3);

    mat_op.destory_weight_sync();
    t_op.destory_bias_sync();
}