#include "main.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv){
    if (argc < 2) exit(1);

    set_global_operators(atoi(argv[1]));

    MNISTSet training, testing;
    load_mnist_set("data/mnist/train-images-idx3-ubyte",
        "data/mnist/train-labels-idx1-ubyte", &training);
    load_mnist_set("data/mnist/t10k-images-idx3-ubyte",
        "data/mnist/t10k-labels-idx1-ubyte", &testing);

    Model model = model_init();
    float lr = 0.1f;
    int batch_size = 500;
    int num_epoch = 5;
    int stochastic = 0;

    if (argc >= 3) stochastic = atoi(argv[2]);
    if (argc >= 4) num_epoch = atoi(argv[3]);
    if (argc >= 5) lr = (float) atof(argv[4]);
    if (argc >= 6) batch_size = atoi(argv[5]);

    float testing_loss;
    StartTimer();

    // Testing
    for (int i = 0; i < num_epoch; i++) {
        printf("----------Epoch %d----------\n", i+1);
        prep_dataset(&training, stochastic);
        float loss = train_one_epoch(&model, training, lr, batch_size, stochastic);
        printf("Training loss is: %f\n", loss);

        float accuracy = accuracy_testing(&model, testing, batch_size, &testing_loss);
        printf("Accuracy is: %f\n", accuracy);
        printf("Testing loss is: %f \n", testing_loss);
    }
    float tot_time = GetTimer();

    printf("Elapsed time is: %f s\n", tot_time);
}