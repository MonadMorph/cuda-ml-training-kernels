#include "main.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

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

    optind = 2;
    int opt;

    while ((opt = getopt(argc, argv, "se:b:l:")) != -1) {
        switch (opt) {
            case 's':
                stochastic = 1;
                break;
            case 'e':
                num_epoch = atoi(optarg);
                break;
            case 'b':
                batch_size = atoi(optarg);
                break;
            case 'l':
                lr = atof(optarg);
                break;
            default:
                fprintf(stderr,
                    "Usage: %s [-s] [-e epochs] [-b batch] [-l lr]\n",
                    argv[0]);
                exit(1);
        }
    }
    const char *mode_desc;

    switch (atoi(argv[1])) {
        case 0:
            mode_desc = "naive CPU matrix multiplication";
            break;
        case 1:
            mode_desc = "custom tiled CPU matrix multiplication";
            break;
        case 2:
            mode_desc = "CPU BLAS matrix multiplication";
            break;
        case 11:
            mode_desc = "custom tiled CUDA matrix multiplication";
            break;
        case 12:
            mode_desc = "cuBLAS matrix multiplication";
            break;
        default:
            mode_desc = "unknown mode";
            break;
    }

    printf("Running MNIST training and testing.\n");
    printf("Mode %d: %s\n", atoi(argv[1]), mode_desc);
    printf("Epochs: %d\n", num_epoch);
    printf("Batch size: %d\n", batch_size);
    printf("Learning rate: %.6f\n", lr);
    printf("Stochastic mode: %s\n", stochastic ? "enabled" : "disabled");

    training.batch_size = batch_size;
    testing.batch_size = batch_size;

    float testing_loss;
    StartTimer();
    data_ops.prep_dataset(&testing, 0);

    // Testing
    for (int i = 0; i < num_epoch; i++) {
        printf("----------Epoch %d----------\n", i+1);
        data_ops.prep_dataset(&training, stochastic);
        float loss = train_one_epoch(&model, training, lr, batch_size, stochastic);
        printf("Training loss is: %f\n", loss);

        float accuracy = accuracy_testing(&model, testing, batch_size, &testing_loss);
        printf("Accuracy is: %f\n", accuracy);
        printf("Testing loss is: %f \n", testing_loss);
    }
    float tot_time = GetTimer();

    printf("Elapsed time is: %f s\n", tot_time);
    destory_model(model);
    data_ops.destory_data(testing);
    data_ops.destory_data(training);
}