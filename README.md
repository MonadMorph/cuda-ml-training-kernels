# CUDA MLP Training from Scratch

A performance-oriented implementation of training a simple multilayer perceptron (MLP) on the MNIST dataset written in C and CUDA.  
The project focuses on understanding the performance characteristics of different matrix multiplication backends, ranging from naive CPU implementations to custom CUDA kernels and cuBLAS.

The codebase implements the entire training pipeline from scratch, including tensor operations, forward and backward propagation, loss computation, and optimization.

The goal of the project is to explore low-level performance engineering in machine learning systems.

---

# Features

- Fully from-scratch MLP training pipeline
- No high-level ML frameworks
- Multiple backends for matrix multiplication
- CPU and GPU implementations
- Custom tiled CUDA kernels
- cuBLAS implementation
- MNIST training and evaluation
- Benchmark comparison across implementations

---

# Run Instructions

## Dependencies
- For CPU: OpenBLAS, OpenMP (gcc-15 on macOS)
- For CUDA: CUDA toolkit, cuBLAS

## Download OpenBLAS

### For Ubuntu/Debian
`sudo apt update`
`sudo apt install libopenblas-dev pkg-config`

### For macOS with Homebrew:
`brew install openblas pkg-config`

## Download dataset
`sh download_mnist.sh`

## Build

### CPU Only
`make cpu`

### With CUDA
`make`

## Run

Basic usage:

```
./test <mode> [stochastic] [num_epoch] [lr] [batch_size]
```

### Modes

| Mode | Description |
|-----|-------------|
| 0 | naive matrix multiplication (CPU) |
| 1 | custom CPU tiled matrix multiplication |
| 2 | CPU BLAS matrix multiplication |
| 11 | custom GPU tiled matrix multiplication |
| 12 | cuBLAS multiplication |

### Default parameters

If optional arguments are not provided, the following defaults are used:

| Parameter | Default |
|-----------|--------|
| batch_size | 500 |
| num_epoch | 5 |
| lr | 0.1 |
| stochastic | 0 |


---

# Model

The model is a simple fully connected multilayer perceptron trained on MNIST.

Architecture:

784 -> Hidden (default 128) -> Hidden (default 256) -> 10


Training is performed using mini-batch gradient descent with cross entropy loss.

All forward and backward operations are implemented manually.


## Project Structure
- `src/cpu/`: CPU backend implementations
- `src/cuda/`: CUDA backend implementations
- `include/`: Header files
- `data/`: MNIST data
- `poc/`: Proof of concept scripts
- 
---

# Dataset

The project expects the MNIST dataset in the following format:

data/mnist/train-images-idx3-ubyte
data/mnist/train-labels-idx1-ubyte
data/mnist/t10k-images-idx3-ubyte
data/mnist/t10k-labels-idx1-ubyte


---

# Performance

Default parameters:

- batch_size = 500  
- num_epoch = 5  
- lr = 0.1  

## CPU (Apple M4, 10 cores)

| Mode | Time (s) | Accuracy |
|-----|--------|--------|
| 0 | 7.23 | 93.67% |
| 1 | 3.60 | 93.65% |
| 2 | 1.14 | 93.60% |

## GPU (RTX 4090)

| Mode | Time (s) | Accuracy |
|-----|--------|--------|
| 11 | 0.201 | 94.11% |
| 11 (stochastic) | 0.327 | 94.11% |
| 12 | 0.225 | 94.05% |
| 12 (stochastic) | 0.343 | 94.05% |

The custom CUDA tiled matrix multiplication slightly outperforms cuBLAS under these conditions.

---

# Implementation Notes

Key implementation aspects include:

- tiled shared-memory CUDA matrix multiplication
- manual forward and backward propagation
- custom tensor abstraction
- kernel-based softmax and loss computation
- merged bias update kernel
- pluggable backend design via operator table

The project is designed to make it easy to compare performance between different implementations.

---

# Purpose

This project was developed as part of a high-performance computing course to explore:

- GPU programming
- CUDA kernel optimization
- BLAS libraries
- performance bottlenecks in ML training systems

It also serves as a learning exercise in building machine learning infrastructure from first principles.