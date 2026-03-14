# Run Instructions

## Dependencies
- For CPU: OpenBLAS, OpenMP (gcc-15 on macOS)
- For CUDA: CUDA toolkit, OpenBLAS

## Build

### CPU Only
`make cpu`

### With CUDA
`make`

## Run

Basic usage:

```
./test <mode> [batch_size] [num_epoch] [lr] [stochastic]
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

### Performance
mode 0: 7.67    accuracy: 93.67%
mode 1: 3.66    accuracy: 93.65%
mode 2: 1.24    accuracy: 93.6%


mode 11:    0.201 acc: 94.11%
mode 11 stocastic 0.327 acc: 94.11%
mode 12:    0.225 acc: 94.05%
mode 12 stocastic 0.343 acc: 94.11%


## Project Structure
- `src/cpu/`: CPU backend implementations
- `src/cuda/`: CUDA backend implementations
- `include/`: Header files
- `data/`: MNIST data
- `poc/`: Proof of concept scripts