# Pytorch Extension

## Glossary

<https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#glossary>

- Grid: A Grid is a collection of Threads. Threads in a Grid execute a Kernel Function and are divided into Thread Blocks.
- Thread Block: A Thread Block is a group of threads which execute on the same multiprocessor (SM). Threads within a Thread Block have access to shared memory and can be explicitly synchronized.
- Kernel Function: A Kernel Function is an implicitly parallel subroutine that executes under the CUDA execution and memory model for every Thread in a Grid.
- Host: The Host refers to the execution environment that initially invoked CUDA. Typically the thread running on a system's CPU processor.

## Useful Interfaces

- Get the current cuda device: `const auto curDevice = at::cuda::current_device();`
- Get grid size according the cuda device: `at::cuda::getApplyGrid` (`#include <ATen/cuda/ApplyGridUtils.cuh>`)

## Release Notes for PyTorch

1.11:
- `THCudaCheck` is deprecated and instead use `C10_CUDA_CHECK`. [Github issue](https://github.com/pytorch/pytorch/pull/66391). Add `#include <ATen/cuda/CUDAContext.h>`.

1.5:
- `Tensor.type()` is deprecated and instead use `Tensor.options()`
- `Tensor.data()` is deprecated and instead use `Tensor.data_ptr()`

1.2:
- `AT_CHECK` -> `TORCH_CHECK`

0.4:
- `AT_ASSERT` -> `AT_CHECK`