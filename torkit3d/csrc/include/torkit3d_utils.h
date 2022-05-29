#pragma once
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor.")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous.")
#define CHECK_CONTIGUOUS_CUDA(x) \
  CHECK_CUDA(x);                 \
  CHECK_CONTIGUOUS(x)

// Get the block size (number of threads per block)
// The size is the largest power of 2 no more than n.
// Usually for algorithms using shared memory and parallel reduction.
inline int getBlock(const int n, const int max_threads_per_block)
{
  const int pow_2 = std::log(static_cast<double>(n)) / std::log(2.0);
  return std::max(std::min(1 << pow_2, max_threads_per_block), 1);
}

// Modified from at::cuda::getApplyGrid
inline void getGrid(uint64_t numBlocks, dim3 &grid, int64_t curDevice)
{
  uint64_t maxGridX = at::cuda::getDeviceProperties(curDevice)->maxGridSize[0];
  if (numBlocks > maxGridX)
    numBlocks = maxGridX;
  grid = dim3(numBlocks);
}