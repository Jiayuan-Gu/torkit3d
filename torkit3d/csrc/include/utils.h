#pragma once
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor.")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous.")
#define CHECK_CONTIGUOUS_CUDA(x) \
  CHECK_CUDA(x);                 \
  CHECK_CONTIGUOUS(x)

// Get the block size (number of threads per block)
// The size is the largest power of 2 no more than n.
// Usually for algorithms using shared memory and parallel reduction.
inline uint64_t getBlockSize(const uint64_t n, const uint64_t max_threads_per_block)
{
  uint64_t block_size = 1;
  while ((block_size < n) && (block_size < max_threads_per_block))
    block_size *= 2;
  return block_size;
}

// Modified from at::cuda::getApplyGrid
inline void getGrid(uint64_t numBlocks, dim3 &grid, int64_t curDevice)
{
  uint64_t maxGridX = at::cuda::getDeviceProperties(curDevice)->maxGridSize[0];
  if (numBlocks > maxGridX)
    numBlocks = maxGridX;
  grid = dim3(numBlocks);
}