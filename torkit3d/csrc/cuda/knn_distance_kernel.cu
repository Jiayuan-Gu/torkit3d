// CUDA Implementation for KNN with distance.

#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/ApplyGridUtils.cuh>
#include "utils.h"
#include "mink.cuh"

// Each block processes a chunk of `query`, and iterates over chunks of `key`.
// We load a chunk of `key` into the shared memory to reduce memory access.
// K is predefined.
template <unsigned int BLOCK_SIZE, unsigned int K, unsigned int DIM, typename scalar_t, typename index_t>
__global__ void knn_distance_kernel(
    scalar_t *__restrict__ distance,    // [B, N1, K]
    index_t *__restrict__ index,        // [B, N1, K]
    const scalar_t *__restrict__ query, // [B, N1, D]
    const scalar_t *__restrict__ key,   // [B, N2, D]
    const int bs,
    const int n1,
    const int n2)
{
  const int n_blocks = (n1 + BLOCK_SIZE - 1) / BLOCK_SIZE; // number of thread blocks for query and outputs
  const int n_chunks = (n2 + BLOCK_SIZE - 1) / BLOCK_SIZE; // number of data chunks for key
  const int total_blocks = bs * n_blocks;
  const int tid = threadIdx.x;

  for (int block_idx = blockIdx.x; block_idx < total_blocks; block_idx += gridDim.x)
  {
    const int batch_idx = block_idx / n_blocks;
    const int chunk_idx1 = block_idx % n_blocks;
    const int query_idx = (chunk_idx1 * BLOCK_SIZE) + tid;
    const int query_offset = (batch_idx * n1 + query_idx) * DIM;

    // Load current query data
    scalar_t cur_query[DIM] = {0.0};
    if (query_idx < n1)
    {
#pragma unroll
      for (int d = 0; d < DIM; ++d)
        cur_query[d] = query[query_offset + d];
    }

    __shared__ scalar_t key_buffer[BLOCK_SIZE * DIM];
    scalar_t mink_dist[K] = {0}; // top K nearest distance
    index_t mink_idx[K] = {0};   // top K nearest indices
    MinK<scalar_t, index_t> mink(mink_dist, mink_idx, K);

    // Sweep over chunks of key data to find k nearest neighbors
    for (int chunk_idx2 = 0; chunk_idx2 < n_chunks; ++chunk_idx2)
    {
      // Load a chunk of key data into shared memory within the block
      const int key_idx_t = (chunk_idx2 * BLOCK_SIZE) + tid;
      const int key_offset = (batch_idx * n2 + key_idx_t) * DIM;
      if (key_idx_t < n2)
      {
#pragma unroll
        for (int d = 0; d < DIM; ++d)
          key_buffer[tid * DIM + d] = key[key_offset + d];
      }
      __syncthreads();

      if (query_idx < n1)
      {
        // Compare the current query point and all key points in the shared memory
        for (int i = 0; i < BLOCK_SIZE; ++i)
        {
          const int key_idx_i = (chunk_idx2 * BLOCK_SIZE) + i;

          // Compute the distance
          scalar_t dist = 0.0;
#pragma unroll
          for (int d = 0; d < DIM; ++d)
          {
            scalar_t diff = key_buffer[i * DIM + d] - cur_query[d];
            dist += diff * diff;
          }

          if (key_idx_i < n2)
          {
            mink.add(dist, key_idx_i);
          }
        }
      }
      __syncthreads();
    }

    // Write the output
    const int out_offset = (batch_idx * n1 + query_idx) * K;
    if (query_idx < n1)
    {
#pragma unroll
      for (int k = 0; k < K; ++k)
      {
        index[out_offset + k] = mink_idx[k];
        distance[out_offset + k] = mink_dist[k];
      }
    }
  }
}

// For general k, we maintain the top k nearest neighbors in the global memory.
template <unsigned int BLOCK_SIZE, unsigned int DIM, typename scalar_t, typename index_t>
__global__ void knn_distance_kernel(
    scalar_t *__restrict__ distance,    // [B, N1, K]
    index_t *__restrict__ index,        // [B, N1, K]
    const scalar_t *__restrict__ query, // [B, N1, D]
    const scalar_t *__restrict__ key,   // [B, N2, D]
    const int bs,
    const int n1,
    const int n2,
    const int k)
{
  const int n_blocks = (n1 + BLOCK_SIZE - 1) / BLOCK_SIZE; // number of thread blocks for query and outputs
  const int n_chunks = (n2 + BLOCK_SIZE - 1) / BLOCK_SIZE; // number of data chunks for key
  const int total_blocks = bs * n_blocks;
  const int tid = threadIdx.x;

  for (int block_idx = blockIdx.x; block_idx < total_blocks; block_idx += gridDim.x)
  {
    const int batch_idx = block_idx / n_blocks;
    const int chunk_idx1 = block_idx % n_blocks;
    const int query_idx = (chunk_idx1 * BLOCK_SIZE) + tid;
    const int query_offset = (batch_idx * n1 + query_idx) * DIM;

    // Load current query data
    scalar_t cur_query[DIM] = {0.0};
    if (query_idx < n1)
    {
#pragma unroll
      for (int d = 0; d < DIM; ++d)
        cur_query[d] = query[query_offset + d];
    }

    __shared__ scalar_t key_buffer[BLOCK_SIZE * DIM];
    const int out_offset = (batch_idx * n1 + query_idx) * k;
    MinK<scalar_t, index_t> mink(distance + out_offset, index + out_offset, k);

    // Sweep over chunks of key data to find k nearest neighbors
    for (int chunk_idx2 = 0; chunk_idx2 < n_chunks; ++chunk_idx2)
    {
      // Load a chunk of key data into shared memory within the block
      const int key_idx_t = (chunk_idx2 * BLOCK_SIZE) + tid;
      const int key_offset = (batch_idx * n2 + key_idx_t) * DIM;
      if (key_idx_t < n2)
      {
#pragma unroll
        for (int d = 0; d < DIM; ++d)
          key_buffer[tid * DIM + d] = key[key_offset + d];
      }
      __syncthreads();

      if (query_idx < n1)
      {
        // Compare the current query point and all key points in the shared memory
        for (int i = 0; i < BLOCK_SIZE; ++i)
        {
          const int key_idx_i = (chunk_idx2 * BLOCK_SIZE) + i;

          // Compute the distance
          scalar_t dist = 0.0;
#pragma unroll
          for (int d = 0; d < DIM; ++d)
          {
            scalar_t diff = key_buffer[i * DIM + d] - cur_query[d];
            dist += diff * diff;
          }

          if (key_idx_i < n2)
          {
            mink.add(dist, key_idx_i);
          }
        }
      }
      __syncthreads();
    }
  }
}

/** Compute K nearest neighbors
 * @param query [B, N1, 3]
 * @param key [B, N2, 3]
 * @param k int
 * @param version 0 for general k with global memory to maintain top k; 1 for fixed k with local memory.
 * @return distance [B, N1, K]
 * @return index [B, N1, K]
 */
std::vector<at::Tensor> knn_distance_cuda(
    const at::Tensor query,
    const at::Tensor key,
    const int64_t k,
    const int version = 0)
{
  // sanity check
  CHECK_CONTIGUOUS_CUDA(query);
  CHECK_CONTIGUOUS_CUDA(key);
  TORCH_CHECK_EQ(query.dim(), 3);
  TORCH_CHECK_EQ(query.size(2), 3);
  TORCH_CHECK_EQ(key.dim(), 3);
  TORCH_CHECK_EQ(key.size(0), query.size(0));
  TORCH_CHECK_GE(key.size(1), k);
  TORCH_CHECK_EQ(key.size(2), 3);

  const auto bs = query.size(0);
  const auto n1 = query.size(1);
  const auto n2 = key.size(1);

  auto distance = at::zeros({bs, n1, k}, query.options());
  auto index = at::zeros({bs, n1, k}, query.options().dtype(at::kLong));

  // Calculate grids and blocks for kernels
  // It seems to be faster to use small block size
  const int MAX_THREADS_PER_BLOCK = 64;
  const auto n_threads = getBlockSize(std::min(n1, n2), MAX_THREADS_PER_BLOCK);
  const auto n_chunks = (n1 + n_threads - 1) / n_threads;
  dim3 grid;
  // const auto curDevice = at::cuda::current_device();
  getGrid(bs * n_chunks, grid, query.get_device());

#define RUN_K(BLOCK_SIZE, K)                                          \
  AT_DISPATCH_FLOATING_TYPES(                                         \
      query.scalar_type(),                                            \
      "knn_distance_cuda",                                            \
      ([&] { knn_distance_kernel<BLOCK_SIZE, K, 3, scalar_t, int64_t> \
                 <<<grid, BLOCK_SIZE>>>(                              \
                     distance.data_ptr<scalar_t>(),                   \
                     index.data_ptr<int64_t>(),                       \
                     query.data_ptr<scalar_t>(),                      \
                     key.data_ptr<scalar_t>(),                        \
                     bs,                                              \
                     n1,                                              \
                     n2); }));

#define RUN(BLOCK_SIZE)                                            \
  AT_DISPATCH_FLOATING_TYPES(                                      \
      query.scalar_type(),                                         \
      "knn_distance_cuda",                                         \
      ([&] { knn_distance_kernel<BLOCK_SIZE, 3, scalar_t, int64_t> \
                 <<<grid, BLOCK_SIZE>>>(                           \
                     distance.data_ptr<scalar_t>(),                \
                     index.data_ptr<int64_t>(),                    \
                     query.data_ptr<scalar_t>(),                   \
                     key.data_ptr<scalar_t>(),                     \
                     bs,                                           \
                     n1,                                           \
                     n2,                                           \
                     k); }));

// Dispatch to different k
#define DISPATCH_K(BLOCK_SIZE) \
  switch (k)                   \
  {                            \
  case 3:                      \
    RUN_K(BLOCK_SIZE, 3)       \
    break;                     \
  case 32:                     \
    RUN_K(BLOCK_SIZE, 32)      \
    break;                     \
  case 64:                     \
    RUN_K(BLOCK_SIZE, 64)      \
    break;                     \
  default:                     \
    RUN(BLOCK_SIZE)            \
  }

#define DISPATCH_VERSION(BLOCK_SIZE)       \
  switch (version)                         \
  {                                        \
  case 0:                                  \
    RUN(BLOCK_SIZE)                        \  
      break;                               \
  case 1:                                  \
    DISPATCH_K(BLOCK_SIZE)                 \
    break;                                 \
  default:                                 \
    TORCH_CHECK(false, "Invalid version"); \
  }

  switch (n_threads)
  {
  // case 512:
  //   DISPATCH_VERSION(512)
  //   break;
  // case 256:
  //   DISPATCH_VERSION(256)
  //   break;
  // case 128:
  //   DISPATCH_VERSION(128)
  //   break;
  case 64:
    DISPATCH_VERSION(64)
    break;
  default:
    DISPATCH_VERSION(32)
  }

  AT_CUDA_CHECK(cudaGetLastError());
  return std::vector<at::Tensor>({distance, index});
}