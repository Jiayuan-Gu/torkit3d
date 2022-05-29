// CUDA Implementation for KNN with distance.

#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/ApplyGridUtils.cuh>
#include "torkit3d_utils.h"

template <unsigned int BLOCK_SIZE, unsigned int K, unsigned int DIM, typename scalar_t, typename index_t>
__global__ void knn_distance_kernel(
    index_t *__restrict__ index,        // [B, N1, K]
    scalar_t *__restrict__ distance,    // [B, N1, K]
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
    scalar_t mink_dist[K] = {1e40}; // top K nearest distance
    int mink_idx[K] = {-1};         // top K nearest indices

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
            // Update k-nn
#pragma unroll
            for (int k = 0; k < K; ++k)
            {
              if (dist < mink_dist[k])
              {
                // bubble sort
                for (int j = K - 1; j > k; --j)
                {
                  mink_dist[j] = mink_dist[j - 1];
                  mink_idx[j] = mink_idx[j - 1];
                }
                mink_dist[k] = dist;
                mink_idx[k] = key_idx_i;
                break;
              }
            }
          }
        }
      }
      __syncthreads();
    }

    // Output
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

std::vector<at::Tensor> knn_distance_cuda(
    const at::Tensor query, // [B, N1, 3]
    const at::Tensor key,   // [B, N2, 3]
    const int64_t k)
{
  // sanity check
  CHECK_CONTIGUOUS_CUDA(query);
  CHECK_CONTIGUOUS_CUDA(key);
  CHECK_EQ(query.dim(), 3);
  CHECK_EQ(key.dim(), 3);
  CHECK_EQ(query.size(0), key.size(0));
  CHECK_EQ(query.size(2), 3);
  CHECK_GE(key.size(1), k);
  CHECK_EQ(key.size(2), 3);
  TORCH_CHECK(k == 3, "only support 3-NN");

  const auto bs = query.size(0);
  const auto n1 = query.size(1);
  const auto n2 = key.size(1);

  auto index = at::zeros({bs, n1, k}, query.options().dtype(at::kLong));
  auto distance = at::zeros({bs, n1, k}, query.options());

  // Calculate grids and blocks for kernels
  const int MAX_THREADS_PER_BLOCK = 512;
  const auto n_threads = getBlock(std::min(n1, n2), MAX_THREADS_PER_BLOCK);
  const auto n_chunks = (n1 + n_threads - 1) / n_threads;
  dim3 grid;
  // const auto curDevice = at::cuda::current_device();
  getGrid(bs * n_chunks, grid, query.get_device());

#define RUN(BLOCK_SIZE)                                               \
  AT_DISPATCH_FLOATING_TYPES(                                         \
      query.scalar_type(),                                            \
      "knn_distance_cuda",                                            \
      ([&] { knn_distance_kernel<BLOCK_SIZE, 3, 3, scalar_t, int64_t> \
                 <<<grid, BLOCK_SIZE>>>(                              \
                     index.data_ptr<int64_t>(),                       \
                     distance.data_ptr<scalar_t>(),                       \
                     query.data_ptr<scalar_t>(),                      \
                     key.data_ptr<scalar_t>(),                        \
                     bs,                                              \
                     n1,                                              \
                     n2); }));

#define CASE(BLOCK_SIZE) \
  case BLOCK_SIZE:       \
    RUN(BLOCK_SIZE)      \
    break;

  switch (n_threads)
  {
    CASE(512)
    CASE(256)
    CASE(128)
    CASE(64)
    CASE(32)
    CASE(16)
    CASE(8)
    CASE(4)
    CASE(2)
    CASE(1)
  default:
    TORCH_CHECK(false, "Invalid case!");
  }

  AT_CUDA_CHECK(cudaGetLastError());
  return std::vector<at::Tensor>({index, distance});
}