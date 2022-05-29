// CUDA Implementation for ball query.

#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include "torkit3d_utils.h"

// Advanced memory loading
// Each block processes a chunk of `index`, with each thread for one elements.
// Each block loads a chunk of `key` into the shared memory to compute distances.
template <unsigned int BLOCK_SIZE, unsigned int DIM, typename scalar_t, typename index_t>
__global__ void ball_query_kernel(
    index_t *__restrict__ index,        // [B, N1, K]
    const scalar_t *__restrict__ query, // [B, N1, D]
    const scalar_t *__restrict__ key,   // [B, N2, D]
    const int bs,
    const int n1,
    const int n2,
    const scalar_t r,
    const int k)
{
  const int n_blocks = (n1 + BLOCK_SIZE - 1) / BLOCK_SIZE; // number of thread blocks for query and index
  const int n_chunks = (n2 + BLOCK_SIZE - 1) / BLOCK_SIZE; // number of data chunks for key
  const int total_blocks = bs * n_blocks;
  const int tid = threadIdx.x;
  const scalar_t r2 = r * r;

  for (int block_idx = blockIdx.x; block_idx < total_blocks; block_idx += gridDim.x)
  {
    const int batch_idx = block_idx / n_blocks;
    const int chunk_idx1 = block_idx % n_blocks;
    const int query_idx = (chunk_idx1 * BLOCK_SIZE) + tid;
    const int query_offset = (batch_idx * n1 + query_idx) * DIM;
    const int index_offset = (batch_idx * n1 + query_idx) * k;

    __shared__ scalar_t key_buffer[BLOCK_SIZE * DIM]; // buffer to store the chunk of key
    index_t nbr_idx = -1;                             // neighbor index in key
    int nbr_cnt = 0;                                  // number of found neighbors

    // Load current query data
    scalar_t cur_query[DIM] = {0.0};
    if (query_idx < n1)
    {
#pragma unroll
      for (int d = 0; d < DIM; ++d)
        cur_query[d] = query[query_offset + d];
    }

    // Sweep over chunks of key data to find neighbors within the radius
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

          // Find a neighbors
          if (key_idx_i < n2 && nbr_cnt < k && dist < r2)
          {
            index[index_offset + nbr_cnt] = key_idx_i;
            // nbr_idx = nbr_cnt == 0 ? key_idx_i : nbr_idx;
            ++nbr_cnt;
          }
        }
      }

      // Sync before loading next chunk
      __syncthreads();
    }

    // Pad if not enough
    if (query_idx < n1 && nbr_cnt < k)
    {
      nbr_idx = index[index_offset]; // pad with first element
      for (int i = nbr_cnt; i < k; ++i)
      {
        index[index_offset + i] = nbr_idx;
      }
    }
  }
}

at::Tensor ball_query_cuda(
    const at::Tensor query, // [B, N1, 3]
    const at::Tensor key,   // [B, N2, 3]
    const float radius,
    const int64_t max_neighbors)
{
  // Sanity check
  CHECK_CONTIGUOUS_CUDA(query);
  CHECK_CONTIGUOUS_CUDA(key);
  CHECK_EQ(query.size(0), key.size(0));
  CHECK_EQ(query.size(2), 3);
  CHECK_EQ(key.size(2), 3);

  const auto batch_size = query.size(0);
  const auto n1 = query.size(1);
  const auto n2 = key.size(1);

  // [B, N1, K], neighborhood indices of each query point in key points
  auto index = at::full({batch_size, n1, max_neighbors}, -1, query.options().dtype(at::kLong));
  index.set_requires_grad(false);

  // Calculate grids and blocks for kernels
  const int MAX_THREADS_PER_BLOCK = 512;
  // TOOD(jigu): find divisble power of 2
  const auto n_threads = getBlock(std::min(n1, n2), MAX_THREADS_PER_BLOCK);
  const auto n_chunks = (n1 + n_threads - 1) / n_threads;
  dim3 grid;
  getGrid(batch_size * n_chunks, grid, query.get_device());
  // printf("%d,%d,%d\n", grid.x, n_chunks, n_threads);

#define RUN(BLOCK_SIZE)                                          \
  AT_DISPATCH_FLOATING_TYPES(                                    \
      query.scalar_type(),                                       \
      "ball_query_cuda",                                         \
      ([&] { ball_query_kernel<BLOCK_SIZE, 3, scalar_t, int64_t> \
                 <<<grid, BLOCK_SIZE>>>(                         \
                     index.data_ptr<int64_t>(),                  \
                     query.data_ptr<scalar_t>(),                 \
                     key.data_ptr<scalar_t>(),                   \
                     batch_size,                                 \
                     n1,                                         \
                     n2,                                         \
                     radius,                                     \
                     max_neighbors); }));

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
  return index;
}