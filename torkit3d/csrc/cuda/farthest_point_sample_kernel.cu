// CUDA Implementation for farthest point sampling.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include "utils.h"

// Each block of threads process one point cloud.
template <unsigned int BLOCK_SIZE, unsigned int DIM, typename scalar_t, typename index_t>
__global__ void farthest_point_sample_kernel(
    index_t *__restrict__ index,         // [B, K]
    const scalar_t *__restrict__ points, // [B, N, D]
    scalar_t *__restrict__ min_dist,     // [B, N]
    const int n,
    const int k)
{
  // Shared memory for max distance
  __shared__ scalar_t smem_dist[BLOCK_SIZE];
  // Shared memory for max distance index
  __shared__ int smem_idx[BLOCK_SIZE];

  const int batch_idx = blockIdx.x;
  const int tid = threadIdx.x;
  points = points + batch_idx * n * DIM;
  min_dist = min_dist + batch_idx * n;
  index = index + batch_idx * k;

  // Select the first point
  int cur_idx = 0;
  if (tid == 0)
    index[0] = cur_idx;

  // Iterate to find the next farthest point
  for (int i = 1; i < k; ++i)
  {
    scalar_t p[DIM];         // last selected point
    scalar_t max_dist = 0.0; // max distance to the current point
    int max_idx = cur_idx;   // corresponding index

    // Load last selected point
#pragma unroll
    for (int d = 0; d < DIM; ++d)
      p[d] = points[cur_idx * DIM + d];

    // Find the farthest point with parallel reduction
    for (int j = tid; j < n; j += BLOCK_SIZE)
    {
      scalar_t dist = 0.0;

      // Compute the distance
#pragma unroll
      for (int d = 0; d < DIM; ++d)
      {
        scalar_t diff = points[j * DIM + d] - p[d];
        dist += diff * diff;
      }

      // Update its (minimum) distance to all selected points
      scalar_t min_dist_j = min_dist[j];
      if (min_dist_j > dist || min_dist_j < 0.0)
        min_dist[j] = dist;
      else
        dist = min_dist_j;

      // Update the farthest distance
      if (dist > max_dist)
      {
        max_dist = dist;
        max_idx = j;
      }
    }

    // Load per-thread max into shared memory
    smem_dist[tid] = max_dist;
    smem_idx[tid] = max_idx;
    __syncthreads();

    // assert BLOCK_SIZE == blockDim.x
    // Reduce max
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1)
    {
      if (tid < s)
      {
        scalar_t dist1 = smem_dist[tid];
        scalar_t dist2 = smem_dist[tid + s];
        if (dist1 < dist2)
        {
          smem_dist[tid] = dist2;
          smem_idx[tid] = smem_idx[tid + s];
        }
      }
      __syncthreads();
    }

    cur_idx = smem_idx[0];
    if (tid == 0)
      index[i] = (index_t)cur_idx;
  }
}

at::Tensor farthest_point_sample_cuda(
    const at::Tensor points, // [B, N, 3]
    const int64_t num_samples)
{
  // Sanity check
  CHECK_CONTIGUOUS_CUDA(points);
  TORCH_CHECK_EQ(points.dim(), 3);
  TORCH_CHECK_EQ(points.size(2), 3);
  TORCH_CHECK_GT(num_samples, 0);
  TORCH_CHECK_GE(points.size(1), num_samples);

  const auto batch_size = points.size(0);
  const auto num_points = points.size(1);

  // [B, N]
  auto index = at::full({batch_size, num_samples}, -1, points.options().dtype(at::kLong));
  index.set_requires_grad(false);

  // NOTE(jigu): Different from the original implementation,
  // which only allocates memory with the size of grid instead of batch size.
  // Store the point-wise (minimum) distance to selected points.
  auto min_dist = at::full({batch_size, num_points}, -1.0, points.options());

  // In order to make full use of shared memory and threads,
  // the number of points should be a power of 2.
  // NOTE(jigu): 512 seems to be faster than 1024 and 256.
  const int MAX_THREADS_PER_BLOCK = 512;
  const auto n_threads = getBlockSize(num_points, MAX_THREADS_PER_BLOCK);
  // printf("n_threads=%d\n", n_threads);

#define RUN(BLOCK_SIZE)                                                     \
  AT_DISPATCH_FLOATING_TYPES(                                               \
      points.scalar_type(),                                                 \
      "farthest_point_sample_cuda",                                         \
      ([&] { farthest_point_sample_kernel<BLOCK_SIZE, 3, scalar_t, int64_t> \
                 <<<batch_size, BLOCK_SIZE>>>(                              \
                     index.data_ptr<int64_t>(),                             \
                     points.data_ptr<scalar_t>(),                           \
                     min_dist.data_ptr<scalar_t>(),                         \
                     num_points,                                            \
                     num_samples); }));

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