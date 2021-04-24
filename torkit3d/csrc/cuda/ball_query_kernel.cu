// CUDA Implementation for ball query.
// AT_ASSERT has become AT_CHECK on master after 0.4.
// AT_CHECK has become TORCH_CHECK on master after 1.2.

#include <algorithm>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAApplyUtils.cuh> // at::cuda::getApplyGrid

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

#define MAX_THREADS 512

inline int opt_n_threads(int work_size)
{
  const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);
  return std::max(std::min(1 << pow_2, MAX_THREADS), 1);
}

// From getApplyGrid: aten/src/ATen/cuda/CUDAApplyUtils.cuh
inline bool getGrid(uint64_t numBlocks, dim3 &grid, int64_t curDevice)
{
  if (curDevice == -1)
    return false;
  uint64_t maxGridX = at::cuda::getDeviceProperties(curDevice)->maxGridSize[0];
  if (numBlocks > maxGridX)
    numBlocks = maxGridX;
  grid = dim3(numBlocks);
  return true;
}

// Load a block of data to make full use of GPU
template <unsigned int BLOCK_SIZE, unsigned int DIM, typename scalar_t, typename index_t>
__global__ void ball_query_kernel(
    index_t *__restrict__ index,
    const scalar_t *__restrict__ query,
    const scalar_t *__restrict__ key,
    const int64_t batch_size,
    const int64_t n1,
    const int64_t n2,
    const scalar_t radius,
    const int64_t max_neighbors)
{

  // calculate the number of blocks
  const int n_blocks1 = (n1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const int n_blocks2 = (n2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const int total_blocks = batch_size * n_blocks1;
  const scalar_t radius_square = radius * radius;

  for (int block_idx = blockIdx.x; block_idx < total_blocks; block_idx += gridDim.x)
  {
    __shared__ scalar_t key_buffer[BLOCK_SIZE * DIM];
    const int batch_idx = block_idx / n_blocks1;
    const int block_idx1 = block_idx % n_blocks1;
    const int query_idx = (block_idx1 * BLOCK_SIZE) + threadIdx.x;
    const int query_offset = (batch_idx * n1 + query_idx) * DIM;

    // load current query point
    scalar_t cur_query[DIM] = {0.0};
    if (query_idx < n1)
    {
#pragma unroll
      for (int i = 0; i < DIM; ++i)
      {
        cur_query[i] = query[query_offset + i];
      }
    }

    index_t cnt_neighbors = 0;
    const int index_offset = batch_idx * n1 * max_neighbors + query_idx * max_neighbors;
    // load a block of key data to reduce the time to read data
    for (int block_idx2 = 0; block_idx2 < n_blocks2; ++block_idx2)
    {
      // load key data
      int key_idx = (block_idx2 * BLOCK_SIZE) + threadIdx.x;
      int key_offset = (batch_idx * n2 + key_idx) * DIM;
      if (key_idx < n2)
      {
#pragma unroll
        for (int i = 0; i < DIM; ++i)
        {
          key_buffer[threadIdx.x * DIM + i] = key[key_offset + i];
        }
      }
      __syncthreads();

      // calculate the distance between current query and key, with the shared memory.
      if (query_idx < n1)
      {
        for (int j = 0; j < BLOCK_SIZE; ++j)
        {
          int key_idx2 = (block_idx2 * BLOCK_SIZE) + j;
          const int buffer_offset = j * DIM;
          scalar_t dist = 0.0;
#pragma unroll
          for (int i = 0; i < DIM; ++i)
          {
            scalar_t diff = key_buffer[buffer_offset + i] - cur_query[i];
            dist += diff * diff;
          }
          if (key_idx2 < n2 && cnt_neighbors < max_neighbors)
          {
            if (dist < radius_square)
            {
              index[index_offset + cnt_neighbors] = key_idx2;
              ++cnt_neighbors;
            }
          }
        }
      }
      __syncthreads();
    }
    // pad with the first term if necessary
    if (query_idx < n1 && cnt_neighbors < max_neighbors)
    {
      index_t pad_val = index[index_offset];
      for (int j = cnt_neighbors; j < max_neighbors; ++j)
      {
        index[index_offset + j] = pad_val;
      }
    }
  }
}

/**
 * Ball query
 * Input:
 *  query: [B, N1, 3]
 *  key: [B, N2, 3]
 *  radius: float
 *  max_neighbors: int
 * Output:
 *  index: [B, N1, K]
 **/
at::Tensor ball_query_cuda(
    const at::Tensor query,
    const at::Tensor key,
    const float radius,
    const int64_t max_neighbors)
{

  // Sanity check
  CHECK_INPUT(query);
  CHECK_INPUT(key);
  CHECK_EQ(query.size(0), key.size(0));
  CHECK_EQ(query.size(2), 3);
  CHECK_EQ(key.size(2), 3);

  const auto batch_size = query.size(0);
  const auto n1 = query.size(1);
  const auto n2 = key.size(1);

  // Allocate new space for output
  auto index = at::full({batch_size, n1, max_neighbors}, -1, query.options().dtype(at::kLong));
  index.set_requires_grad(false);

  // Calculate grids and blocks for kernels
  const auto n_threads = opt_n_threads(std::min(n1, n2));
  const auto n_blocks = (n1 + n_threads - 1) / n_threads;
  dim3 grid;
  const auto curDevice = at::cuda::current_device();
  getGrid(batch_size * n_blocks, grid, curDevice);

#define RUN(BLOCK_SIZE)                                                                    \
  AT_DISPATCH_FLOATING_TYPES(query.scalar_type(), "ball_query_cuda", ([&] {               \
                               ball_query_kernel<BLOCK_SIZE, 3, scalar_t, int64_t> \
                                   <<<grid, BLOCK_SIZE>>>(                                 \
                                       index.data_ptr<int64_t>(),                          \
                                       query.data_ptr<scalar_t>(),                         \
                                       key.data_ptr<scalar_t>(),                           \
                                       batch_size,                                         \
                                       n1,                                                 \
                                       n2,                                                 \
                                       radius,                                             \
                                       max_neighbors);                                     \
                             }));

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
  default:
    RUN(16)
  }

  THCudaCheck(cudaGetLastError());

  return index;
}