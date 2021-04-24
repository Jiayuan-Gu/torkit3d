// CUDA Implementation for KNN with distance.

#include <algorithm>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

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

/****************************
* Kernel for searching point
*****************************/
template <unsigned int BLOCK_SIZE, unsigned int K, unsigned int DIM, typename scalar_t, typename index_t>
__global__ void knn_distance_kernel(
    index_t *__restrict__ index,
    scalar_t *__restrict__ distance,
    const scalar_t *__restrict__ query,
    const scalar_t *__restrict__ key,
    const int64_t batch_size,
    const int64_t num_query,
    const int64_t num_key)
{

  // calculate the number of blocks
  const int n_blocks1 = (num_query + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const int n_blocks2 = (num_key + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const int total_blocks = batch_size * n_blocks1;

  for (int block_idx = blockIdx.x; block_idx < total_blocks; block_idx += gridDim.x)
  {
    __shared__ scalar_t key_buffer[BLOCK_SIZE * DIM];
    const int batch_idx = block_idx / n_blocks1;
    const int block_idx1 = block_idx % n_blocks1;
    const int query_idx = (block_idx1 * BLOCK_SIZE) + threadIdx.x;
    const int query_offset = (batch_idx * num_query + query_idx) * DIM;

    // load current query point
    scalar_t cur_query[DIM] = {0.0};
    if (query_idx < num_query)
    {
#pragma unroll
      for (int i = 0; i < DIM; ++i)
      {
        cur_query[i] = query[query_offset + i];
      }
    }

    // record topk
    scalar_t min_dist[K] = {1e40};
    int min_idx[K] = {-1};

    // load a block of key data to reduce the time to read data
    for (int block_idx2 = 0; block_idx2 < n_blocks2; ++block_idx2)
    {
      // load key data
      int key_idx = (block_idx2 * BLOCK_SIZE) + threadIdx.x;
      int key_offset = (batch_idx * num_key + key_idx) * DIM;
      if (key_idx < num_key)
      {
#pragma unroll
        for (int i = 0; i < DIM; ++i)
        {
          key_buffer[threadIdx.x * DIM + i] = key[key_offset + i];
        }
      }
      __syncthreads();

      // calculate the distance between current query and key, with the shared memory.
      if (query_idx < num_query)
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
          if (key_idx2 < num_key)
          {
// update min distance
#pragma unroll
            for (int k = 0; k < K; ++k)
            {
              if (dist < min_dist[k])
              {
                for (int l = K - 1; l > k; --l)
                {
                  min_dist[l] = min_dist[l - 1];
                  min_idx[l] = min_idx[l - 1];
                }
                min_dist[k] = dist;
                min_idx[k] = key_idx2;
                break;
              }
            }
          }
        }
      }
      __syncthreads();
    }

    // output
    const int out_offset = (batch_idx * num_query + query_idx) * K;
    if (query_idx < num_query)
    {
#pragma unroll
      for (int k = 0; k < K; ++k)
      {
        index[out_offset + k] = min_idx[k];
        distance[out_offset + k] = min_dist[k];
      }
    }
  }
}

/**
 * Forward
 * Input:
 *  query: [B, N1, 3]
 *  key: [B, N2, 3]
 *  k: int
 * Output:
 *  index: [B, N1, K]
 *  distance: [B, N1, K]
 **/
std::vector<at::Tensor> knn_distance_cuda(
    const at::Tensor query,
    const at::Tensor key,
    const int64_t k)
{

  // sanity check
  CHECK_INPUT(query);
  CHECK_INPUT(key);
  CHECK_EQ(query.dim(), 3);
  CHECK_EQ(key.dim(), 3);
  CHECK_EQ(query.size(0), key.size(0));
  CHECK_EQ(query.size(2), 3);
  CHECK_GE(key.size(1), k);
  CHECK_EQ(key.size(2), 3);
  TORCH_CHECK(k == 3, "Only support 3-NN.");

  const auto batch_size = query.size(0);
  const auto num_query = query.size(1);
  const auto dim = query.size(2);
  const auto num_key = key.size(1);

  auto index = at::zeros({batch_size, num_query, k}, query.options().dtype(at::kLong));
  auto distance = at::zeros({batch_size, num_query, k}, query.options());

  // Calculate grids and blocks for kernels
  const auto n_threads = opt_n_threads(std::min(num_query, num_key));
  const auto n_blocks = (num_query + n_threads - 1) / n_threads;
  dim3 grid;
  const auto curDevice = at::cuda::current_device();
  getGrid(batch_size * n_blocks, grid, curDevice);

#define RUN(BLOCK_SIZE)                                                               \
  AT_DISPATCH_FLOATING_TYPES(query.scalar_type(), "knn_distance_cuda", ([&] {         \
                               knn_distance_kernel<BLOCK_SIZE, 3, 3, scalar_t, int64_t> \
                                   <<<grid, BLOCK_SIZE>>>(                            \
                                       index.data_ptr<int64_t>(),                     \
                                       distance.data_ptr<scalar_t>(),                 \
                                       query.data_ptr<scalar_t>(),                    \
                                       key.data_ptr<scalar_t>(),                      \
                                       batch_size,                                    \
                                       num_query,                                     \
                                       num_key);                                      \
                             }));

#define CASE(BLOCK_SIZE) \
  case BLOCK_SIZE:           \
    RUN(BLOCK_SIZE)          \
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

  return std::vector<at::Tensor>({index, distance});
}