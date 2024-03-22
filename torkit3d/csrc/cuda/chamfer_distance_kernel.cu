// Chamfer distance kernel

#include <cmath>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/ApplyGridUtils.cuh> // at::cuda::getApplyGrid
#include "utils.h"

template <unsigned int BLOCK_SIZE, unsigned int DIM, typename scalar_t, typename index_t>
__global__ void chamfer_distance_forward_kernel(
    scalar_t *__restrict__ dist,       // [B, N1]
    index_t *__restrict__ idx,         // [B, N1]
    const scalar_t *__restrict__ xyz1, // [B, N1, D]
    const scalar_t *__restrict__ xyz2, // [B, N2, D]
    const int bs,
    const int n1,
    const int n2)
{
    const int n_chunks1 = (n1 + BLOCK_SIZE - 1) / BLOCK_SIZE; // number of chunks in xyz1
    const int n_chunks2 = (n2 + BLOCK_SIZE - 1) / BLOCK_SIZE; // number of chunks in xyz2
    const int total_blocks = bs * n_chunks1;
    const int tid = threadIdx.x;

    for (int block_idx = blockIdx.x; block_idx < total_blocks; block_idx += gridDim.x)
    {
        const int batch_idx = block_idx / n_chunks1;
        const int chunk_idx1 = block_idx % n_chunks1;
        const int xyz1_idx = (chunk_idx1 * BLOCK_SIZE) + tid;
        const int xyz1_offset = (batch_idx * n1 + xyz1_idx) * DIM;

        // Load current xyz1 point
        scalar_t cur_xyz1[DIM] = {0.0};
        if (xyz1_idx < n1)
        {
#pragma unroll
            for (int d = 0; d < DIM; ++d)
                cur_xyz1[d] = xyz1[xyz1_offset + d];
        }

        __shared__ scalar_t xyz2_buffer[BLOCK_SIZE * DIM];
        scalar_t min_dist = 1e40;
        index_t min_idx = -1;

        // Sweep over chunks of xyz2 data to find closest point
        for (int chunk_idx2 = 0; chunk_idx2 < n_chunks2; ++chunk_idx2)
        {
            // Load a chunk of xyz2 data into shared memory
            int xyz2_idx = (chunk_idx2 * BLOCK_SIZE) + tid;
            int xyz2_offset = (batch_idx * n2 + xyz2_idx) * DIM;
            if (xyz2_idx < n2)
            {
#pragma unroll
                for (int d = 0; d < DIM; ++d)
                    xyz2_buffer[tid * DIM + d] = xyz2[xyz2_offset + d];
            }
            __syncthreads();

            // Compare current xyz1 point with all points in shared memory
            for (int i = 0; i < BLOCK_SIZE; ++i)
            {
                xyz2_idx = (chunk_idx2 * BLOCK_SIZE) + i;

                // Compute the distance
                scalar_t dist = 0.0;
#pragma unroll
                for (int d = 0; d < DIM; ++d)
                {
                    scalar_t diff = xyz2_buffer[i * DIM + d] - cur_xyz1[d];
                    dist += diff * diff;
                }

                if (xyz2_idx < n2 && dist < min_dist)
                {
                    min_dist = dist;
                    min_idx = xyz2_idx;
                }
            }
            __syncthreads();
        }

        if (xyz1_idx < n1)
        {
            const int output_offset = batch_idx * n1 + xyz1_idx;
            dist[output_offset] = min_dist;
            idx[output_offset] = min_idx;
        }
    }
}

std::vector<at::Tensor> chamfer_distance_forward_cuda(
    const at::Tensor xyz1, // [B, N1, 3]
    const at::Tensor xyz2  // [B, N2, 3]
)
{
    // Sanity check
    CHECK_CONTIGUOUS_CUDA(xyz1);
    CHECK_CONTIGUOUS_CUDA(xyz2);
    TORCH_CHECK_EQ(xyz1.size(0), xyz2.size(0));
    TORCH_CHECK_EQ(xyz1.size(2), 3);
    TORCH_CHECK_EQ(xyz2.size(2), 3);

    const auto bs = xyz1.size(0);
    const auto n1 = xyz1.size(1);
    const auto n2 = xyz2.size(1);

    // Outputs
    auto dist1 = at::zeros({bs, n1}, xyz1.options());
    auto idx1 = at::zeros({bs, n1}, xyz1.options().dtype(at::kLong));
    auto dist2 = at::zeros({bs, n2}, xyz2.options());
    auto idx2 = at::zeros({bs, n2}, xyz2.options().dtype(at::kLong));

    // Calculate grids and blocks for kernels
    // TODO(jigu): dynamic block size
    const int BLOCK_SIZE = 256;
    const auto n_chunks1 = (n1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const auto n_chunks2 = (n2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid1, grid2;
    // const auto curDevice = at::cuda::current_device();
    const auto curDevice = xyz1.get_device();
    getGrid(bs * n_chunks1, grid1, curDevice);
    getGrid(bs * n_chunks2, grid2, curDevice);
    // printf("(%ld, %ld, %ld, %ld, %ld)\n", bs, n1, n2, n_chunks1, n_chunks2);

    AT_DISPATCH_FLOATING_TYPES(
        xyz1.scalar_type(),
        "chamfer_distance_forward_cuda",
        ([&]
         { chamfer_distance_forward_kernel<BLOCK_SIZE, 3, scalar_t, int64_t>
               <<<grid1, BLOCK_SIZE>>>(
                   dist1.data_ptr<scalar_t>(),
                   idx1.data_ptr<int64_t>(),
                   xyz1.data_ptr<scalar_t>(),
                   xyz2.data_ptr<scalar_t>(),
                   bs, n1, n2); }));

    AT_DISPATCH_FLOATING_TYPES(
        xyz2.scalar_type(),
        "chamfer_distance_forward_cuda",
        ([&]
         { chamfer_distance_forward_kernel<BLOCK_SIZE, 3, scalar_t, int64_t>
               <<<grid2, BLOCK_SIZE>>>(
                   dist2.data_ptr<scalar_t>(),
                   idx2.data_ptr<int64_t>(),
                   xyz2.data_ptr<scalar_t>(),
                   xyz1.data_ptr<scalar_t>(),
                   bs, n2, n1); }));

    AT_CUDA_CHECK(cudaGetLastError());
    return std::vector<at::Tensor>({dist1, idx1, dist2, idx2});
}

template <typename scalar_t, typename index_t>
__global__ void chamfer_distance_backward_kernel(
    scalar_t *__restrict__ grad_xyz1,       // [B, N1]
    scalar_t *__restrict__ grad_xyz2,       // [B, N2]
    const scalar_t *__restrict__ grad_dist, // [B, N1]
    const index_t *__restrict__ index,      // [B, N1]
    const scalar_t *__restrict__ xyz1,      // [B, N1, 3]
    const scalar_t *__restrict__ xyz2,      // [B, N2, 3]
    const int bs,
    const int n1,
    const int n2)
{
    const int totalElements = bs * n1;
    for (int linearId = blockIdx.x * blockDim.x + threadIdx.x;
         linearId < totalElements;
         linearId += gridDim.x * blockDim.x)
    {
        const int batch_idx = linearId / n1;

        const int xyz1_offset = linearId * 3;
        const scalar_t *xyz1_i = xyz1 + xyz1_offset;
        scalar_t x1 = xyz1_i[0];
        scalar_t y1 = xyz1_i[1];
        scalar_t z1 = xyz1_i[2];

        const int xyz2_offset = (batch_idx * n2 + index[linearId]) * 3;
        const scalar_t *xyz2_i = xyz2 + xyz2_offset;
        scalar_t x2 = xyz2_i[0];
        scalar_t y2 = xyz2_i[1];
        scalar_t z2 = xyz2_i[2];

        scalar_t g = grad_dist[linearId] * 2;
        scalar_t gx = g * (x1 - x2);
        scalar_t gy = g * (y1 - y2);
        scalar_t gz = g * (z1 - z2);

        scalar_t *grad_xyz1_i = grad_xyz1 + xyz1_offset;
        atomicAdd(grad_xyz1_i + 0, gx);
        atomicAdd(grad_xyz1_i + 1, gy);
        atomicAdd(grad_xyz1_i + 2, gz);
        scalar_t *grad_xyz2_i = grad_xyz2 + xyz2_offset;
        atomicAdd(grad_xyz2_i + 0, -gx);
        atomicAdd(grad_xyz2_i + 1, -gy);
        atomicAdd(grad_xyz2_i + 2, -gz);
    }
}

std::vector<at::Tensor> chamfer_distance_backward_cuda(
    const at::Tensor grad_dist1, // [B, N1]
    const at::Tensor grad_dist2, // [B, N2]
    const at::Tensor xyz1,       // [B, N, 3]
    const at::Tensor xyz2,       // [B, N2, 3]
    const at::Tensor idx1,       // [B, N1]
    const at::Tensor idx2        // [B, N2]
)
{
    CHECK_CONTIGUOUS_CUDA(grad_dist1);
    CHECK_CONTIGUOUS_CUDA(grad_dist2);
    CHECK_CONTIGUOUS_CUDA(xyz1);
    CHECK_CONTIGUOUS_CUDA(xyz2);
    CHECK_CONTIGUOUS_CUDA(idx1);
    CHECK_CONTIGUOUS_CUDA(idx2);

    const auto bs = grad_dist1.size(0);
    const auto n1 = grad_dist1.size(1);
    const auto n2 = grad_dist2.size(1);

    // Sanity check
    TORCH_CHECK_EQ(grad_dist2.size(0), bs);
    TORCH_CHECK_EQ(xyz1.size(0), bs);
    TORCH_CHECK_EQ(xyz2.size(0), bs);
    TORCH_CHECK_EQ(xyz1.size(1), n1);
    TORCH_CHECK_EQ(xyz2.size(1), n2);
    TORCH_CHECK_EQ(xyz1.size(2), 3);
    TORCH_CHECK_EQ(xyz2.size(2), 3);
    TORCH_CHECK_EQ(idx1.size(0), bs);
    TORCH_CHECK_EQ(idx2.size(0), bs);
    TORCH_CHECK_EQ(idx1.size(1), n1);
    TORCH_CHECK_EQ(idx2.size(1), n2);

    auto grad_xyz1 = at::zeros({bs, n1, 3}, grad_dist1.options());
    auto grad_xyz2 = at::zeros({bs, n2, 3}, grad_dist2.options());

    // Calculate grids and blocks for kernels
    const int BLOCK_SIZE = at::cuda::getApplyBlockSize();
    dim3 grid1, grid2;
    // const auto curDevice = at::cuda::current_device();
    const auto curDevice = grad_dist1.get_device();
    TORCH_CHECK(at::cuda::getApplyGrid(bs * n1, grid1, curDevice, BLOCK_SIZE), "unable to get grid");
    TORCH_CHECK(at::cuda::getApplyGrid(bs * n2, grid2, curDevice, BLOCK_SIZE), "unable to get grid");

    AT_DISPATCH_FLOATING_TYPES(
        grad_dist1.scalar_type(),
        "chamfer_distance_backward_cuda",
        ([&]
         { chamfer_distance_backward_kernel<scalar_t, int64_t>
               <<<grid1, BLOCK_SIZE>>>(
                   grad_xyz1.data_ptr<scalar_t>(),
                   grad_xyz2.data_ptr<scalar_t>(),
                   grad_dist1.data_ptr<scalar_t>(),
                   idx1.data_ptr<int64_t>(),
                   xyz1.data_ptr<scalar_t>(),
                   xyz2.data_ptr<scalar_t>(),
                   bs, n1, n2); }));

    AT_DISPATCH_FLOATING_TYPES(
        grad_dist2.scalar_type(),
        "chamfer_distance_backward_cuda",
        ([&]
         { chamfer_distance_backward_kernel<scalar_t, int64_t>
               <<<grid2, BLOCK_SIZE>>>(
                   grad_xyz2.data_ptr<scalar_t>(),
                   grad_xyz1.data_ptr<scalar_t>(),
                   grad_dist2.data_ptr<scalar_t>(),
                   idx2.data_ptr<int64_t>(),
                   xyz2.data_ptr<scalar_t>(),
                   xyz1.data_ptr<scalar_t>(),
                   bs, n2, n1); }));

    AT_CUDA_CHECK(cudaGetLastError());
    return std::vector<at::Tensor>({grad_xyz1, grad_xyz2});
}
