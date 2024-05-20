// Chamfer distance kernel

#include <cmath>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include "utils.h"

template <unsigned int BLOCK_SIZE>
__global__ void mask_iou_kernel(
    float *__restrict__ iou,        // [M1, M2]
    const bool *__restrict__ mask1, // [M1, N]
    const bool *__restrict__ mask2, // [M2, N]
    const int m1,
    const int m2,
    const int n)
{
    const int tid = threadIdx.x;
    const int total_blocks = m1 * m2;

    for (int block_idx = blockIdx.x; block_idx < total_blocks; block_idx += gridDim.x)
    {
        const int i = block_idx / m2;
        const int j = block_idx % m2;

        const int offset1 = i * n;
        const int offset2 = j * n;

        int t_intersection = 0;
        int t_union = 0;

        for (int k = tid; k < n; k += blockDim.x)
        {
            bool x1 = mask1[offset1 + k];
            bool x2 = mask2[offset2 + k];
            t_intersection += x1 && x2;
            t_union += x1 || x2;
        }

        __shared__ int buffer_intersection[BLOCK_SIZE];
        __shared__ int buffer_union[BLOCK_SIZE];

        buffer_intersection[tid] = t_intersection;
        buffer_union[tid] = t_union;
        __syncthreads();

        for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1)
        {
            if (tid < stride)
            {
                buffer_intersection[tid] += buffer_intersection[tid + stride];
                buffer_union[tid] += buffer_union[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0)
        {
            iou[block_idx] = (float)buffer_intersection[0] / (float)max(buffer_union[0], 1);
        }
    }
}

at::Tensor mask_iou_cuda(
    const at::Tensor mask1, // [M1, N]
    const at::Tensor mask2  // [M2, N]
)
{
    // Sanity check
    CHECK_CONTIGUOUS_CUDA(mask1);
    CHECK_CONTIGUOUS_CUDA(mask2);
    TORCH_CHECK_EQ(mask1.size(1), mask2.size(1));

    const auto m1 = mask1.size(0);
    const auto m2 = mask2.size(0);
    const auto n = mask1.size(1);

    // Outputs
    auto iou = at::zeros({m1, m2}, mask1.options().dtype(at::kFloat));

    const int BLOCK_SIZE = 256;
    dim3 grid;
    getGrid(m1 * m2, grid, mask1.get_device());

    // printf("(%ld, %ld, %ld, %ld)\n", m1, m2, n, grid.x);

    mask_iou_kernel<BLOCK_SIZE>
        <<<grid, BLOCK_SIZE>>>(
            iou.data_ptr<float>(),
            mask1.data_ptr<bool>(),
            mask2.data_ptr<bool>(),
            m1, m2, n);

    AT_CUDA_CHECK(cudaGetLastError());
    AT_CUDA_CHECK(cudaDeviceSynchronize());

    return iou;
}
