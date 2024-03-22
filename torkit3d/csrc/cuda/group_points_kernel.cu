// Memory efficient gather points

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/ApplyGridUtils.cuh> // at::cuda::getApplyGrid
#include "utils.h"

at::Tensor group_points_forward_cuda(
    const at::Tensor input, // [B, C, N1]
    const at::Tensor index  // [B, N2, K]
)
{
  // Sanity check
  CHECK_CUDA(input);
  CHECK_CUDA(index);
  TORCH_CHECK_EQ(input.dim(), 3);
  TORCH_CHECK_EQ(index.dim(), 3);
  TORCH_CHECK_EQ(input.size(0), index.size(0));

  const auto b = input.size(0);
  const auto c = input.size(1);
  const auto n1 = input.size(2);
  const auto n2 = index.size(1);
  const auto k = index.size(2);

  auto input_expand = input.unsqueeze(2).expand({b, c, n2, n1}); // [B, C, N2, N1]
  auto index_expand = index.unsqueeze(1).expand({b, c, n2, k});  // [B, C, N2, K]

  auto output = input_expand.gather(3, index_expand); // [B, C, N2, K]
  return output;
}

template <typename scalar_t, typename index_t>
__global__ void group_points_backward_kernel(
    scalar_t *grad_input,
    const scalar_t *__restrict__ grad_output,
    const index_t *__restrict__ index,
    const index_t b,
    const index_t c,
    const index_t n1,
    const index_t n2,
    const index_t k,
    const index_t totalElements)
{
  for (index_t linearId = blockIdx.x * blockDim.x + threadIdx.x;
       linearId < totalElements;
       linearId += gridDim.x * blockDim.x)
  {
    // Compute offsets
    index_t k_offset = linearId % k;
    index_t linearId1 = linearId / k;
    index_t n2_offset = linearId1 % n2;
    index_t linearId2 = linearId1 / n2;
    index_t c_offset = linearId2 % c;
    index_t b_offset = linearId2 / c;

    index_t srcOffset = k_offset + n2_offset * k + c_offset * (k * n2) + b_offset * (k * n2 * c);
    index_t dstOffset = c_offset * n1 + b_offset * (n1 * c);
    index_t indexOffset = k_offset + n2_offset * k + b_offset * (k * n2);

    index_t indexValue = index[indexOffset];
    assert(indexValue >= 0 && indexValue < n1);
    atomicAdd(&grad_input[dstOffset + indexValue], grad_output[srcOffset]);
  }
}

at::Tensor group_points_backward_cuda(
    const at::Tensor grad_output, // [B, C, N2, K]
    const at::Tensor index,       // [B, N2, K]
    const int64_t n1)
{
  // Sanity check
  CHECK_CONTIGUOUS_CUDA(grad_output);
  CHECK_CONTIGUOUS_CUDA(index);
  TORCH_CHECK_EQ(grad_output.dim(), 4);
  TORCH_CHECK_EQ(index.dim(), 3);
  TORCH_CHECK_EQ(index.size(0), grad_output.size(0));
  TORCH_CHECK_EQ(index.size(1), grad_output.size(2));
  TORCH_CHECK_EQ(index.size(2), grad_output.size(3));

  const auto b = grad_output.size(0);
  const auto c = grad_output.size(1);
  const auto n2 = grad_output.size(2);
  const auto k = grad_output.size(3);

  // Allocate output memory
  auto grad_input = at::zeros({b, c, n1}, grad_output.options()); // [B, C, N1]
  CHECK_CONTIGUOUS_CUDA(grad_input);

  // Calculate grids and blocks for kernels
  const auto totalElements = grad_output.numel();
  const int BLOCK_SIZE = at::cuda::getApplyBlockSize();
  dim3 grid;
  TORCH_CHECK(at::cuda::getApplyGrid(totalElements, grid, grad_output.get_device(), BLOCK_SIZE), "unable to get grid");

  AT_DISPATCH_FLOATING_TYPES(
      grad_output.scalar_type(),
      "group_points_backward_cuda",
      ([&]
       { group_points_backward_kernel<scalar_t, int64_t>
             <<<grid, BLOCK_SIZE>>>(
                 grad_input.data_ptr<scalar_t>(),
                 grad_output.data_ptr<scalar_t>(),
                 index.data_ptr<int64_t>(),
                 b,
                 c,
                 n1,
                 n2,
                 k,
                 totalElements); }));

  AT_CUDA_CHECK(cudaGetLastError());
  return grad_input;
}