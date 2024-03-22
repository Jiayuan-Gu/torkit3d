// CUDA Implementation for feature interpolation

#include <vector>
#include <ATen/ATen.h>
#include <ATen/cuda/ApplyGridUtils.cuh>
#include "utils.h"

template <typename scalar_t, typename index_t>
__global__ void interpolate_forward_kernel(
    scalar_t *output,                    // [B, C, N2]
    const scalar_t *__restrict__ input,  // [B, C, N1]
    const index_t *__restrict__ index,   // [B, N2, K]
    const scalar_t *__restrict__ weight, // [B, N2, K]
    const int b,
    const int c,
    const int n1,
    const int n2,
    const int k,
    const int totalElements)
{
  for (int linearId = blockIdx.x * blockDim.x + threadIdx.x;
       linearId < totalElements;
       linearId += gridDim.x * blockDim.x)
  {
    // Compute offsets
    const int n2_idx = linearId % n2;
    const int bc_idx = linearId / n2;
    const int c_idx = bc_idx % c;
    const int b_idx = bc_idx / c;

    scalar_t outputValue = 0.0;
    int srcOffset = c_idx * n1 + b_idx * (n1 * c);
    int indexOffset = n2_idx * k + b_idx * (k * n2);
    int weightOffset = n2_idx * k + b_idx * (k * n2);

    for (int i = 0; i < k; ++i)
    {
      index_t indexValue = index[indexOffset + i];
      scalar_t weightValue = weight[weightOffset + i];
      assert(indexValue >= 0 && indexValue < n1);
      outputValue += input[srcOffset + indexValue] * weightValue;
    }

    int dstOffset = n2_idx + c_idx * n2 + b_idx * (n2 * c);
    output[dstOffset] = outputValue;
  }
}

at::Tensor interpolate_forward_cuda(
    const at::Tensor input, // [B, C, N1]
    const at::Tensor index, // [B, N2, K]
    const at::Tensor weight // [B, N2, K]
)
{
  // Sanity check
  CHECK_CONTIGUOUS_CUDA(input);
  CHECK_CONTIGUOUS_CUDA(index);
  CHECK_CONTIGUOUS_CUDA(weight);
  TORCH_CHECK_EQ(input.dim(), 3);
  TORCH_CHECK_EQ(index.dim(), 3);
  TORCH_CHECK_EQ(weight.dim(), 3);
  TORCH_CHECK_EQ(input.size(0), index.size(0));
  TORCH_CHECK_EQ(input.size(0), weight.size(0));
  TORCH_CHECK_EQ(index.size(1), weight.size(1));
  TORCH_CHECK_EQ(index.size(2), weight.size(2));
  TORCH_CHECK_EQ(index.size(2), 3);

  const auto b = input.size(0);
  const auto c = input.size(1);
  const auto n1 = input.size(2);
  const auto n2 = index.size(1);
  const auto k = index.size(2);

  auto output = at::zeros({b, c, n2}, input.options());

  // Calculate grids and blocks for kernels
  const auto totalElements = output.numel();
  const int BLOCK_SIZE = at::cuda::getApplyBlockSize();
  dim3 grid;
  // const int curDevice = at::cuda::current_device();
  const int curDevice = input.get_device();
  TORCH_CHECK(at::cuda::getApplyGrid(totalElements, grid, curDevice, BLOCK_SIZE), "unable to get grid");

  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(),
      "interpolate_forward_cuda",
      ([&]
       { interpolate_forward_kernel<scalar_t, int64_t>
             <<<grid, BLOCK_SIZE>>>(
                 output.data_ptr<scalar_t>(),
                 input.data_ptr<scalar_t>(),
                 index.data_ptr<int64_t>(),
                 weight.data_ptr<scalar_t>(),
                 b,
                 c,
                 n1,
                 n2,
                 k,
                 totalElements); }));

  AT_CUDA_CHECK(cudaGetLastError());
  return output;
}

template <typename scalar_t, typename index_t>
__global__ void interpolate_backward_kernel(
    scalar_t *grad_input,                     // [B, C, N1]
    const scalar_t *__restrict__ grad_output, // [B, C, N2]
    const index_t *__restrict__ index,        // [B, N2, K]
    const scalar_t *__restrict__ weight,      // [B, N2, K]
    const int b,
    const int c,
    const int n1,
    const int n2,
    const int k,
    const int totalElements)
{
  for (int linearId = blockIdx.x * blockDim.x + threadIdx.x;
       linearId < totalElements;
       linearId += gridDim.x * blockDim.x)
  {
    // Compute offsets
    int n2_idx = linearId % n2;
    int bc_idx = linearId / n2;
    int c_idx = bc_idx % c;
    int b_idx = bc_idx / c;

    // grad_output
    int srcOffset = n2_idx + c_idx * n2 + b_idx * (n2 * c);
    scalar_t gradValue = grad_output[srcOffset];

    int indexOffset = n2_idx * k + b_idx * (k * n2);
    int weightOffset = n2_idx * k + b_idx * (k * n2);
    // grad input
    int dstOffset = c_idx * n1 + b_idx * (n1 * c);

    for (int i = 0; i < k; ++i)
    {
      index_t indexValue = index[indexOffset + i];
      scalar_t weightValue = weight[weightOffset + i];
      assert(indexValue >= 0 && indexValue < n1);
      atomicAdd(&grad_input[dstOffset + indexValue], gradValue * weightValue);
    }
  }
}

at::Tensor interpolate_backward_cuda(
    const at::Tensor grad_output, // [B, C, N2]
    const at::Tensor index,       // [B, N2, K]
    const at::Tensor weight,      // [B, N2, K]
    const int64_t n1)
{
  // Sanity check
  CHECK_CONTIGUOUS_CUDA(grad_output);
  CHECK_CONTIGUOUS_CUDA(index);
  CHECK_CONTIGUOUS_CUDA(weight);
  TORCH_CHECK_EQ(grad_output.dim(), 3);
  TORCH_CHECK_EQ(index.dim(), 3);
  TORCH_CHECK_EQ(weight.dim(), 3);
  TORCH_CHECK_EQ(grad_output.size(0), index.size(0));
  TORCH_CHECK_EQ(grad_output.size(0), weight.size(0));
  TORCH_CHECK_EQ(grad_output.size(2), index.size(1));
  TORCH_CHECK_EQ(index.size(1), weight.size(1));
  TORCH_CHECK_EQ(index.size(2), weight.size(2));
  TORCH_CHECK(index.size(2) == 3, "Only support k=3.");

  const auto b = grad_output.size(0);
  const auto c = grad_output.size(1);
  const auto n2 = grad_output.size(2);
  const auto k = index.size(2);

  auto grad_input = at::zeros({b, c, n1}, grad_output.options());

  // Calculate grids and blocks for kernels
  const auto totalElements = grad_output.numel();
  const int BLOCK_SIZE = at::cuda::getApplyBlockSize();
  dim3 grid;
  // const int curDevice = at::cuda::current_device();
  const int curDevice = grad_output.get_device();
  TORCH_CHECK(at::cuda::getApplyGrid(totalElements, grid, curDevice, BLOCK_SIZE), "unable to get grid");

  AT_DISPATCH_FLOATING_TYPES(
      grad_output.scalar_type(),
      "interpolate_backward_cuda",
      ([&]
       { interpolate_backward_kernel<scalar_t, int64_t>
             <<<grid, BLOCK_SIZE>>>(
                 grad_input.data_ptr<scalar_t>(),
                 grad_output.data_ptr<scalar_t>(),
                 index.data_ptr<int64_t>(),
                 weight.data_ptr<scalar_t>(),
                 b,
                 c,
                 n1,
                 n2,
                 k,
                 totalElements); }));

  AT_CUDA_CHECK(cudaGetLastError());
  return grad_input;
}