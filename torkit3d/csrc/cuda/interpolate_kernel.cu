// CUDA Implementation for feature interpolation

#include <vector>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

#define MAX_THREADS 512

/********************************
* Forward
*********************************/
template <typename scalar_t, typename index_t>
__global__ void interpolate_forward_kernel(
    scalar_t *output,
    const scalar_t *__restrict__ input,
    const index_t *__restrict__ index,
    const scalar_t *__restrict__ weight,
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
    index_t n2_offset = linearId % n2;
    index_t linearId1 = linearId / n2;
    index_t c_offset = linearId1 % c;
    index_t b_offset = linearId1 / c;

    scalar_t outputValue = 0.0;
    index_t srcOffset = c_offset * n1 + b_offset * (n1 * c);
    index_t indexOffset = n2_offset * k + b_offset * (k * n2);
    index_t weightOffset = n2_offset * k + b_offset * (k * n2);

    for (int i = 0; i < k; ++i)
    {
      index_t indexValue = index[indexOffset + i];
      scalar_t weightValue = weight[weightOffset + i];
      assert(indexValue >= 0 && indexValue < n1);
      outputValue += input[srcOffset + indexValue] * weightValue;
    }

    index_t dstOffset = n2_offset + c_offset * n2 + b_offset * (n2 * c);
    output[dstOffset] = outputValue;
  }
}

/**
 * Forward
 * Input:
 *  input: [B, C, N1]
 *  index: [B, N2, K]
 *  weight: [B, N2, K]
 * Output:
 *  output: [B, C, N2]
 **/
at::Tensor interpolate_forward_cuda(
    const at::Tensor input,
    const at::Tensor index,
    const at::Tensor weight)
{

  // Sanity check
  CHECK_INPUT(input);
  CHECK_INPUT(index);
  CHECK_INPUT(weight);
  CHECK_EQ(input.dim(), 3);
  CHECK_EQ(index.dim(), 3);
  CHECK_EQ(weight.dim(), 3);
  CHECK_EQ(input.size(0), index.size(0));
  CHECK_EQ(input.size(0), weight.size(0));
  CHECK_EQ(index.size(1), weight.size(1));
  CHECK_EQ(index.size(2), weight.size(2));
  CHECK_EQ(index.size(2), 3);

  const auto b = input.size(0);
  const auto c = input.size(1);
  const auto n1 = input.size(2);
  const auto n2 = index.size(1);
  const auto k = index.size(2);

  auto output = at::zeros({b, c, n2}, input.options());

  // Calculate grids and blocks for kernels
  const auto totalElements = output.numel();
  const dim3 block = at::cuda::getApplyBlock();
  dim3 grid;
  const int curDevice = at::cuda::current_device();
  // getApplyGrid: aten/src/ATen/cuda/CUDAApplyUtils.cuh
  THArgCheck(at::cuda::getApplyGrid(totalElements, grid, curDevice), 1, "Too many elements to calculate");

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "interpolate_forward_cuda", ([&] {
                               interpolate_forward_kernel<scalar_t, int64_t>
                                   <<<grid, block>>>(
                                       output.data_ptr<scalar_t>(),
                                       input.data_ptr<scalar_t>(),
                                       index.data_ptr<int64_t>(),
                                       weight.data_ptr<scalar_t>(),
                                       b,
                                       c,
                                       n1,
                                       n2,
                                       k,
                                       totalElements);
                             }));

  THCudaCheck(cudaGetLastError());

  return output;
}

/**********************************
* Backward
***********************************/
template <typename scalar_t, typename index_t>
__global__ void interpolate_backward_kernel(
    scalar_t *grad_input,
    const scalar_t *__restrict__ grad_output,
    const index_t *__restrict__ index,
    const scalar_t *__restrict__ weight,
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
    index_t n2_offset = linearId % n2;
    index_t linearId1 = linearId / n2;
    index_t c_offset = linearId1 % c;
    index_t b_offset = linearId1 / c;

    // grad_output
    index_t srcOffset = n2_offset + c_offset * n2 + b_offset * (n2 * c);
    scalar_t gradValue = grad_output[srcOffset];

    index_t indexOffset = n2_offset * k + b_offset * (k * n2);
    index_t weightOffset = n2_offset * k + b_offset * (k * n2);
    // grad input
    index_t dstOffset = c_offset * n1 + b_offset * (n1 * c);

    for (int i = 0; i < k; ++i)
    {
      index_t indexValue = index[indexOffset + i];
      scalar_t weightValue = weight[weightOffset + i];
      assert(indexValue >= 0 && indexValue < n1);
      atomicAdd(&grad_input[dstOffset + indexValue], gradValue * weightValue);
    }
  }
}

/**
 * Backward
 * Input:
 *  grad_output: [B, C, N2]
 *  index: [B, N2, K]
 *  weight: [B, N2, K]
 * Output:
 *  grad_input: [B, C, N1]
 **/
at::Tensor interpolate_backward_cuda(
    const at::Tensor grad_output,
    const at::Tensor index,
    const at::Tensor weight,
    const int64_t n1)
{

  // Sanity check
  CHECK_INPUT(grad_output);
  CHECK_INPUT(index);
  CHECK_INPUT(weight);
  CHECK_EQ(grad_output.dim(), 3);
  CHECK_EQ(index.dim(), 3);
  CHECK_EQ(weight.dim(), 3);
  CHECK_EQ(grad_output.size(0), index.size(0));
  CHECK_EQ(grad_output.size(0), weight.size(0));
  CHECK_EQ(grad_output.size(2), index.size(1));
  CHECK_EQ(index.size(1), weight.size(1));
  CHECK_EQ(index.size(2), weight.size(2));
  TORCH_CHECK(index.size(2) == 3, "Only support k=3.");

  const auto b = grad_output.size(0);
  const auto c = grad_output.size(1);
  const auto n2 = grad_output.size(2);
  const auto k = index.size(2);

  auto grad_input = at::zeros({b, c, n1}, grad_output.options());

  // Calculate grids and blocks for kernels
  const auto totalElements = grad_output.numel();
  const dim3 block = at::cuda::getApplyBlock();
  dim3 grid;
  const int curDevice = at::cuda::current_device();
  // getApplyGrid: aten/src/ATen/cuda/CUDAApplyUtils.cuh
  THArgCheck(at::cuda::getApplyGrid(totalElements, grid, curDevice), 1, "Too many elements to calculate");

  AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "interpolate_backward_cuda", ([&] {
                               interpolate_backward_kernel<scalar_t, int64_t>
                                   <<<grid, block>>>(
                                       grad_input.data_ptr<scalar_t>(),
                                       grad_output.data_ptr<scalar_t>(),
                                       index.data_ptr<int64_t>(),
                                       weight.data_ptr<scalar_t>(),
                                       b,
                                       c,
                                       n1,
                                       n2,
                                       k,
                                       totalElements);
                             }));

  THCudaCheck(cudaGetLastError());

  return grad_input;
}