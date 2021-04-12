/* Memory efficient gather points */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>  // at::cuda::getApplyGrid

// NOTE: AT_ASSERT has become TORCH_CHECK on master after 0.4.
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


/**
  * Forward
  * Input:
  *   input: [B, C, N1]
  *   index: [B, N2, K]
  * Output:
  *   output: [B, C, N2, K]
  **/
at::Tensor group_points_forward_cuda(
    const at::Tensor input,
    const at::Tensor index) {

  // Sanity check
  CHECK_CUDA(input);
  CHECK_CUDA(index);
  CHECK_EQ(input.dim(), 3);
  CHECK_EQ(index.dim(), 3);
  CHECK_EQ(input.size(0), index.size(0));
    
  const auto b = input.size(0);
  const auto c = input.size(1);
  const auto n1 = input.size(2);
  const auto n2 = index.size(1);
  const auto k = index.size(2);

  auto input_expand = input.unsqueeze(2).expand({b, c, n2, n1});  // (B, C, N2, N1)
  auto index_expand = index.unsqueeze(1).expand({b, c, n2, k});  // (B, C, N2, K)

  auto output = input_expand.gather(3, index_expand);  // (B, C, N2, K)

  return output;
}

template <typename scalar_t, typename index_t>
__global__ void group_points_backward_kernel(
    scalar_t* grad_input,
    const scalar_t* __restrict__ grad_output,
    const index_t* __restrict__ index,
    const index_t b,
    const index_t c,
    const index_t n1,
    const index_t n2,
    const index_t k,
    const index_t totalElements) {
  for (index_t linearId = blockIdx.x * blockDim.x + threadIdx.x;
       linearId < totalElements;
       linearId += gridDim.x * blockDim.x) {
    // Compute offsets
    index_t k_offset = linearId % k;
    index_t linearId1 = linearId / k;
    index_t n2_offset = linearId1 % n2;
    index_t linearId2 = linearId1 / n2;
    index_t c_offset = linearId2 % c;
    index_t b_offset = linearId2 / c;

    index_t srcOffset = k_offset 
      + n2_offset * k 
      + c_offset * (k * n2)
      + b_offset * (k * n2 * c);
    index_t dstOffset = c_offset * n1 + b_offset * (n1 * c);
    index_t indexOffset = k_offset + n2_offset * k + b_offset * (k * n2);

    index_t indexValue = index[indexOffset];
    assert(indexValue >= 0 && indexValue < n1);
    atomicAdd(&grad_input[dstOffset + indexValue], grad_output[srcOffset]);
  }
}

/* 
Backward interface
Input:
  grad_output: (B, C, N2, K)
  index: (B, N2, K)
Output:
  grad_input: (B, C, N1)
*/
/**
  * Backward
  * Input:
  *   grad_output: [B, C, N2, K]
  *   index: [B, N2, K]
  * Output:
  *   grad_input: [B, C, N1]
  **/
at::Tensor group_points_backward_cuda(
    const at::Tensor grad_output,
    const at::Tensor index,
    const int64_t num_points) {

  // Sanity check
  CHECK_INPUT(grad_output);
  CHECK_INPUT(index);
  CHECK_EQ(grad_output.dim(), 4);
  CHECK_EQ(index.dim(), 3);
  CHECK_EQ(index.size(0), grad_output.size(0));
  CHECK_EQ(index.size(1), grad_output.size(2));
  CHECK_EQ(index.size(2), grad_output.size(3));

  const auto b = grad_output.size(0);
  const auto c = grad_output.size(1);
  const auto n2 = grad_output.size(2);
  const auto k = grad_output.size(3);

  // Allocate new space for output
  auto grad_input = at::zeros({b, c, num_points}, grad_output.options());
  CHECK_CUDA(grad_input);
  CHECK_CONTIGUOUS(grad_input);

  // Calculate grids and blocks for kernels 
  const auto totalElements = grad_output.numel();
  const dim3 block = at::cuda::getApplyBlock();
  dim3 grid;
  const int curDevice = at::cuda::current_device();
  // getApplyGrid: aten/src/ATen/cuda/CUDAApplyUtils.cuh
  THArgCheck(at::cuda::getApplyGrid(totalElements, grid, curDevice), 1, "Too many elements to calculate");

  AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "group_points_backward_cuda", ([&] {
    group_points_backward_kernel<scalar_t, int64_t>
      <<<grid, block>>>(
        grad_input.data_ptr<scalar_t>(),
        grad_output.data_ptr<scalar_t>(),
        index.data_ptr<int64_t>(),
        b,
        c,
        num_points,
        n2,
        k,
        totalElements);
  }));

  THCudaCheck(cudaGetLastError());

  return grad_input;
}