#pragma once

#include <vector>
#include <ATen/ATen.h>

#ifdef WITH_CUDA
at::Tensor interpolate_forward_cuda(
    const at::Tensor input,
    const at::Tensor index,
    const at::Tensor weight);

at::Tensor interpolate_backward_cuda(
    const at::Tensor grad_output,
    const at::Tensor index,
    const at::Tensor weight,
    const int64_t n1);
#endif
