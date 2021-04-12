#pragma once

#include <ATen/ATen.h>

#ifdef WITH_CUDA
at::Tensor group_points_forward_cuda(
    const at::Tensor input,
    const at::Tensor index);

at::Tensor group_points_backward_cuda(
    const at::Tensor grad_output,
    const at::Tensor index,
    const int64_t num_points);
#endif
