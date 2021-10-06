#pragma once

#include <ATen/ATen.h>

#ifdef WITH_CUDA
std::vector<at::Tensor> chamfer_distance_forward_cuda(
    const at::Tensor xyz1,
    const at::Tensor xyz2);

std::vector<at::Tensor> chamfer_distance_backward_cuda(
    const at::Tensor grad_dist1,
    const at::Tensor grad_dist2,
    const at::Tensor xyz1,
    const at::Tensor xyz2,
    const at::Tensor idx1,
    const at::Tensor idx2);

#endif