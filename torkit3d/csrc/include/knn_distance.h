#pragma once

#include <vector>
#include <ATen/ATen.h>

#ifdef WITH_CUDA
// Optimized for fixed and small k
std::vector<at::Tensor> knn_distance_cuda(
    const at::Tensor query_xyz,
    const at::Tensor key_xyz,
    const int64_t k);
#endif