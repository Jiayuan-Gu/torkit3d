#pragma once

#include <ATen/ATen.h>

#ifdef WITH_CUDA
at::Tensor sample_farthest_points_cuda(
    const at::Tensor points,
    const int64_t num_samples);
#endif