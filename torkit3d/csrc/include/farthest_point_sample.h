#pragma once

#include <ATen/ATen.h>

#ifdef WITH_CUDA
at::Tensor farthest_point_sample_cuda(
    const at::Tensor points,
    const int64_t num_samples);
#endif