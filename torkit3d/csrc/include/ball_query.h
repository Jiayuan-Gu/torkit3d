#pragma once

#include <ATen/ATen.h>

#ifdef WITH_CUDA
at::Tensor ball_query_cuda(
    const at::Tensor query,
    const at::Tensor key,
    const float radius,
    const int64_t max_neighbors);
#endif