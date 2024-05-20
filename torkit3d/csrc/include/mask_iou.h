#pragma once

#include <ATen/ATen.h>

#ifdef WITH_CUDA
at::Tensor mask_iou_cuda(
    const at::Tensor mask1,
    const at::Tensor mask2);

#endif