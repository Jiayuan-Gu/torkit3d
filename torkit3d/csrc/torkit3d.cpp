#include <torch/extension.h>
#include <torkit3d.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
#ifdef WITH_CUDA
  m.def("group_points_forward_cuda", &group_points_forward_cuda);
  m.def("group_points_backward_cuda", &group_points_backward_cuda);
#endif
}