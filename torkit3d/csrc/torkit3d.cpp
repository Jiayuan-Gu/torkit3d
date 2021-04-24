#include <torch/extension.h>
#include <torkit3d.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
#ifdef WITH_CUDA
  m.def("group_points_forward_cuda", &group_points_forward_cuda);
  m.def("group_points_backward_cuda", &group_points_backward_cuda);
  m.def("ball_query_cuda", &ball_query_cuda);
  m.def("farthest_point_sample_cuda", &farthest_point_sample_cuda);
  m.def("interpolate_forward_cuda", &interpolate_forward_cuda);
  m.def("interpolate_backward_cuda", &interpolate_backward_cuda);
  m.def("knn_distance_cuda", &knn_distance_cuda);
#endif
}