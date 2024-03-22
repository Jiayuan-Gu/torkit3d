#include <torch/extension.h>

#include <group_points.h>
#include <ball_query.h>
#include <farthest_point_sample.h>
#include <interpolate.h>
#include <knn_distance.h>
#include <chamfer_distance.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
#ifdef WITH_CUDA
  m.def("farthest_point_sample_cuda", &farthest_point_sample_cuda);
  m.def("ball_query_cuda", &ball_query_cuda);
  m.def("group_points_forward_cuda", &group_points_forward_cuda);
  m.def("group_points_backward_cuda", &group_points_backward_cuda);
  m.def("interpolate_forward_cuda", &interpolate_forward_cuda);
  m.def("interpolate_backward_cuda", &interpolate_backward_cuda);
  m.def("knn_distance_cuda", &knn_distance_cuda);
  m.def("chamfer_distance_forward_cuda", &chamfer_distance_forward_cuda);
  m.def("chamfer_distance_backward_cuda", &chamfer_distance_backward_cuda);
#endif
}