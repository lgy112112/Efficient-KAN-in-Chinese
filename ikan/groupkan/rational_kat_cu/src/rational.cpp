#include <torch/extension.h>
#include "utils.h"

// torch::Tensor rational_fwd(
//   torch::Tensor x, 
//   torch::Tensor n, 
//   torch::Tensor d) {
//   CHECK_INPUT(x);
//   CHECK_INPUT(n);
//   CHECK_INPUT(d);
//   return rational_fwd_cuda(x, n, d);
// }

// torch::Tensor rational_fwd_optimized(
//   torch::Tensor x, 
//   torch::Tensor n, 
//   torch::Tensor d) {
//   CHECK_INPUT(x);
//   CHECK_INPUT(n);
//   CHECK_INPUT(d);
//   return rational_fwd_cuda_optimized(x, n, d);
// }

// std::vector<torch::Tensor> rational_bwd(
//   torch::Tensor grad_output, 
//   torch::Tensor x, 
//   torch::Tensor n, 
//   torch::Tensor d) {
//   CHECK_INPUT(grad_output);
//   CHECK_INPUT(x);
//   CHECK_INPUT(n);
//   CHECK_INPUT(d);
//   return rational_bwd_cuda(grad_output, x, n, d);
// }

// std::vector<torch::Tensor> rational_bwd_optimized(
//   torch::Tensor grad_output, 
//   torch::Tensor x, 
//   torch::Tensor n, 
//   torch::Tensor d) {
//   CHECK_INPUT(grad_output);
//   CHECK_INPUT(x);
//   CHECK_INPUT(n);
//   CHECK_INPUT(d);
//   return rational_bwd_cuda_optimized(grad_output, x, n, d);
// }

torch::Tensor rational_fwd_1dgroup(
  torch::Tensor x, 
  torch::Tensor n, 
  torch::Tensor d,
  int group) {
  CHECK_INPUT(x);
  CHECK_INPUT(n);
  CHECK_INPUT(d);

  // check group <= 32
  CHECK_LESS(group, 32);

  return rational_fwd_cuda_1dgroup(x, n, d, group);
}

std::vector<torch::Tensor> rational_bwd_1dgroup(
  torch::Tensor grad_output, 
  torch::Tensor x, 
  torch::Tensor n, 
  torch::Tensor d,
  int group) {
  CHECK_INPUT(grad_output);
  CHECK_INPUT(x);
  CHECK_INPUT(n);
  CHECK_INPUT(d);

  // check group <= 32
  CHECK_LESS(group, 32);

  return rational_bwd_cuda_1dgroup(grad_output, x, n, d, group);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // m.def("rational_fwd", &rational_fwd, 
  //   "rational forward (CUDA)");
  // m.def("rational_bwd", &rational_bwd,
  //   "rational backward (CUDA)");
  // m.def("rational_bwd_optimized", &rational_bwd_optimized,
  //   "rational backward optimized (CUDA)");
  // m.def("rational_fwd_optimized", &rational_fwd_optimized,
  //   "rational forward optimized (CUDA)");
  m.def("rational_fwd_1dgroup", &rational_fwd_1dgroup,
    "rational forward 1dgroup (CUDA)");
  m.def("rational_bwd_1dgroup", &rational_bwd_1dgroup,
    "rational backward 1dgroup (CUDA)");
}