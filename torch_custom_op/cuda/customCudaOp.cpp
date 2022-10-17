#include <torch/extension.h>
#include <vector>


torch::Tensor linearCudaForward(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias);

std::vector<torch::Tensor> linearCudaBackward(
    torch::Tensor gradOutput, 
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor linearForward(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias)
    {
      CHECK_INPUT(input);
      CHECK_INPUT(weight);
      CHECK_INPUT(bias);
      return linearCudaForward(input, weight, bias);
    }

std::vector<torch::Tensor> linearBackward(
    torch::Tensor gradOutput, 
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias)
    {
      CHECK_INPUT(gradOutput);
      CHECK_INPUT(input);
      CHECK_INPUT(weight);
      CHECK_INPUT(bias);
      return linearCudaBackward(gradOutput, input, weight, bias);
    }


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &linearForward, "Linear(cuda) forward");
  m.def("backward", &linearBackward, "Linear(cuda) backward");
}

