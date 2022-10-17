#include <torch/extension.h>
#include <vector>


torch::Tensor linearForward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias){
    torch::Tensor output;
    output = torch::matmul(input, weight) + bias;
    return output;
}

std::vector<torch::Tensor> linearBackward(torch::Tensor gradOutput, torch::Tensor input, torch::Tensor weight, torch::Tensor bias){
    torch::Tensor gradInput, gradWeight, gradBias;
    gradInput = torch::matmul(gradOutput, weight.transpose(0,1));
    gradWeight = torch::matmul(input.view({-1, input.size(-1)}).transpose(0,1), gradOutput.view({-1, gradOutput.size(-1)}));
    gradBias = torch::sum(gradOutput.view({-1, gradOutput.size(-1)}), 0);

    return {gradInput, gradWeight, gradBias};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &linearForward, "Linear forward");
  m.def("backward", &linearBackward, "Linear backward");
}
