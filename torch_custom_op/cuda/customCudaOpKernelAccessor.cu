#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <torch/extension.h>

template <typename scalar_t>
__global__ void
linearForwardKernel4D(const torch::PackedTensorAccessor32<scalar_t, 4> input,
                      const torch::PackedTensorAccessor32<scalar_t, 2> weight,
                      const scalar_t *bias,
                      torch::PackedTensorAccessor32<scalar_t, 4> out) {
  int dim0 = input.size(0), dim1 = input.size(1), dim2 = input.size(2),
      dim3 = weight.size(1), dimExt = weight.size(0);
  int w = (blockIdx.x * blockDim.x + threadIdx.x) % dim3;
  int x = (blockIdx.x * blockDim.x + threadIdx.x) / dim3;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= dim2 || y >= dim1 || z >= dim0) {
    return;
  }

  scalar_t res = 0;
  for (int i = 0; i < dimExt; i++) {
    res += input[z][y][x][i] * weight[i][w];
    // printf("%d, %d, %d, %d, %d\t %f, %f, %f\n",z, y, x, w, i,
    // input[z][y][x][i], weight[i][w], bias[w]);
  }

  out[z][y][x][w] = res + bias[w];
}

template <typename scalar_t>
__global__ void linearWeightBackwardKernel4D(
    const torch::PackedTensorAccessor32<scalar_t, 4> input,
    const torch::PackedTensorAccessor32<scalar_t, 4> gradOutput,
    torch::PackedTensorAccessor32<scalar_t, 2> gradWeight) {
  int dim0 = gradWeight.size(0), dim1 = gradWeight.size(1),
      dimExt0 = input.size(0), dimExt1 = input.size(1), dimExt2 = input.size(2);
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dim1 || y >= dim0) {
    return;
  }
  scalar_t res = 0;
  for (int i = 0; i < dimExt0; i++)
    for (int j = 0; j < dimExt1; j++)
      for (int k = 0; k < dimExt2; k++) {
        res += input[i][j][k][y] * gradOutput[i][j][k][x];
      }
  gradWeight[y][x] = res;
}

template <typename scalar_t>
__global__ void linearInputBackwardKernel4D(
    const torch::PackedTensorAccessor32<scalar_t, 4> gradOutput,
    const torch::PackedTensorAccessor32<scalar_t, 2> weight,
    torch::PackedTensorAccessor32<scalar_t, 4> gradinput) {
  int dim0 = gradinput.size(0), dim1 = gradinput.size(1),
      dim2 = gradinput.size(2), dim3 = gradinput.size(3),
      dimExt = weight.size(1);
  int w = (blockIdx.x * blockDim.x + threadIdx.x) % dim3;
  int x = (blockIdx.x * blockDim.x + threadIdx.x) / dim3;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= dim2 || y >= dim1 || z >= dim0) {
    return;
  }
  scalar_t res = 0;
  for (int i = 0; i < dimExt; i++) {
    res += gradOutput[z][y][x][i] * weight[w][i];
  }
  gradinput[z][y][x][w] = res;
}

template <typename scalar_t>
__global__ void linearBiasBackwardKernel4D(
    const torch::PackedTensorAccessor32<scalar_t, 4> gradOutput,
    scalar_t *gradBias) {
  int dim0 = gradOutput.size(3), dimExt0 = gradOutput.size(0),
      dimExt1 = gradOutput.size(1), dimExt2 = gradOutput.size(2);
  int x = blockIdx.x * blockDim.x + threadIdx.x;

  if (x >= dim0) {
    return;
  }

  scalar_t res = 0;
  for (int i = 0; i < dimExt0; i++)
    for (int j = 0; j < dimExt1; j++)
      for (int k = 0; k < dimExt2; k++)
        res += gradOutput[i][j][k][x];

  gradBias[x] = res;
}

torch::Tensor linearCudaForward(torch::Tensor input, torch::Tensor weight,
                                torch::Tensor bias) {
  int dims = input.dim();
  torch::Tensor output;
  output = torch::zeros(
      {input.size(0), input.size(1), input.size(2), weight.size(1)},
      input.options());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weight.type(), "Linear_Cuda_Forward", [&] {
        dim3 block(32, 32, 1);
        dim3 grid((output.size(3) * output.size(2) - 1) / block.x + 1,
                  (output.size(1) - 1) / block.y + 1,
                  (output.size(0) - 1) / block.z + 1);
        linearForwardKernel4D<scalar_t>
            <<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                input.packed_accessor32<scalar_t, 4>(),
                weight.packed_accessor32<scalar_t, 2>(),
                bias.data_ptr<scalar_t>(),
                output.packed_accessor32<scalar_t, 4>());
      });
  return output;
}

std::vector<torch::Tensor> linearCudaBackward(torch::Tensor gradOutput,
                                              torch::Tensor input,
                                              torch::Tensor weight,
                                              torch::Tensor bias) {
  int dims = input.dim();
  torch::Tensor gradInput = torch::zeros_like(input);
  torch::Tensor gradWeight = torch::zeros_like(weight);
  torch::Tensor gradBias = torch::zeros_like(bias);

  AT_DISPATCH_FLOATING_TYPES(
      weight.scalar_type(), "LinearWeightCudaBackward", [&] {
        dim3 block0(32, 32, 1);
        dim3 grid0((weight.size(1) - 1) / block0.x + 1,
                   (weight.size(0) - 1) / block0.y + 1, 1);
        linearWeightBackwardKernel4D<scalar_t>
            <<<grid0, block0, 0, at::cuda::getCurrentCUDAStream()>>>(
                input.packed_accessor32<scalar_t, 4>(),
                gradOutput.packed_accessor32<scalar_t, 4>(),
                gradWeight.packed_accessor32<scalar_t, 2>());
      });

  AT_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "LinearInputCudaBackward", [&] {
        dim3 block1(32, 32, 1);
        dim3 grid1((input.size(3) * input.size(2) - 1) / block1.x + 1,
                   (input.size(1) - 1) / block1.y + 1,
                   (input.size(0) - 1) / block1.z + 1);
        linearInputBackwardKernel4D<scalar_t>
            <<<grid1, block1, 0, at::cuda::getCurrentCUDAStream()>>>(
                gradOutput.packed_accessor32<scalar_t, 4>(),
                weight.packed_accessor32<scalar_t, 2>(),
                gradInput.packed_accessor32<scalar_t, 4>());
      });
  AT_DISPATCH_FLOATING_TYPES(bias.scalar_type(), "LinearBiasCudaBackward", [&] {
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    dim3 block2(1024, 1, 1);
    dim3 grid2((gradBias.size(0) - 1) / block2.x + 1, 1, 1);
    linearBiasBackwardKernel4D<scalar_t>
        <<<grid2, block2, 0, at::cuda::getCurrentCUDAStream()>>>(
            gradOutput.packed_accessor32<scalar_t, 4>(),
            gradBias.data_ptr<scalar_t>());
  });
  return {gradInput, gradWeight, gradBias};
}
