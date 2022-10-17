#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

template<typename scalar_t>
__global__ void linearForwardKernel(
    const  scalar_t* input, 
    const  scalar_t* weight, 
    const  scalar_t* bias, 
    scalar_t* out, 
    size_t height, 
    size_t width, 
    size_t inChannel){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height){
        return;
    }

    const scalar_t *idata = input + y * inChannel;
    const scalar_t *iweight = weight + x * inChannel;

    int res = 0;
    for (int i = 0; i < inChannel; i++){
        res += idata[i] * iweight[i];
    }

    out[y * height + x] = res + bias[x];
}

template<typename scalar_t>
__global__ void linearWeightBackwardKernel(
    const  scalar_t* input, 
    const  scalar_t* gradOutput, 
    scalar_t* gradWeight, 
    size_t height, 
    size_t width, 
    size_t size)
    {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height){
        return;
    }

    const scalar_t *idata = input + y * size;
    const scalar_t *igrad = gradOutput + x * size;

    int res = 0;
    for (int i = 0; i < size; i++){
        res += igrad[i] * idata[i];
    }

    gradWeight[y * height + x] = res;
}

template<typename scalar_t>
__global__ void linearBiasBackwardKernel(
    const  scalar_t* gradOutput, 
    scalar_t* gradBias, 
    scalar_t width, 
    size_t size)
    {
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width){
        return;
    }

    const scalar_t *igrad = gradOutput + x * size;

    int res = 0;
    for (int i = 0; i < size; i++){
        res += igrad[i];
    }

    gradBias[x] = res;
}

torch::Tensor linearCudaForward(
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias)
    {
        int width = 1, height = 1;
        int outputDims[input.dim()];
        for (int i = 0; i < input.dim() - 1; i++){
            outputDims[i] = input.size(i);
            height *= input.size(i);
        }
        outputDims[input.dim() - 1] = weight.size(-1);
        width = weight.size(-1);

        torch::Tensor output;
        if (input.dim() == 2)
            output = torch::zeros({outputDims[0], outputDims[1]}, torch::dtype(torch::kFloat32));
        else if(input.dim() == 3)
            output = torch::zeros({outputDims[0], outputDims[1], outputDims[2]}, torch::dtype(torch::kFloat32));
        else if(input.dim() == 4)
            output = torch::zeros({outputDims[0], outputDims[1], outputDims[2], outputDims[3]}, torch::dtype(torch::kFloat32));
        else
            throw "Input shape not implimented!";

        dim3 block(32, 32, 1);
        dim3 grid((width - 1)/block.x + 1, (height - 1)/block.y + 1, 1);
        AT_DISPATCH_FLOATING_TYPES(weight.scalar_type(), "Linear_Cuda_Forward", [&]{
            linearForwardKernel<scalar_t><<<grid, block>>>(
              input.view({-1, input.size(-1)}).data_ptr<scalar_t>(),
              weight.transpose(0, 1).data_ptr<scalar_t>(),
              bias.data_ptr<scalar_t>(),
              output.view({-1, output.size(-1)}).data_ptr<scalar_t>(),
              size_t(width),
              size_t(height),
              size_t(weight.size(0))
            );
        });
        return output;
    }


std::vector<torch::Tensor> linearCudaBackward(
    torch::Tensor gradOutput, 
    torch::Tensor input, 
    torch::Tensor weight, 
    torch::Tensor bias)
    {
        int inChannel = 1, outChannel = 1, size = 1;
        std::vector<int> outputDims;
        for (int i = 0; i < input.dim() - 1; i++){
            outputDims.push_back(input.size(i));
            size *= input.size(i);
        }
        outputDims.push_back(weight.size(-1));
        inChannel = weight.size(0);
        outChannel = weight.size(-1);
        torch::Tensor gradInput = torch::zeros_like(input);
        torch::Tensor gradWeight = torch::zeros_like(weight);
        torch::Tensor gradBias = torch::zeros_like(bias);

        dim3 block0(32, 32, 1);
        dim3 grid0((outChannel - 1)/block0.x + 1, (inChannel - 1)/block0.y + 1, 1);
        AT_DISPATCH_FLOATING_TYPES(weight.scalar_type(), "LinearWeightCudaBackward", [&]{
            linearWeightBackwardKernel<scalar_t><<<grid0, block0>>>(
                input.transpose(0, 1).data_ptr<scalar_t>(),
                gradOutput.transpose(0, 1).data_ptr<scalar_t>(),
                gradWeight.data_ptr<scalar_t>(),
                size_t(inChannel),
                size_t(outChannel),
                size_t(size)
            );
            });
        dim3 block1(1024, 1, 1);
        dim3 grid1((outChannel - 1)/block1.x + 1, 1, 1);
        AT_DISPATCH_FLOATING_TYPES(weight.scalar_type(), "LinearBiasCudaBackward", [&]{
          linearBiasBackwardKernel<scalar_t><<<grid1, block1>>>(
            gradOutput.data_ptr<scalar_t>(),
            gradBias.data_ptr<scalar_t>(),
            size_t(outChannel),
            size_t(size)
          );
        });

        dim3 block2(32, 32, 1);
        dim3 grid2((inChannel - 1)/block2.x + 1, (size - 1)/block2.y + 1, 1);
        AT_DISPATCH_FLOATING_TYPES(weight.scalar_type(), "LinearInputCudaBackward", ([&]{
            linearWeightBackwardKernel<scalar_t><<<grid2, block2>>>(
                gradOutput.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(),
                gradInput.data_ptr<scalar_t>(),
                size_t(size),
                size_t(inChannel),
                size_t(outChannel)
            );
            }));
        
        return {gradInput, gradWeight, gradBias};
    }
