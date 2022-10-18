/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_runtime.h>
 #include "LinearPlugin.h"
 #include <assert.h>
#include <stdio.h>

#define BDIMX 32
#define BDIMY 32
#define IPAD 2

#define CHECK(call)\
 {\
   const cudaError_t error=call;\
   if(error!=cudaSuccess)\
   {\
       printf("ERROR: %s:%d,",__FILE__,__LINE__);\
       printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
       exit(1);\
   }\
}
 
// 矩阵转置，便于矩阵乘法的访存优化
__global__ void matrixTranspose(const float *input, float *output, const int height, const int width){
    __shared__ float buffer[BDIMY][BDIMX + IPAD];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 行读取，行写入
    if (x < width || y < height)
        buffer[threadIdx.y][threadIdx.x] = input[y * width + x];
    __syncthreads();

    if (x < width || y < height)
        // 列读取，行写入，加pad消除冲突
        output[y * height + x] = buffer[threadIdx.x][threadIdx.y];
}

 // 用于计算的 kernel
 __global__ void LinearKernel(const float *input, float *output,
                                const float* __restrict__ weight,
                                const float* __restrict__ bias,
                                const int inChannel,
                                const int outChanenel,
                                const int nElement)
 {   
     int rowNb, colNb;
     const int index = blockIdx.x * blockDim.x + threadIdx.x;
 
     rowNb = index / outChanenel;
     colNb = index % outChanenel;
     if (index >= nElement )
         return;
 
     const float *idata = input + rowNb * inChannel;
     const float *iweight = weight + colNb * inChannel;
 
     float res = 0.;
     for (int i = 0; i < inChannel; i++){
         res += idata[i] * iweight[i];
     } 
     output[index] = res + bias[colNb];
 }
 
 template<typename T>
 void writeToBuffer(char *&buffer, const T &val)
 {
     *reinterpret_cast<T *>(buffer) = val;
     buffer += sizeof(T);
 }
 
 namespace nvinfer1
 {
 // 这里各成员函数按照被调用顺序或重要程度顺序排列
 // class LinearPlugin
 LinearPlugin::LinearPlugin(const std::string &name, const int inChannel, const int outChannel, const float* weight, const float* bias):
     name_(name), inChannel(inChannel), outChannel(outChannel)
 {
     WHERE_AM_I();
     weight_ = (float *)malloc(inChannel * outChannel * sizeof(float));
     bias_ = (float *)malloc(outChannel * sizeof(float));
     memcpy(reinterpret_cast<void *>(const_cast<float *>(weight_)), weight, inChannel * outChannel * sizeof(float));
     memcpy(reinterpret_cast<void *>(const_cast<float *>(bias_)), bias, outChannel * sizeof(float));
 
     CHECK(cudaMalloc(&weight_d, inChannel * outChannel * sizeof(float)));
     CHECK(cudaMalloc(&bias_d, outChannel * sizeof(float)));
     CHECK(cudaMemcpy(reinterpret_cast<void *>(const_cast<float *>(weight_d)), reinterpret_cast<const void *>(weight_), inChannel * outChannel * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice));
     CHECK(cudaMemcpy(reinterpret_cast<void *>(const_cast<float *>(bias_d)), reinterpret_cast<const void *>(bias_), outChannel * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice));
 }
 
 LinearPlugin::LinearPlugin(const std::string &name, const void *buffer, size_t length):
     name_(name)
 {
     WHERE_AM_I();
     std::cout << "Start initializing Linear plugin" << std::endl;
     const char *d = static_cast<const char *>(buffer), *a = d;
     inChannel = *(reinterpret_cast<const int*>(d));
     d += sizeof(int);
     outChannel = *(reinterpret_cast<const int*>(d));
     d += sizeof(int);
     weight_ = (float *)malloc(inChannel * outChannel * sizeof(float));
     memcpy(reinterpret_cast<void *>(const_cast<float *>(weight_)), const_cast<float*>(reinterpret_cast<const float*>(d)), inChannel * outChannel * sizeof(float));
     d += inChannel * outChannel * sizeof(float);
     bias_ = (float*)malloc(outChannel * sizeof(float));
     memcpy(reinterpret_cast<void *>(const_cast<float *>(bias_)), const_cast<float*>(reinterpret_cast<const float*>(d)), outChannel * sizeof(float));
     d += outChannel * sizeof(float);
     
     CHECK(cudaMalloc(&weight_d, inChannel * outChannel * sizeof(float)));
     CHECK(cudaMalloc(&bias_d, outChannel * sizeof(float)));
     CHECK(cudaMemcpy(reinterpret_cast<void *>(const_cast<float *>(weight_d)), reinterpret_cast<const void *>(weight_), inChannel * outChannel * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice));
     CHECK(cudaMemcpy(reinterpret_cast<void *>(const_cast<float *>(bias_d)), reinterpret_cast<const void *>(bias_), outChannel * sizeof(float), cudaMemcpyKind::cudaMemcpyHostToDevice));
 
     assert(a == d - length);
     std::cout << "Succeed initializing Linear plugin" << std::endl;
 }
 
 LinearPlugin::~LinearPlugin()
 {
     WHERE_AM_I();
 }
 
 IPluginV2DynamicExt *LinearPlugin::clone() const noexcept
 {
     WHERE_AM_I();
     std::cout << "start cloning!" << std::endl;
     auto p = new LinearPlugin(name_, inChannel, outChannel, weight_, bias_);
     p->setPluginNamespace(namespace_.c_str());
     std::cout << "Succeed cloning!" << std::endl;
 
     return p;
 }
 
 int32_t LinearPlugin::getNbOutputs() const noexcept
 {
     WHERE_AM_I();
     return 1;
 }
 
 DataType LinearPlugin::getOutputDataType(int32_t index, DataType const *inputTypes, int32_t nbInputs) const noexcept
 {
     WHERE_AM_I();
     return inputTypes[0];
 }
 
 DimsExprs LinearPlugin::getOutputDimensions(int32_t outputIndex, const DimsExprs *inputs, int32_t nbInputs, IExprBuilder &exprBuilder) noexcept
 {
     WHERE_AM_I();
     DimsExprs outputDim;
     outputDim.nbDims = inputs[0].nbDims;
     outputDim.d[0] = inputs[0].d[0];
     if (outputDim.nbDims == 2)
     {
         outputDim.d[1] = exprBuilder.constant(outChannel);
     }
     else if (outputDim.nbDims == 3)
     {
         outputDim.d[1] = inputs[0].d[1];
         outputDim.d[2] = exprBuilder.constant(outChannel);
     }
     else if (outputDim.nbDims == 4)
     {
         outputDim.d[1] = inputs[0].d[1];
         outputDim.d[2] = inputs[0].d[2];
         outputDim.d[3] = exprBuilder.constant(outChannel);
     }
     
     return outputDim;
 }
 
 bool LinearPlugin::supportsFormatCombination(int32_t pos, const PluginTensorDesc *inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
 {
     WHERE_AM_I();
 #ifdef DEBUG
     bool res;
     switch (pos)
     {
     case 0:
         res = inOut[0].type == DataType::kFLOAT && inOut[0].format == TensorFormat::kLINEAR;
         break;
     case 1:
         res = inOut[1].format == inOut[0].format && inOut[1].type == inOut[0].type;
         break;
     default: // should NOT be here!
         res = false;
     }
 
     std::cout << "\tpos=" << pos << ",res=" << res << "->[";
     for (int i = 0; i < nbInputs + nbOutputs; ++i)
     {
         std::cout << getFormatString(inOut[i].format) << ",";
     }
     std::cout << "],[";
     for (int i = 0; i < nbInputs + nbOutputs; ++i)
     {
         std::cout << dataTypeToString(inOut[i].type) << ",";
     }
     std::cout << "]" << std::endl;
     return res;
 #else
     std::cout << "Start checking format!" << std::endl;
     switch (pos)
     {
     bool res;
     case 0:
         res = inOut[0].type == DataType::kFLOAT && inOut[0].format == TensorFormat::kLINEAR;
         return res;
     case 1:
         res = inOut[1].type == inOut[0].type && inOut[1].format == inOut[0].format;
         return res;
     default: // should NOT be here!
         return false;
     }
 
     return false;
 #endif
 }
 
 void LinearPlugin::configurePlugin(const DynamicPluginTensorDesc *in, int32_t nbInputs, const DynamicPluginTensorDesc *out, int32_t nbOutputs) noexcept
 {
     WHERE_AM_I();
     assert(nbInputs == 1 && nbOutputs == 1);
     // std::cout << int(in->desc.dims.nbDims) -1 << in->desc.dims.d[0] << in->desc.dims.d[1] << in->desc.dims.d[2] << in->desc.dims.d[3] << std::endl;
     // std::cout << int(out->desc.dims.nbDims) -1 << out->desc.dims.d[0] << out->desc.dims.d[1] <<out->desc.dims.d[2] <<out->desc.dims.d[3] <<std::endl;
     // assert(in->desc.dims.d[int(in->desc.dims.nbDims) -1] == inChannel);
     // assert(out->desc.dims.d[int(out->desc.dims.nbDims) - 1] == outChannel);
     return;
 }
 
 size_t LinearPlugin::getWorkspaceSize(const PluginTensorDesc *inputs, int32_t nbInputs, const PluginTensorDesc *outputs, int32_t nbOutputs) const noexcept
 {
     WHERE_AM_I();
     return 0;
 }
 
 int32_t LinearPlugin::enqueue(const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) noexcept
 {
     WHERE_AM_I();
     std::cout << "Start enqueuing!" << std::endl;   
    int nElement = 1;
    for (int i=0; i < outputDesc[0].dims.nbDims; ++i)
     {   
         nElement *= outputDesc[0].dims.d[i];
     }
     
    const int height = inChannel, width=outChannel;
    float * transpoedWeight_d = nullptr;
    cudaMalloc((void**)&transpoedWeight_d, sizeof(float) * nElement);
    {   dim3 block(BDIMX, BDIMY, 1);
        dim3 grid(CEIL_DIVIDE(height, block.x), CEIL_DIVIDE(width, block.y), 1);
        matrixTranspose<<<grid, block, 0, stream>>>(weight_d, transpoedWeight_d, height, width);
        CHECK(cudaDeviceSynchronize());
    }

    dim3 grid(CEIL_DIVIDE(nElement, 256), 1, 1), block(256, 1, 1);
    LinearKernel<<<grid, block, 0, stream>>>((float *)inputs[0], (float *)outputs[0], transpoedWeight_d, bias_d, inChannel, outChannel, nElement);
    std::cout << "Succeed enqueuing!" << std::endl;    
    return 0;
 }
 
 void LinearPlugin::destroy() noexcept
 {
     WHERE_AM_I();
     free(reinterpret_cast<void *>(const_cast<float *>(weight_)));
     free(reinterpret_cast<void *>(const_cast<float *>(bias_)));
     cudaFree(reinterpret_cast<void *>(const_cast<float *>(weight_d)));
     cudaFree(reinterpret_cast<void *>(const_cast<float *>(bias_d)));
     delete this;
     return;
 }
 
 int32_t LinearPlugin::initialize() noexcept
 {
     WHERE_AM_I();
     return 0;
 }
 
 void LinearPlugin::terminate() noexcept
 {
     WHERE_AM_I();
     return;
 }
 
 size_t LinearPlugin::getSerializationSize() const noexcept
 {
     WHERE_AM_I();
     size_t total_size = 0;
     total_size += sizeof(inChannel);
     total_size += sizeof(outChannel);
     total_size += inChannel * outChannel * sizeof(float);
     total_size += outChannel * sizeof(float);
     return total_size;
 }
 
 void LinearPlugin::serialize(void *buffer) const noexcept
 {
     WHERE_AM_I();
     std::cout << "Start serializing!" << std::endl;    
     char * d = reinterpret_cast<char*>(buffer);
     writeToBuffer(d, inChannel);
     writeToBuffer(d, outChannel);
     memcpy(d, weight_, inChannel * outChannel * sizeof(float));
     d += inChannel * outChannel * sizeof(float);
     memcpy(d, bias_, outChannel * sizeof(float));
     d += outChannel * sizeof(float);
     std::cout << "Succeed serializing!" << std::endl;
     return;
 }
 
 void LinearPlugin::setPluginNamespace(const char *pluginNamespace) noexcept
 {
     WHERE_AM_I();
     namespace_ = pluginNamespace;
     return;
 }
 
 const char *LinearPlugin::getPluginNamespace() const noexcept
 {
     WHERE_AM_I();
     return namespace_.c_str();
 }
 
 const char *LinearPlugin::getPluginType() const noexcept
 {
     WHERE_AM_I();
     return PLUGIN_NAME;
 }
 
 const char *LinearPlugin::getPluginVersion() const noexcept
 {
     WHERE_AM_I();
     return PLUGIN_VERSION;
 }
 
 void LinearPlugin::attachToContext(cudnnContext *contextCudnn, cublasContext *contextCublas, IGpuAllocator *gpuAllocator) noexcept
 {
     WHERE_AM_I();
     return;
 }
 
 void LinearPlugin::detachFromContext() noexcept
 {
     WHERE_AM_I();
     return;
 }
 
 // class LinearPluginCreator
 PluginFieldCollection    LinearPluginCreator::fc_ {};
 std::vector<PluginField> LinearPluginCreator::attr_;
 
 LinearPluginCreator::LinearPluginCreator()
 {
     WHERE_AM_I();
     attr_.clear();
    attr_.emplace_back(PluginField("in_channel", nullptr, PluginFieldType::kINT32, 1));
    attr_.emplace_back(PluginField("out_channel", nullptr, PluginFieldType::kINT32, 1));
    attr_.emplace_back(PluginField("weight", nullptr, PluginFieldType::kFLOAT32, 1));
    attr_.emplace_back(PluginField("bias", nullptr, PluginFieldType::kFLOAT32, 1));
     fc_.nbFields = attr_.size();
     fc_.fields = attr_.data();
 }
 
 LinearPluginCreator::~LinearPluginCreator()
 {
     WHERE_AM_I();
 }
 
 // 最重要的两个成员函数，分别用于“接受参数创建 Plugin” 和 “去序列化创建 Plugin”
 IPluginV2 *LinearPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept
 {
     WHERE_AM_I();
     int                   inChannel  = 0;
     int                   outChannel = 0;
    float                 *weight = nullptr;
    float                 *bias = nullptr;
     std::cout << "Start creating plugin!" << std::endl;
     for (int i = 0; i < fc->nbFields; ++i)
     {   
        std::string filed_name(fc->fields[i].name);
        if (filed_name == "in_channel")
         {
             inChannel = reinterpret_cast<const int *>(fc->fields[i].data)[0];
        }else if (filed_name == "out_channel")
         {
             outChannel = reinterpret_cast<const int *>(fc->fields[i].data)[0];
        }else if (filed_name == "weight")
         {
            weight = const_cast<float *>(reinterpret_cast<const float *>(fc->fields[i].data));
         }else if (filed_name == "bias")
         {
            bias = const_cast<float *>(reinterpret_cast<const float *>(fc->fields[i].data));
         }
     }
    LinearPlugin *pObj = new LinearPlugin(name, inChannel, outChannel, weight, bias);
     pObj->setPluginNamespace(namespace_.c_str());
     std::cout << "Succeed creating plugin!" << std::endl;
     return pObj;
 }
 
 IPluginV2 *LinearPluginCreator::deserializePlugin(const char *name, const void *serialData, size_t serialLength) noexcept
 {
     WHERE_AM_I();
     std::cout << "Start deserialing plugin!" << std::endl;
     LinearPlugin *pObj = new LinearPlugin(name, serialData, serialLength);
     pObj->setPluginNamespace(namespace_.c_str());
     std::cout << "succeed deserialing plugin!" << std::endl;
     return pObj;
 }
 
 void LinearPluginCreator::setPluginNamespace(const char *pluginNamespace) noexcept
 {
     WHERE_AM_I();
     namespace_ = pluginNamespace;
     return;
 }
 
 const char *LinearPluginCreator::getPluginNamespace() const noexcept
 {
     WHERE_AM_I();
     return namespace_.c_str();
 }
 
 const char *LinearPluginCreator::getPluginName() const noexcept
 {
     WHERE_AM_I();
     return PLUGIN_NAME;
 }
 
 const char *LinearPluginCreator::getPluginVersion() const noexcept
 {
     WHERE_AM_I();
     return PLUGIN_VERSION;
 }
 
 const PluginFieldCollection *LinearPluginCreator::getFieldNames() noexcept
 {
     WHERE_AM_I();
     return &fc_;
 }
 
 REGISTER_TENSORRT_PLUGIN(LinearPluginCreator);
 
 } // namespace nvinfer1
 
struct Rectangle
{
    double length {};
    double width {};
};
