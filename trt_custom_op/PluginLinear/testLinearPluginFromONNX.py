#
# Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from ast import parse
import ctypes
from statistics import mode
from turtle import window_height
from cuda import cudart
import numpy as np
import os
import tensorrt as trt

pluginName = "linearCuda"
soFile = "./LinearPlugin.so"
onnxFile = "./linear_cuda_modified.onnx"
np.random.seed(97)

def printArrayInfomation(x, info="", n=5):
    print( '%s:%s,SumAbs=%.5e,Var=%.5f,Max=%.5f,Min=%.5f,SAD=%.5f'%( \
        info,str(x.shape),np.sum(abs(x)),np.var(x),np.max(x),np.min(x),np.sum(np.abs(np.diff(x.reshape(-1)))) ))
    print('\t', x.reshape(-1)[:n], x.reshape(-1)[-n:])

def check(a, b, weak=False, checkEpsilon=1e-5):
    if weak:
        res = np.all(np.abs(a - b) < checkEpsilon)
    else:
        res = np.all(a == b)
    diff0 = np.max(np.abs(a - b))
    diff1 = np.max(np.abs(a - b) / (np.abs(b) + checkEpsilon))
    print("check:%s, absDiff=%f, relDiff=%f" % (res, diff0, diff1))

def addLinearCPU(inputH: np.array, weight:np.array, bias:np.array):
    output = np.matmul(inputH, weight) + bias
    return output

def getLinearPlugin(inChannel, outChannel, weight, bias):
    for c in trt.get_plugin_registry().plugin_creator_list:
        #print(c.name)
        if c.name == pluginName:
            parameterList = []
            parameterList.append(trt.PluginField("i_channel", np.int32(inChannel), trt.PluginFieldType.INT32))
            parameterList.append(trt.PluginField("out_channel", np.int32(outChannel), trt.PluginFieldType.INT32))
            parameterList.append(trt.PluginField("weight", np.ascontiguousarray(weight), trt.PluginFieldType.FLOAT32))
            parameterList.append(trt.PluginField("bias", bias, trt.PluginFieldType.FLOAT32))
            return c.create_plugin(c.name, trt.PluginFieldCollection(parameterList))
            print("succeed creating plugin!")
    return None

def run(shape, inchannel, outchannel, weight, bias):
    testCase = "<shape=%s,inchannel=%d, outchannel=%d>" % (shape, inchannel, outchannel)
    trtFile = "./model-onnx-Dim%s.plan" % str(len(shape))
    print("Test %s" % testCase)
    logger = trt.Logger(trt.Logger.ERROR)
    trt.init_libnvinfer_plugins(logger, '')
    ctypes.cdll.LoadLibrary(soFile)
    if os.path.isfile(trtFile):
        with open(trtFile, "rb") as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        if engine == None:
            print("Failed loading engine!")
            return
        print("Succeeded loading engine!")
    else:
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 6 << 30)

        # if bUseFP16Mode:
        #     config.set_flag(trt.BuilderFlag.FP16)
        # if bUseINT8Mode:
        #     config.set_flag(trt.BuilderFlag.INT8)
        #     config.int8_calibrator = calibrator.MyCalibrator(calibrationDataPath, nCalibration, (1, 1, nHeight, nWidth),
        #                                              cacheFile)

        parser = trt.OnnxParser(network, logger)
        if not os.path.exists(onnxFile):
            print("Fail finding onnx file!")
            exit()
        print("Succeeded finding ONNX file!")
        with open(onnxFile, "rb") as model:
            if not parser.parse(model.read()):
                print("Fail parsing .onnx file!")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                exit()
        print("Succeed parsing .onnx file")

        inputTensor = network.get_input(0)
        profile.set_shape(inputTensor.name, [1, 32, 32, 10], [8, 32, 32, 10], [32, 32, 32, 10])
        config.add_optimization_profile(profile)
        
        engineString = builder.build_serialized_network(network, config)
        if engineString == None:
            print("Failed building engine!")
            return
        print("Succeeded building engine!")
        with open(trtFile, "wb") as f:
            f.write(engineString)
        engine = trt.Runtime(logger).deserialize_cuda_engine(engineString)

    print("Starting building context!")
    context = engine.create_execution_context()
    print("Succeed building context!")
    context.set_binding_shape(0, shape)
    #print("Binding all? %s"%(["No","Yes"][int(context.all_binding_shapes_specified)]))
    nInput = np.sum([engine.binding_is_input(i) for i in range(engine.num_bindings)])

    nOutput = engine.num_bindings - nInput
    #for i in range(nInput):
    #    print("Bind[%2d]:i[%2d]->" % (i, i), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))
    #for i in range(nInput, nInput + nOutput):
    #    print("Bind[%2d]:o[%2d]->" % (i, i - nInput), engine.get_binding_dtype(i), engine.get_binding_shape(i), context.get_binding_shape(i), engine.get_binding_name(i))

    bufferH = []
    # bufferH.append(np.random.randn(*shape).astype(np.float32))
    bufferH.append(np.random.rand(*shape).astype(np.float32))

    for i in range(nOutput):
        bufferH.append(np.empty(context.get_binding_shape(nInput + i), dtype=trt.nptype(engine.get_binding_dtype(nInput + i))))
    bufferD = []
    for i in range(engine.num_bindings):
        bufferD.append(cudart.cudaMalloc(bufferH[i].nbytes)[1])

    for i in range(nInput):
        cudart.cudaMemcpy(bufferD[i], np.ascontiguousarray(bufferH[i].reshape(-1)).ctypes.data, bufferH[i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    print("Start executing!")
    context.execute_v2(bufferD)

    for i in range(nOutput):
        cudart.cudaMemcpy(bufferH[nInput + i].ctypes.data, bufferD[nInput + i], bufferH[nInput + i].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    outputCPU = addLinearCPU(bufferH[:nInput], weight, bias)
    """
    for i in range(nInput):
        printArrayInfomation(bufferH[i])
    for i in range(nOutput):
        printArrayInfomation(bufferH[nInput+i])
    for i in range(nOutput):
        printArrayInfomation(outputCPU[i])
    """
    cudart.cudaMemcpy(bufferH[0].ctypes.data, bufferD[0], bufferH[0].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)

    # print(np.sum(bufferH[nInput:][0]), np.sum(outputCPU[0]))
    check(bufferH[nInput:][0], outputCPU[0], True)

    for buffer in bufferD:
        cudart.cudaFree(buffer)
    print("Test %s finish!\n" % testCase)

if __name__ == "__main__":
    os.system("rm -rf ./*.plan")
    np.set_printoptions(precision=3, linewidth=100, suppress=True)
    # weight = np.random.randn(8, 10).astype(np.float32)
    # bias   = np.random.randn(10).astype(np.float32)
    # run([8, 8, 8, 8], 8, 10, weight, bias)
    # run([8, 32, 8], 8, 10, weight, bias)

    weight = np.load("./weight.npy")
    bias   = np.load("./bias.npy")
    run([8, 32, 32, 10], 10, 12, weight, bias)

    print("Test all finish!")
