from cgi import test
from operator import mod
import random
import os
from telnetlib import TM
import torch
import torch.nn as nn
import numpy as np
import customOp_cpp
import customOp_cuda
from time import time, sleep
# from torch.utils.cpp_extension import load
# customOp_cuda = load(
#     'customOp_cuda', ["cuda/customCudaOp.cpp", "cuda/customCudaOpKernelAccessor.cu"], verbose=True)

def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed_torch()

def equal(m1: torch.Tensor, m2: torch.Tensor, thresh=1e-4):
    diff = torch.max(torch.abs(m1 - m2)) 
    if diff < thresh:
        return True
    else:
        mean = torch.mean(torch.abs(m1 - m2)) 
        return str(float(diff)) + "(max error)/" + str(float(mean)) + "(mean error)"

class CustomLinear1(nn.Module):
    def __init__(self, in_channel:int, out_channel:int) -> None:
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.weight = torch.nn.parameter.Parameter(torch.ones(in_channel, out_channel, dtype=torch.float32))
        self.bias = torch.nn.parameter.Parameter(torch.ones(out_channel, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self):
        seed_torch()
        stdv = 1.0
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, x) -> torch.Tensor:
        assert x.shape[-1] == self.in_channel, "Iput's last dim must match in_channel: {self.inchannel}"
        return torch.matmul(x, self.weight) + self.bias

class LinearOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.nn.parameter.Parameter, bias: torch.nn.parameter.Parameter,**kwargs) -> torch.Tensor:
        assert input.shape[-1] == weight.shape[0]
        ctx.save_for_backward(input, weight, bias)
        output = torch.matmul(input, weight) + bias
        return output
    
    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> torch.Tensor:
        input, weight, bias = ctx.saved_tensors
        weight_grad = torch.mm(input.view(-1, int(input.shape[-1])).T, output_grad.view(-1, int(output_grad.shape[-1])))
        bias_grad = torch.sum(output_grad.view(-1, output_grad.shape[-1]), dim=0)
        input_grad = torch.mm(weight, output_grad.view(-1, int(output_grad.shape[-1])).T)
        return input_grad, weight_grad, bias_grad

class CustomLinear2(CustomLinear1):
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super().__init__(in_channel, out_channel)
    
    def forward(self, x)->torch.Tensor:
        return LinearOp.apply(x, self.weight, self.bias)


class LinearOp3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.nn.parameter.Parameter, bias: torch.nn.parameter.Parameter,**kwargs) -> torch.Tensor:
        assert input.shape[-1] == weight.shape[0]
        ctx.save_for_backward(input, weight, bias)
        output = customOp_cpp.forward(input, weight, bias)
        return output
    
    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> torch.Tensor:
        input, weight, bias = ctx.saved_tensors
        input_grad, weight_grad, bias_grad = customOp_cpp.backward(output_grad, input, weight, bias)
        return input_grad, weight_grad, bias_grad

class CustomLinear3(CustomLinear1):
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super().__init__(in_channel, out_channel)
        
    def forward(self, x: torch.Tensor)->torch.Tensor:
        return LinearOp3.apply(x, self.weight, self.bias)


class LinearOp4(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.nn.parameter.Parameter, bias: torch.nn.parameter.Parameter,**kwargs) -> torch.Tensor:
        assert input.shape[-1] == weight.shape[0]
        ctx.save_for_backward(input, weight, bias)
        output = customOp_cuda.forward(input, weight, bias)
        return output
    
    @staticmethod
    def backward(ctx, output_grad: torch.Tensor) -> torch.Tensor:
        input, weight, bias = ctx.saved_tensors
        input_grad, weight_grad, bias_grad = customOp_cuda.backward(output_grad.contiguous(), input, weight, bias)
        return input_grad, weight_grad, bias_grad

class CustomLinear4(CustomLinear1):
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super().__init__(in_channel, out_channel)
        
    def forward(self, x: torch.Tensor)->torch.Tensor:
        return LinearOp4.apply(x, self.weight, self.bias)

def main():
    warmup_iteration = 100
    test_iteration = 10000

    co1 = CustomLinear1(10, 12).cuda()
    co2 = CustomLinear2(10, 12).cuda()
    co3 = CustomLinear3(10, 12).cuda()
    co4 = CustomLinear4(10, 12).cuda()


    x = torch.randn(11, 23, 24, 10).to(torch.device("cuda:0"))
    module_f, func_f, cpp_f, cuda_f = 0, 0, 0, 0
    module_b, func_b, cpp_b, cuda_b = 0, 0, 0, 0

    
    for i in range(warmup_iteration):
        o1 = co1(x)
        o2 = co2(x)
        o3 = co3(x)
        o4 = co4(x)

        loss1 = torch.sum(o1) * 2
        loss1.backward()
        loss2 = torch.sum(o2) * 2
        loss2.backward()
        loss3 = torch.sum(o3) * 2
        loss3.backward()
        loss4 = torch.sum(o4) * 2
        loss4.backward()

    for i in range(test_iteration):
        t = time()
        o1 = co1(x)
        t_m = time()
        o2 = co2(x)
        t_f = time()
        o3 = co3(x)
        t_cpp = time()
        o4 = co4(x)
        t_cuda = time()

        module_f += t_m - t
        func_f += t_f - t_m
        cpp_f += t_cpp - t_f
        cuda_f += t_cuda - t_cpp

        t = time()
        loss1 = torch.sum(o1) * 2
        loss1.backward()
        t_m = time()
        loss2 = torch.sum(o2) * 2
        loss2.backward()
        t_f = time()
        loss3 = torch.sum(o3) * 2
        loss3.backward()
        t_cpp = time()
        loss4 = torch.sum(o4) * 2
        loss4.backward()
        t_cuda = time()

        module_b += t_m - t
        func_b += t_f - t_m
        cpp_b += t_cpp - t_f
        cuda_b += t_cuda - t_cpp

    # 数据量大了之后由于大量浮点数乘加操产生不小的介对误差（对齐浮点数小数点导致），权重的梯度误差可能会比较大，用并行规约可以优化 (猜测，误差未解决)
    print(f"auto_grad/cpp/cuda output equal orginal? {equal(o1, o2)}, {equal(o1, o3)}, {equal(o1, o4)}")
    print(f"auto_grad/cpp/cuda weight's grad equal orginal? {equal(co1.weight.grad, co2.weight.grad)}, {equal(co1.weight.grad, co3.weight.grad)}, {equal(co1.weight.grad, co4.weight.grad)}")
    print(f"auto_grad/cpp/cuda bias's grad equal orginal? {equal(co1.bias.grad, co2.bias.grad)}, {equal(co1.bias.grad, co3.bias.grad)}, {equal(co1.bias.grad, co4.bias.grad)}")

    print("model:         \tMoudle\tfunction\tcpp\tcuda")
    print(f"Inference(s): \t{module_f:.3f}\t{func_f:.3f}\t{cpp_f:.3f}\t{cuda_f:.3f}")
    print(f"Backward(s):  \t{module_b:.3f}\t{func_b:.3f}\t{cpp_b:.3f}\t{cuda_b:.3f}")

def deb():
    co1 = CustomLinear1(10, 12).cuda()
    co2 = CustomLinear2(10, 12).cuda()
    co3 = CustomLinear3(10, 12).cuda()
    co4 = CustomLinear4(10, 12).cuda()


    x = torch.randn(32 , 12, 32, 10).to(torch.device("cuda:0"))

    for i in range(100):
        o1 = co1(x)
        o2 = co2(x)
        o3 = co3(x)
        o4 = co4(x)

        # print(o1)
        # print(o4)
        
        loss1 = torch.sum(o1) * 2
        loss1.backward()
        loss2 = torch.sum(o2) * 2
        loss2.backward()
        loss3 = torch.sum(o3) * 2
        loss3.backward()
        loss4 = torch.sum(o4) * 2
        loss4.backward()

    print(torch.equal(co1.weight.grad, co2.weight.grad), torch.equal(co1.weight.grad, co3.weight.grad),  torch.equal(co1.weight.grad, co4.weight.grad))

    print(torch.max(co1.weight.grad.data - co4.weight.grad.data))


if __name__ == '__main__':
    main()
    # deb()
