import torch
import os.path as osp
# from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args
from custom_op import LinearOp3, LinearOp4, LinearOp, CustomLinear1, customOp_cpp, customOp_cuda


# def my_linear_py(g, input, weight, bias, in_channel, out_channel):
#     return g.op("custom::linearPy", input, weight, bias, in_channel, out_channel)

class LinearPy(LinearOp):
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.nn.parameter.Parameter, bias: torch.nn.parameter.Parameter, in_channel, out_channel) -> torch.Tensor:
        assert input.shape[-1] == weight.shape[0]
        ctx.save_for_backward(input, weight, bias)
        output = torch.matmul(input, weight) + bias
        return output

    @staticmethod
    def symbolic(g, input, weight, bias, in_channel, out_channel):
        return g.op("custom::linearPy", input, weight, bias, in_channel_i=in_channel, out_channel_i=out_channel)

class CustomLinearPy(CustomLinear1):
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super().__init__(in_channel, out_channel)
    
    def forward(self, x: torch.Tensor)->torch.Tensor:
        return LinearPy.apply(x, self.weight, self.bias, self.in_channel, self.out_channel)


class LinearCpp(LinearOp3):
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.nn.parameter.Parameter, bias: torch.nn.parameter.Parameter, in_channel, out_channel) -> torch.Tensor:
        assert input.shape[-1] == weight.shape[0]
        ctx.save_for_backward(input, weight, bias)
        output = customOp_cpp.forward(input, weight, bias)
        return output

    @staticmethod
    def symbolic(g, input, weight, bias, in_channel, out_channel):
        return g.op("custom::linearCpp", input, weight, bias, in_channel_i=in_channel, out_channel_i=out_channel)

class CustomLinearCpp(CustomLinear1):
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super().__init__(in_channel, out_channel)
    
    def forward(self, x: torch.Tensor)->torch.Tensor:
        return LinearCpp.apply(x, self.weight, self.bias, self.in_channel, self.out_channel)


class LinearCuda(LinearOp):
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.nn.parameter.Parameter, bias: torch.nn.parameter.Parameter, in_channel, out_channel) -> torch.Tensor:
        assert input.shape[-1] == weight.shape[0]
        ctx.save_for_backward(input, weight, bias)
        output = customOp_cuda.forward(input, weight, bias)
        return output

    @staticmethod
    def symbolic(g, input, weight, bias, in_channel, out_channel):
        return g.op("custom::linearCuda", input, weight, bias, in_channel_i=in_channel, out_channel_i=out_channel)

class CustomLinearCuda(CustomLinear1):
    def __init__(self, in_channel: int, out_channel: int) -> None:
        super().__init__(in_channel, out_channel)
    
    def forward(self, x: torch.Tensor)->torch.Tensor:
        return LinearCuda.apply(x, self.weight, self.bias, self.in_channel, self.out_channel)

def main():
    m = CustomLinear1(10, 12)
    m_py = CustomLinearPy(10, 12)
    m_cpp = CustomLinearCpp(10, 12)
    m_cuda = CustomLinearCuda(10, 12).cuda()
    x = torch.randn(10, 32, 32, 10)

    torch.onnx.export(m, x, "linear.onnx", opset_version=13)
    torch.onnx.export(m_py, x, "linear_py.onnx", opset_version=13)
    torch.onnx.export(m_cpp, x, "linear_cpp.onnx", opset_version=13)
    torch.onnx.export(
        m_cuda, 
        x.cuda(),
        "linear_cuda.onnx", 
        input_names=["x"], 
        output_names=["y"], 
        # verbose=True, 
        keep_initializers_as_inputs=True, 
        # export_params=False,
        opset_version=13, 
        dynamic_axes={
            "x": {
                0: "batch_size"
            },
            "y": {
                0: "batch_size"
            }}
            )

    print("export onnx success!")


if __name__ == "__main__":
    main()