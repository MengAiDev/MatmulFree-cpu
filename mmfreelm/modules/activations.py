# -*- coding: utf-8 -*-

# Copyright (c) 2023, Tri Dao.

import torch
import torch.nn.functional as F


@torch.jit.script
def swish(x):
    return F.silu(x)

# 1/sqrt(2*pi)-> 0.3989423
# 1/sqrt(2)   -> 0.70710678
# sqrt(2/pi)  -> 0.79788456


# this function is tanh approximation of gelu
# actual gelu is:
# x * 0.5 * (1.0 + torch.erf(x * 0.70710678))
@torch.jit.script
def bias_gelu(y, bias):
    x = bias + y
    return (x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))).to(dtype=y.dtype))


# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
@torch.jit.script
def bias_gelu_bwd(g, y, bias):
    """Assume that y has shape (B, D) and bias has shape (D)"""
    x = bias + y
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (
        1 + tanh_out
    )
    grad_y = ff * g
    return grad_y.to(dtype=y.dtype), grad_y.sum(dim=(0), dtype=bias.dtype)


class GeLUFunction(torch.autograd.Function):

    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        return bias_gelu(input, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        tmp = bias_gelu_bwd(grad_output, input, bias)
        return tmp, tmp


bias_gelu_impl = GeLUFunction.apply


# this function is tanh approximation of gelu
# actual gelu is:
# x * 0.5 * (1.0 + torch.erf(x * 0.70710678))
@torch.jit.script
def gelu_fwd(x):
    return (x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))).to(dtype=x.dtype))


# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
@torch.jit.script
def gelu_bwd(g, x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (
        1 + tanh_out
    )
    return (ff * g).to(dtype=x.dtype)


class FastGeLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return gelu_fwd(input)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        tmp = gelu_bwd(grad_output, input)
        return tmp


fast_gelu_impl = FastGeLUFunction.apply


@torch.jit.script
def relu_bwd(g, x):
    return torch.where(x >= 0, g, 0.0).to(dtype=x.dtype)


@torch.jit.script
def sqrelu_fwd(x):
    r = F.relu(x)
    return (r * r).to(dtype=x.dtype)


@torch.jit.script
def sqrelu_bwd(g, x):
    return (2.0 * g * F.relu(x)).to(dtype=x.dtype)


# 移除 Jiterator 实现，提供纯 PyTorch 的 SwiGLU 实现
class SwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        # SwiGLU: x * SiLU(y)
        sigmoid_y = torch.sigmoid(y)
        z = x * sigmoid_y * y  # SiLU(y) = y * sigmoid(y)
        ctx.save_for_backward(x, y, sigmoid_y)
        return z

    @staticmethod
    def backward(ctx, grad_output):
        x, y, sigmoid_y = ctx.saved_tensors
        
        # SiLU 的导数: d(SiLU(y))/dy = sigmoid(y) + y * sigmoid(y) * (1 - sigmoid(y))
        silu_grad = sigmoid_y + y * sigmoid_y * (1 - sigmoid_y)
        
        # 计算梯度
        grad_x = grad_output * sigmoid_y * y
        grad_y = grad_output * x * silu_grad
        
        return grad_x, grad_y


class SwiGLULinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, weight, bias):
        # 使用新的 SwiGLU 实现
        sigmoid_y = torch.sigmoid(y)
        z = x * sigmoid_y * y
        
        out = F.linear(z.to(weight.dtype), weight, bias)
        
        # 保存反向传播所需的值
        ctx.save_for_backward(x, y, sigmoid_y, weight)
        ctx.linear_bias_is_none = bias is None
        
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        x, y, sigmoid_y, weight = ctx.saved_tensors
        
        # 重塑梯度
        dout = dout.reshape(-1, dout.shape[-1])
        
        # 计算 dz
        dz = F.linear(dout, weight.t()).view_as(x)
        
        # SiLU 的导数
        silu_grad = sigmoid_y + y * sigmoid_y * (1 - sigmoid_y)
        
        # 计算梯度
        dx = dz * sigmoid_y * y
        dy = dz * x * silu_grad
        
        # 计算线性层的梯度
        z = x * sigmoid_y * y
        dlinear_weight = torch.einsum("bo,bi->oi", dout, z.reshape(-1, z.shape[-1]))
        dlinear_bias = None if ctx.linear_bias_is_none else dout.sum(0)
        
        return dx, dy, dlinear_weight, dlinear_bias


# 使用新的纯 PyTorch 实现
swiglu = SwiGLUFunction.apply
swiglu_linear = SwiGLULinearFunction.apply

ACT2FN = {
    'silu': swish,
    'swish': swish,
    'gelu': fast_gelu_impl,
    'bias_gelu': bias_gelu_impl,
}