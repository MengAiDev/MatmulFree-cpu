# -*- coding: utf-8 -*-

# Copyright (c) 2023, Tri Dao.
# Implement residual + RMSNorm with Swish gate on CPU.

# edited by MengAiDev, 2025

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

def rms_norm_ref(x, weight, bias, residual=None, eps=1e-6, prenorm=False, upcast=False):
    dtype = x.dtype
    if upcast:
        weight = weight.float()
        bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        residual = residual.float() if residual is not None else residual
    if residual is not None:
        x = (x + residual).to(x.dtype)
    rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
    out = (x * rstd * weight) + \
        bias if bias is not None else (x * rstd * weight)
    out = out.to(dtype)
    return out if not prenorm else (out, x)

class RMSNormSwishGateFn(Function):
    @staticmethod
    def forward(ctx, x, o, weight, bias, residual, eps, prenorm, residual_in_fp32):
        # Residual addition
        has_residual = residual is not None
        x_orig = x
        if has_residual:
            x = x + residual
        
        # RMSNorm
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        inv_var = torch.rsqrt(variance + eps)
        x_hat = x * inv_var
        
        # Normalized output (without gate)
        y = weight * x_hat
        
        # Swish gate
        sigmoid_o = torch.sigmoid(o)
        gate = o * sigmoid_o
        output = y * gate
        
        # Save for backward
        ctx.save_for_backward(
            x_hat, weight, inv_var, o, sigmoid_o, gate, y,
            None if not has_residual else residual
        )
        ctx.eps = eps
        ctx.has_residual = has_residual
        ctx.prenorm = prenorm
        ctx.x_shape_og = x_orig.shape
        ctx.o_shape_og = o.shape
        
        if prenorm:
            return output, x
        return output

    @staticmethod
    def backward(ctx, doutput, *args):
        # Retrieve saved tensors
        x_hat, weight, inv_var, o, sigmoid_o, gate, y, residual = ctx.saved_tensors
        has_residual = ctx.has_residual
        prenorm = ctx.prenorm
        
        # Gradient from next layer if prenorm
        dresidual = args[0] if prenorm else None
        
        # Gradient through Swish gate: dL/dgate
        dgate = doutput * y
        doutput_y = doutput * gate
        
        # Gradient through gate: dL/do
        dsigmoid_o = dgate * o
        do = dgate * sigmoid_o + dsigmoid_o * sigmoid_o * (1 - sigmoid_o)
        
        # Gradient through RMSNorm: dL/dy
        dy = doutput_y
        
        # Gradient for weight and transformed input
        dweight = (dy * x_hat).sum(dim=0)
        dx_hat = dy * weight
        
        # Gradient for RMSNorm
        D = x_hat.size(-1)
        dinv_var = (dx_hat * x_hat * inv_var.reciprocal()).sum(dim=-1, keepdim=True)
        dvariance = dinv_var * (-0.5) * (inv_var ** 3)
        dx = dx_hat * inv_var + 2.0 * x_hat * dvariance / D
        
        # Add residual gradient if exists
        if has_residual:
            if dresidual is not None:
                dx += dresidual
            dresidual_in = dx
        else:
            dresidual_in = None
        
        # Reshape gradients to match input shapes
        dx = dx.reshape(ctx.x_shape_og)
        do = do.reshape(ctx.o_shape_og)
        
        return dx, do, dweight, None, dresidual_in, None, None, None

def rms_norm_fn(x, o, weight, bias, residual=None, prenorm=False, residual_in_fp32=False, eps=1e-6):
    return RMSNormSwishGateFn.apply(x, o, weight, bias, residual, eps, prenorm, residual_in_fp32)

class FusedRMSNormSwishGate(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(hidden_size))
        self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.weight)

    def forward(self, x, o, residual=None, prenorm=False, residual_in_fp32=False):
        return rms_norm_fn(
            x,
            o,
            self.weight,
            self.bias,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
        )