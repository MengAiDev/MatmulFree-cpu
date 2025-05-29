# -*- coding: utf-8 -*-

# Copyright (c) 2023, Tri Dao.
# https://github.com/state-spaces/mamba/blob/fb7b5310fa865dbd62aa059b1e26f2b431363e2a/mamba_ssm/ops/triton/layernorm.py
# Implement residual + layer_norm / rms_norm.

# For the backward pass, we keep weight_grad and bias_grad in registers and accumulate.
# This is faster for dimensions up to 8k, but after that it's much slower due to register spilling.
# The models we train have hidden dim up to 8k anyway (e.g. Llama 70B), so this is fine.

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmfreelm.utils import contiguous

def activation_quant(x):
    """
    Per-token quantization to 8 bits. No grouping is needed for quantization.

    Args:
        x: An activation tensor with shape [n, d].

    Returns:
        A quantized activation tensor with shape [n, d].
    """
    # Compute the scale factor
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    # Quantize and then de-quantize the tensor
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y

def weight_quant(w):
    """
    Per-tensor quantization to 1.58 bits. No grouping is needed for quantization.

    Args:
        w: A weight tensor with shape [d, k].

    Returns:
        A quantized weight tensor with shape [d, k].
    """
    # Compute the scale factor
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    # Quantize and then de-quantize the tensor
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u

def layer_norm_ref(x, weight, bias, residual=None, eps=1e-6, prenorm=False, upcast=False):
    dtype = x.dtype
    if upcast:
        weight = weight.float()
        bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        residual = residual.float() if residual is not None else residual
    if residual is not None:
        x = (x + residual).to(x.dtype)
    out = F.layer_norm(x.to(weight.dtype), x.shape[-1:], weight=weight, bias=bias, eps=eps).to(
        dtype
    )
    return out if not prenorm else (out, x)

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

def _layer_norm_fwd(
    x, weight, bias, eps, residual=None, out_dtype=None, residual_dtype=None, is_rms_norm=False
):
    if residual is not None:
        residual_dtype = residual.dtype
    M, N = x.shape
    # Handle residual connection
    if residual is not None:
        x = x + residual
    residual_out = x

    # Compute normalization
    if is_rms_norm:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        y = x * torch.rsqrt(variance + eps)
        if weight is not None:
            y = y * weight
        if bias is not None:
            y = y + bias
    else:
        y = F.layer_norm(x, (N,), weight, bias, eps)
    
    if out_dtype is not None:
        y = y.to(out_dtype)
    if residual_dtype is not None and residual_out.dtype != residual_dtype:
        residual_out = residual_out.to(residual_dtype)
    
    # Dummy mean and rstd for compatibility
    mean = torch.zeros(M, dtype=torch.float32, device=x.device) if not is_rms_norm else None
    rstd = torch.ones(M, dtype=torch.float32, device=x.device)
    return y, mean, rstd, residual_out

def _layer_norm_bwd(
    dy,
    x,
    weight,
    bias,
    eps,
    mean,
    rstd,
    dresidual=None,
    has_residual=False,
    is_rms_norm=False,
    x_dtype=None,
    recompute_output=False,
):
    M, N = x.shape
    # Recompute mean and rstd if needed for RMSNorm
    if is_rms_norm:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        rstd = 1 / torch.sqrt(variance + eps)
        xhat = x * rstd
    else:
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        rstd = 1 / torch.sqrt(variance + eps)
        xhat = (x - mean) * rstd

    # Compute gradients for weight and bias
    dw = (dy * xhat).sum(dim=0) if weight is not None else None
    db = dy.sum(dim=0) if bias is not None else None

    # Compute gradient for input
    if weight is not None:
        wdy = dy * weight
    else:
        wdy = dy
        
    if is_rms_norm:
        c1 = (xhat * wdy).sum(dim=-1, keepdim=True) / N
        dx = (wdy - xhat * c1) * rstd
    else:
        c1 = (xhat * wdy).sum(dim=-1, keepdim=True) / N
        c2 = wdy.sum(dim=-1, keepdim=True) / N
        dx = (wdy - (xhat * c1 + c2)) * rstd

    # Handle residual gradient
    if has_residual and dresidual is not None:
        dx = dx + dresidual
    dresidual_in = dx if has_residual and dx.dtype != x.dtype else None

    # Recompute output if needed
    y = xhat * weight + bias if recompute_output and weight is not None and bias is not None else None
    return (dx, dw, db, dresidual_in) if not recompute_output else (dx, dw, db, dresidual_in, y)

class LayerNormFn(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(
        ctx,
        x,
        weight,
        bias,
        residual=None,
        eps=1e-6,
        prenorm=False,
        residual_in_fp32=False,
        is_rms_norm=False,
    ):
        x_shape_og = x.shape
        x = x.reshape(-1, x.shape[-1])
        if residual is not None:
            residual = residual.reshape(-1, residual.shape[-1])
        residual_dtype = (
            residual.dtype
            if residual is not None
            else (torch.float32 if residual_in_fp32 else None)
        )
        y, mean, rstd, residual_out = _layer_norm_fwd(
            x, weight, bias, eps, residual, residual_dtype=residual_dtype, is_rms_norm=is_rms_norm
        )
        ctx.save_for_backward(residual_out, weight, bias, mean, rstd)
        ctx.x_shape_og = x_shape_og
        ctx.eps = eps
        ctx.is_rms_norm = is_rms_norm
        ctx.has_residual = residual is not None
        ctx.prenorm = prenorm
        ctx.x_dtype = x.dtype
        y = y.reshape(x_shape_og)
        return y if not prenorm else (y, residual_out.reshape(x_shape_og))

    @staticmethod
    @contiguous
    def backward(ctx, dy, *args):
        x, weight, bias, mean, rstd = ctx.saved_tensors
        dy = dy.reshape(-1, dy.shape[-1])
        assert dy.shape == x.shape
        if ctx.prenorm:
            dresidual = args[0]
            dresidual = dresidual.reshape(-1, dresidual.shape[-1])
            assert dresidual.shape == x.shape
        else:
            dresidual = None
        dx, dw, db, dresidual_in = _layer_norm_bwd(
            dy,
            x,
            weight,
            bias,
            ctx.eps,
            mean,
            rstd,
            dresidual,
            ctx.has_residual,
            ctx.is_rms_norm,
            x_dtype=ctx.x_dtype,
        )
        return (
            dx.reshape(ctx.x_shape_og),
            dw,
            db,
            dresidual_in.reshape(ctx.x_shape_og) if ctx.has_residual else None,
            None,
            None,
            None,
            None,
        )

def layer_norm_fn(
    x,
    weight,
    bias,
    residual=None,
    eps=1e-6,
    prenorm=False,
    residual_in_fp32=False,
    is_rms_norm=False,
):
    return LayerNormFn.apply(x, weight, bias, residual, eps, prenorm, residual_in_fp32, is_rms_norm)

def rms_norm_fn(
    x,
    weight,
    bias,
    residual=None,
    prenorm=False,
    residual_in_fp32=False,
    eps=1e-6
):
    return LayerNormFn.apply(x, weight, bias, residual, eps, prenorm, residual_in_fp32, True)

class LayerNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5
    ) -> LayerNorm:
        super().__init__()

        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        s += f", eps={self.eps}"
        s += ")"
        return s

    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        return layer_norm_fn(
            x,
            self.weight,
            self.bias,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32
        )

class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5
    ) -> RMSNorm:
        super().__init__()

        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        s += f", eps={self.eps}"
        s += ")"
        return s

    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        return rms_norm_fn(
            x,
            self.weight,
            self.bias,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
        )

class LayerNormLinearFn(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(
        ctx,
        x,
        norm_weight,
        norm_bias,
        linear_weight,
        linear_bias,
        residual=None,
        eps=1e-6,
        prenorm=False,
        residual_in_fp32=False,
        is_rms_norm=False,
    ):
        x_shape_og = x.shape
        x = x.reshape(-1, x.shape[-1])
        if residual is not None:
            residual = residual.reshape(-1, residual.shape[-1])
        residual_dtype = (
            residual.dtype
            if residual is not None
            else (torch.float32 if residual_in_fp32 else None)
        )
        y, mean, rstd, residual_out = _layer_norm_fwd(
            x,
            norm_weight,
            norm_bias,
            eps,
            residual,
            out_dtype=None if not torch.is_autocast_enabled() else torch.get_autocast_gpu_dtype(),
            residual_dtype=residual_dtype,
            is_rms_norm=is_rms_norm,
        )
        y = y.reshape(x_shape_og)
        dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else y.dtype
        linear_weight = linear_weight.to(dtype)
        linear_bias = linear_bias.to(
            dtype) if linear_bias is not None else None

        linear_weight = weight_quant(linear_weight)
        y = activation_quant(y)
        out = F.linear(y.to(linear_weight.dtype), linear_weight, linear_bias)
        
        # Save for backward
        ctx.save_for_backward(residual_out, norm_weight, norm_bias, linear_weight, y)
        ctx.x_shape_og = x_shape_og
        ctx.eps = eps
        ctx.is_rms_norm = is_rms_norm
        ctx.has_residual = residual is not None
        ctx.prenorm = prenorm
        ctx.x_dtype = x.dtype
        ctx.linear_bias_is_none = linear_bias is None
        return out if not prenorm else (out, residual_out.reshape(x_shape_og))

    @staticmethod
    @contiguous
    def backward(ctx, dout, *args):
        residual_out, norm_weight, norm_bias, linear_weight, y = ctx.saved_tensors
        dout = dout.reshape(-1, dout.shape[-1])
        y = y.reshape(dout.shape[0], -1)  # Ensure y matches dout dimensions
        
        # Linear backward
        dlinear_bias = None if ctx.linear_bias_is_none else dout.sum(0)
        dlinear_weight = torch.einsum("bo,bi->oi", dout, y)
        dy = F.linear(dout, linear_weight.t(), None)
        
        # LayerNorm backward
        if ctx.prenorm:
            dresidual = args[0]
            dresidual = dresidual.reshape(-1, dresidual.shape[-1])
            assert dresidual.shape == residual_out.shape
        else:
            dresidual = None
        
        dx, dnorm_weight, dnorm_bias, dresidual_in = _layer_norm_bwd(
            dy,
            residual_out,
            norm_weight,
            norm_bias,
            ctx.eps,
            None,  # mean not used in backward
            None,  # rstd not used in backward
            dresidual,
            ctx.has_residual,
            ctx.is_rms_norm,
            x_dtype=ctx.x_dtype,
        )
        
        return (
            dx.reshape(ctx.x_shape_og),
            dnorm_weight,
            dnorm_bias,
            dlinear_weight,
            dlinear_bias,
            dresidual_in.reshape(ctx.x_shape_og) if ctx.has_residual else None,
            None,
            None,
            None,
            None,
        )

def layer_norm_linear_fn(
    x,
    norm_weight,
    norm_bias,
    linear_weight,
    linear_bias,
    residual=None,
    eps=1e-6,
    prenorm=False,
    residual_in_fp32=False,
    is_rms_norm=False,
):
    return LayerNormLinearFn.apply(
        x,
        norm_weight,
        norm_bias,
        linear_weight,
        linear_bias,
        residual,
        eps,
        prenorm,
        residual_in_fp32,
        is_rms_norm,
    )

class LayerNormLinear(nn.Module):

    def __init__(
        self,
        hidden_size,
        elementwise_affine: bool = True,
        eps=1e-5
    ) -> LayerNormLinear:
        super().__init__()

        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        s += f", eps={self.eps}"
        s += ")"
        return s

    def forward(self, x, weight, bias, residual=None, prenorm=False, residual_in_fp32=False):
        return layer_norm_linear_fn(
            x,
            self.weight,
            self.bias,
            weight,
            bias,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            is_rms_norm=False
        )

class RMSNormLinear(nn.Module):

    def __init__(
        self,
        hidden_size,
        elementwise_affine: bool = True,
        eps=1e-5
    ) -> RMSNormLinear:
        super().__init__()

        self.hidden_size = hidden_size
        self.elementwise_affine = elementwise_affine
        self.eps = eps

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)
        self.register_parameter("bias", None)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}({self.hidden_size}"
        if not self.elementwise_affine:
            s += f", elementwise_affine={self.elementwise_affine}"
        s += f", eps={self.eps}"
        s += ")"
        return s

    def forward(self, x, weight, bias, residual=None, prenorm=False, residual_in_fp32=False):
        return layer_norm_linear_fn(
            x,
            self.weight,
            self.bias,
            weight,
            bias,
            residual=residual,
            eps=self.eps,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            is_rms_norm=True
        )