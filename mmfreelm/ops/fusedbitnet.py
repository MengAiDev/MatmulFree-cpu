from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmfreelm.modules import RMSNorm
from mmfreelm.utils import contiguous


def activation_quant(x):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y


def weight_quant(w):
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u


def _layer_norm_fwd_quant(
    x, weight, bias, eps, residual=None, out_dtype=None, residual_dtype=None, is_rms_norm=False
):
    if residual is not None:
        x = x + residual
        residual_out = x if residual_dtype is None or residual_dtype == x.dtype else x.to(residual_dtype)
    else:
        residual_out = x

    if is_rms_norm:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        rstd = torch.rsqrt(variance + eps)
        x_hat = x * rstd
        mean = None
    else:
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        rstd = torch.rsqrt(variance + eps)
        x_hat = (x - mean) * rstd

    if weight is not None:
        x_hat = x_hat * weight
    if bias is not None:
        x_hat = x_hat + bias

    y = activation_quant(x_hat)
    if out_dtype is not None:
        y = y.to(out_dtype)

    mean = mean.squeeze(-1) if mean is not None else None
    rstd = rstd.squeeze(-1)
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
    dtype = x.dtype
    x = x.float()
    dy = dy.float()
    if weight is not None:
        weight = weight.float()
    if bias is not None:
        bias = bias.float()
    if mean is not None:
        mean = mean.float().view(M, 1)
    rstd = rstd.float().view(M, 1)

    if is_rms_norm:
        x_norm = x * rstd
    else:
        x_norm = (x - mean) * rstd

    d_bias = dy.sum(dim=0) if bias is not None else None
    d_weight = (dy * x_norm).sum(dim=0) if weight is not None else None

    if weight is not None:
        dy = dy * weight

    if is_rms_norm:
        dx = dy * rstd
        d_rstd = (dy * x).sum(dim=-1, keepdim=True)
        d_var = d_rstd * (-0.5) * (rstd ** 3)
        d_x2 = d_var * (2.0 / N) * x
        dx = dx + d_x2
    else:
        dx = dy * rstd
        d_rstd = (dy * (x - mean)).sum(dim=-1, keepdim=True)
        d_var = d_rstd * (-0.5) * (rstd ** 3)
        d_x2 = d_var * (2.0 / N) * (x - mean)
        d_mean = -dx.sum(dim=-1, keepdim=True) - d_x2.sum(dim=-1, keepdim=True)
        dx = dx + d_x2 + d_mean / N

    if dresidual is not None:
        dx += dresidual

    dx = dx.to(x_dtype) if x_dtype is not None else dx.to(dtype)
    dresidual_in = dx if has_residual and (x_dtype is None or dx.dtype == x_dtype) else None

    if recompute_output:
        if is_rms_norm:
            x_hat = x * rstd
        else:
            x_hat = (x - mean) * rstd
        if weight is not None:
            x_hat = x_hat * weight
        if bias is not None:
            x_hat = x_hat + bias
        y = activation_quant(x_hat)
        return dx, d_weight, d_bias, dresidual_in, y

    return dx, d_weight, d_bias, dresidual_in


class LayerNormLinearQuantFn(torch.autograd.Function):
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
            residual.dtype if residual is not None
            else (torch.float32 if residual_in_fp32 else None)
        )
        
        y, mean, rstd, residual_out = _layer_norm_fwd_quant(
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
        linear_weight = weight_quant(linear_weight).to(dtype)
        linear_bias = linear_bias.to(dtype) if linear_bias is not None else None
        
        out = F.linear(y.to(linear_weight.dtype), linear_weight, linear_bias)
        
        ctx.save_for_backward(
            residual_out, norm_weight, norm_bias, linear_weight, mean, rstd, y
        )
        ctx.x_shape_og = x_shape_og
        ctx.eps = eps
        ctx.is_rms_norm = is_rms_norm
        ctx.has_residual = residual is not None
        ctx.prenorm = prenorm
        ctx.x_dtype = x.dtype
        ctx.linear_bias_is_none = linear_bias is None
        ctx.y_shape = y.shape
        
        return out if not prenorm else (out, residual_out.reshape(x_shape_og))

    @staticmethod
    @contiguous
    def backward(ctx, dout, *args):
        dout = dout.reshape(-1, dout.shape[-1])
        (
            residual_out, norm_weight, norm_bias, linear_weight, mean, rstd, y
        ) = ctx.saved_tensors
        
        dy = F.linear(dout, linear_weight.t())
        dlinear_bias = None if ctx.linear_bias_is_none else dout.sum(0)
        
        if ctx.prenorm:
            dresidual = args[0].reshape(ctx.y_shape)
        else:
            dresidual = None
        
        dx, dnorm_weight, dnorm_bias, dresidual_in = _layer_norm_bwd(
            dy.reshape(ctx.y_shape),
            residual_out,
            norm_weight,
            norm_bias,
            ctx.eps,
            mean,
            rstd,
            dresidual,
            ctx.has_residual,
            ctx.is_rms_norm,
            x_dtype=ctx.x_dtype,
            recompute_output=False
        )
        
        dlinear_weight = torch.einsum("bo,bi->oi", dout, y.reshape(ctx.y_shape))
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


def layer_norm_linear_quant_fn(
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
    return LayerNormLinearQuantFn.apply(
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


class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False):
        super(BitLinear, self).__init__(in_features, out_features, bias=bias)
        self.norm = RMSNorm(in_features, eps=1e-8)

    def forward(self, x):
        w = self.weight
        x_norm = self.norm(x)
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        y = F.linear(x_quant, w_quant)
        return y


class FusedBitLinear(BitLinear):
    def __init__(self, in_features, out_features, bias=False):
        super(FusedBitLinear, self).__init__(in_features, out_features, bias=bias)

    def forward(self, x):
        return layer_norm_linear_quant_fn(
            x,
            self.norm.weight,
            self.norm.bias,
            self.weight,
            self.bias,
            is_rms_norm=True
        )