# -*- coding: utf-8 -*-

# Copyright (c) 2023, Songlin Yang

# Edited by MengAiDev, 2025

# Benchmark results on 4-core AMD EPYC 7763 64-Core Processor:

# Forward pass comparison:
# Output max diff: 0.015625
# Final state max diff: 0.015625

# Backward pass comparison:
# dx max diff: 0.03125
# dg max diff: 0.09375

# Performance Benchmark:
# Seq Len Recurrent (ms)  Recurrent Bwd (ms)
# 128     3.97            6.72
# 256     7.97            13.48
# 512     15.71           27.66
# 1024    35.06           55.80
# 2048    81.36           129.07
# 4096    137.68          230.54
# 8192    284.49          506.31
# 16384   544.93          1018.15

from typing import Tuple, Optional
import torch
import torch.nn as nn

class FusedRecurrentHGRNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, g, initial_state=None, output_final_state=False):
        # x: [B, H, T, D], g: [B, H, T, D]
        B, H, T, D = x.shape
        
        # 确保张量在内存中连续
        x = x.contiguous()
        g = g.contiguous()
        if initial_state is not None:
            initial_state = initial_state.contiguous()
        
        # 初始化输出和最终状态
        o = torch.empty_like(x)
        final_state = None
        if output_final_state:
            final_state = torch.empty(B, H, D, dtype=x.dtype, device=x.device)
        
        # 初始化隐藏状态
        h = torch.zeros(B, H, D, dtype=torch.float32, device=x.device)
        if initial_state is not None:
            h = initial_state.clone().to(torch.float32)
        
        # 逐时间步计算前向传播
        for t in range(T):
            h = g[:, :, t] * h + x[:, :, t]
            o[:, :, t] = h
        
        if output_final_state:
            final_state.copy_(h)
        
        # 保存反向传播所需的值
        ctx.save_for_backward(g, o, initial_state)
        ctx.output_final_state = output_final_state
        return o, final_state

    @staticmethod
    def backward(ctx, do, dht=None):
        # 获取保存的张量
        g, o, initial_state = ctx.saved_tensors
        B, H, T, D = do.shape
        
        # 确保梯度张量连续
        do = do.contiguous()
        
        # 初始化梯度
        dx = torch.empty_like(do)
        dg = torch.empty_like(g)
        dh = torch.zeros(B, H, D, dtype=torch.float32, device=do.device)
        
        # 如果提供了最终状态的梯度，则合并到最后一个时间步
        if dht is not None:
            do = do.clone()
            do[:, :, -1] += dht
        
        # 反向时间步计算梯度
        for t in range(T-1, -1, -1):
            # 计算当前时间步的梯度
            dh = dh + do[:, :, t]
            dx[:, :, t] = dh
            dg[:, :, t] = dh * (initial_state if t == 0 and initial_state is not None else o[:, :, t-1])
            
            # 更新隐藏状态梯度
            dh = dh * g[:, :, t]
        
        return dx, dg, None, None

def fused_recurrent_hgrn(
    x: torch.Tensor,
    g: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    if initial_state is not None:
        initial_state = initial_state.detach()
    o, final_state = FusedRecurrentHGRNFunction.apply(x, g, initial_state, output_final_state)
    return o, final_state


if __name__ == '__main__':
    from copy import deepcopy
    B, H, T, D = 8, 4, 512, 128
    dtype = torch.bfloat16
    
    # 生成测试数据
    torch.manual_seed(42)
    x = torch.randn((B, H, T, D), dtype=dtype, device='cpu')
    g = torch.randn((B, H, T, D), dtype=dtype, device='cpu').sigmoid()
    x = (1 - g) * x
    do = torch.randn_like(x)
    h0 = torch.randn_like(x[:, :, 0])
    
    # 保存原始数据用于比较
    x_ref, g_ref = deepcopy(x).requires_grad_(), deepcopy(g).requires_grad_()
    x_tri, g_tri = deepcopy(x).requires_grad_(), deepcopy(g).requires_grad_()
    
    # 参考实现 (PyTorch原生实现)
    def naive_recurrent_hgrn(x, g, initial_state=None, output_final_state=False):
        B, H, T, D = x.shape
        o = torch.empty_like(x)
        h = torch.zeros(B, H, D, dtype=x.dtype, device=x.device)
        if initial_state is not None:
            h = initial_state.clone()
        
        for t in range(T):
            h = g[:, :, t] * h + x[:, :, t]
            o[:, :, t] = h
        
        final_state = h if output_final_state else None
        return o, final_state
    
    # 运行参考实现
    ref, ref_ht = naive_recurrent_hgrn(x_ref, g_ref, h0, output_final_state=True)
    ref.backward(do)
    ref_dx, ref_dg = x_ref.grad, g_ref.grad
    
    # 运行迁移后的实现
    tri, tri_ht = fused_recurrent_hgrn(x_tri, g_tri, h0, output_final_state=True)
    tri.backward(do)
    tri_dx, tri_dg = x_tri.grad, g_tri.grad
    
    # 比较结果
    print("Forward pass comparison:")
    print("Output max diff:", torch.max(torch.abs(ref - tri)).item())
    print("Final state max diff:", torch.max(torch.abs(ref_ht - tri_ht)).item())
    
    print("\nBackward pass comparison:")
    print("dx max diff:", torch.max(torch.abs(ref_dx - tri_dx)).item())
    print("dg max diff:", torch.max(torch.abs(ref_dg - tri_dg)).item())
    
    # 性能测试
    import time
    seq_lens = [128 * 2 ** i for i in range(0, 8)]
    
    print("\nPerformance Benchmark:")
    print("Seq Len\tRecurrent (ms)\tRecurrent Bwd (ms)")
    
    for seq_len in seq_lens:
        # 准备数据
        x = torch.randn((B, H, seq_len, D), dtype=dtype, device='cpu')
        g = torch.randn((B, H, seq_len, D), dtype=dtype, device='cpu').sigmoid()
        x = (1 - g) * x
        x, g = x.requires_grad_(), g.requires_grad_()
        do = torch.randn_like(x)
        
        # 前向传播性能
        start = time.time()
        o, _ = fused_recurrent_hgrn(x, g)
        fwd_time = (time.time() - start) * 1000
        
        # 反向传播性能
        start = time.time()
        o.backward(do)
        bwd_time = (time.time() - start) * 1000
        
        print(f"{seq_len}\t{fwd_time:.2f}\t\t{bwd_time:.2f}")