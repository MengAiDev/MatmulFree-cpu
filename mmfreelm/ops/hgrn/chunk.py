# -*- coding: utf-8 -*-

# Copyright (c) 2024, Yu Zhang, Songlin Yang

# edited by MengAiDev, 2025

# Benchmark results on 4 core AMD EPYC 7763 64-Core Processor

# Benchmarking CPU implementation...
# Sequence Length Chunk Fwd (ms)  Chunk Bwd (ms) 
# 128             13.11           20.20          
# 256             28.95           31.76          
# 512             54.14           62.64          
# 1024            114.02          125.12         
# 2048            239.29          255.63         
# 4096            511.81          551.23         
# 8192            1014.24         1035.15        


from typing import Tuple
import torch
import torch.nn.functional as F
from torch.autograd import Function

class ChunkHGRNFunction(Function):
    @staticmethod
    def forward(ctx, x, g, initial_state=None, output_final_state=False):
        B, H, T, D = x.shape
        BT = 128  # 分块大小
        
        # 初始化输出和g的累积
        o = torch.zeros_like(x, dtype=torch.float32)
        gc = torch.zeros_like(g, dtype=torch.float32)
        
        # 如果没有初始状态，初始化为零
        h = torch.zeros(B, H, D, dtype=torch.float32, device=x.device)
        if initial_state is not None:
            h = initial_state.clone()
        
        num_chunks = (T + BT - 1) // BT
        chunk_final_states = torch.zeros(B, H, num_chunks, D, dtype=torch.float32, device=x.device)
        
        # 第一遍：处理每个块内部
        for i in range(num_chunks):
            start = i * BT
            end = min((i+1)*BT, T)
            chunk_len = end - start
            
            # 获取当前块的数据
            x_chunk = x[:, :, start:end, :]
            g_chunk = g[:, :, start:end, :]
            
            # 初始化当前块的隐藏状态和g累积
            h_chunk = torch.zeros(B, H, chunk_len, D, dtype=torch.float32, device=x.device)
            gc_chunk = torch.zeros(B, H, chunk_len, D, dtype=torch.float32, device=x.device)
            
            # 处理块内每个时间步
            for t in range(chunk_len):
                prev_h = h if t == 0 else h_chunk[:, :, t-1, :]
                h_current = torch.exp(g_chunk[:, :, t, :]) * prev_h + x_chunk[:, :, t, :]
                gc_current = g_chunk[:, :, t, :] if t == 0 else gc_chunk[:, :, t-1, :] + g_chunk[:, :, t, :]
                
                h_chunk[:, :, t, :] = h_current
                gc_chunk[:, :, t, :] = gc_current
            
            # 保存当前块的输出和g累积
            o[:, :, start:end, :] = h_chunk
            gc[:, :, start:end, :] = gc_chunk
            chunk_final_states[:, :, i, :] = h_chunk[:, :, -1, :]
            
            # 重置下一个块的初始状态
            h = torch.zeros(B, H, D, dtype=torch.float32, device=x.device)
        
        # 第二遍：跨块修正
        if num_chunks > 1:
            for i in range(1, num_chunks):
                start = i * BT
                end = min((i+1)*BT, T)
                chunk_len = end - start
                
                # 获取前一个块的最终状态
                prev_final = o[:, :, start-1, :]
                gc_chunk = gc[:, :, start:end, :]
                
                # 应用修正
                correction = prev_final.unsqueeze(2) * torch.exp(gc_chunk)
                o[:, :, start:end, :] += correction
        
        # 处理最终状态
        final_state = o[:, :, -1, :].clone() if output_final_state else None
        o = o.to(x.dtype)
        
        # 保存反向传播所需变量
        ctx.save_for_backward(g, o, gc)
        ctx.initial_state = initial_state
        return o, final_state

    @staticmethod
    def backward(ctx, do, d_final_state=None):
        g, o, gc = ctx.saved_tensors
        B, H, T, D = do.shape
        BT = 128
        num_chunks = (T + BT - 1) // BT
        
        # 初始化梯度
        dx = torch.zeros_like(o, dtype=torch.float32)
        dg = torch.zeros_like(g, dtype=torch.float32)
        
        # 处理最终状态的梯度
        if d_final_state is not None:
            do = do.clone()
            do[:, :, -1, :] += d_final_state
        
        # 初始化反向传播的隐藏状态
        dh_next = torch.zeros(B, H, D, dtype=torch.float32, device=do.device)
        
        # 反向传播：跨块修正
        for i in range(num_chunks-1, -1, -1):
            start = i * BT
            end = min((i+1)*BT, T)
            chunk_len = end - start
            
            # 获取当前块的数据
            do_chunk = do[:, :, start:end, :].float()
            g_chunk = g[:, :, start:end, :].float()
            o_chunk = o[:, :, start:end, :].float()
            gc_chunk = gc[:, :, start:end, :].float()
            
            # 如果不是最后一个块，添加跨块梯度
            if i < num_chunks - 1:
                # 下一个块的第一个位置的dx
                dx_next = dx[:, :, end, :]
                # 应用到当前块的每个位置
                do_chunk += dx_next.unsqueeze(2) * torch.exp(gc_chunk)
            
            # 当前块内部的反向传播
            dx_chunk = torch.zeros_like(do_chunk)
            dg_chunk = torch.zeros_like(g_chunk)
            dh = dh_next.clone()
            
            # 从后向前处理块内时间步
            for t in range(chunk_len-1, -1, -1):
                # 计算当前梯度
                dh += do_chunk[:, :, t, :]
                dx_current = dh.clone()
                
                # 计算g的梯度
                if t > 0:
                    prev_o = o_chunk[:, :, t-1, :]
                else:
                    prev_o = o[:, :, start-1, :] if i > 0 else torch.zeros(B, H, D, device=o.device)
                
                dg_current = prev_o * dx_current * torch.exp(g_chunk[:, :, t, :])
                
                # 更新隐藏状态梯度
                dh = dh * torch.exp(g_chunk[:, :, t, :])
                
                # 保存结果
                dx_chunk[:, :, t, :] = dx_current
                dg_chunk[:, :, t, :] = dg_current
            
            # 保存当前块的梯度
            dx[:, :, start:end, :] = dx_chunk
            dg[:, :, start:end, :] = dg_chunk
            dh_next = dh.clone()
        
        # 处理初始状态的梯度
        if ctx.initial_state is not None:
            dg[:, :, 0, :] += ctx.initial_state * dx[:, :, 0, :] * torch.exp(g[:, :, 0, :])
        
        return dx.to(o.dtype), dg.to(g.dtype), None, None

def chunk_hgrn(
    x: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    if initial_state is not None:
        initial_state = initial_state.detach()
    o, final_state = ChunkHGRNFunction.apply(x, g, initial_state, output_final_state)
    return o, final_state

if __name__ == '__main__':
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    
    B, H, D = 16, 4, 128
    seq_lens = [128, 256, 512, 1024, 2048, 4096, 8192]
    results = {
        'chunk_fwd': [],
        'chunk_bwd': [],
    }
    
    print("Benchmarking CPU implementation...")
    print(f"{'Sequence Length':<15} {'Chunk Fwd (ms)':<15} {'Chunk Bwd (ms)':<15}")
    
    for seq_len in seq_lens:
        # 准备数据
        x = torch.randn((B, H, seq_len, D), dtype=torch.bfloat16, device='cpu')
        g = torch.randn((B, H, seq_len, D), dtype=torch.bfloat16, device='cpu').sigmoid()
        x = (1 - g) * x
        do = torch.randn_like(x, dtype=torch.bfloat16)
        
        # 预热
        for _ in range(3):
            x1, g1 = x.clone().requires_grad_(), g.clone().requires_grad_()
            o, _ = chunk_hgrn(x1, g1)
            o.backward(do)
        
        # 前向传播计时
        fwd_time = 0.0
        for _ in range(10):
            x1, g1 = x.clone().requires_grad_(), g.clone().requires_grad_()
            start = time.time()
            o, _ = chunk_hgrn(x1, g1)
            fwd_time += (time.time() - start) * 1000
        fwd_time_avg = fwd_time / 10
        
        # 反向传播计时
        bwd_time = 0.0
        for _ in range(10):
            x1, g1 = x.clone().requires_grad_(), g.clone().requires_grad_()
            o, _ = chunk_hgrn(x1, g1)
            
            start = time.time()
            o.backward(do)
            bwd_time += (time.time() - start) * 1000
        bwd_time_avg = bwd_time / 10
        
        # 记录结果
        results['chunk_fwd'].append(fwd_time_avg)
        results['chunk_bwd'].append(bwd_time_avg)
        
        print(f"{seq_len:<15} {fwd_time_avg:<15.2f} {bwd_time_avg:<15.2f}")
