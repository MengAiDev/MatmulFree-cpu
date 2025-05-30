# -*- coding: utf-8 -*-

# Copyright (c) 2023, Tri Dao.

# edited by MengAiDev, 2025

from typing import Tuple

import torch
import torch.nn as nn

# `all_gather_into_tensor` and `reduce_scatter_tensor` are new placeholders for
# `_all_gather_base` and `_reduce_scatter_base`. They require the most recent
# version of PyTorch. The following 2 lines are for backward compatibility with
# older PyTorch.
if "all_gather_into_tensor" not in dir(torch.distributed):
    torch.distributed.all_gather_into_tensor = torch.distributed._all_gather_base

class CrossEntropyLossFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        logits,
        labels,
        smoothing=0.0,
        logit_scale=1.0,
        lse_square_scale=0.0,
        ignored_index=-100,
        inplace_backward=False,
        process_group=None,
    ):
        n_rows, n_cols = logits.shape
        assert labels.shape == (n_rows,)
        world_size = 1 if process_group is None else torch.distributed.get_world_size(process_group)
        total_classes = world_size * n_cols
        rank = 0 if process_group is None else torch.distributed.get_rank(process_group)
        class_start_idx = rank * n_cols

        logits_scaled = logits * logit_scale
        lse_local = torch.logsumexp(logits_scaled, dim=-1)  # 计算局部LogSumExp

        if world_size > 1:  # 张量并行处理
            # 收集所有设备的LSE
            lse_all = torch.empty(world_size, n_rows, dtype=lse_local.dtype, device=lse_local.device)
            torch.distributed.all_gather_into_tensor(lse_all, lse_local, group=process_group)
            lse_global = torch.logsumexp(lse_all, dim=0)  # 全局LogSumExp

            # 计算局部损失贡献
            local_loss = torch.zeros(n_rows, device=logits.device)
            mask_local = (labels != ignored_index)
            in_partition = (labels >= class_start_idx) & (labels < class_start_idx + n_cols)
            local_labels = labels - class_start_idx

            if smoothing > 0:
                sum_logits_local = logits_scaled.sum(dim=-1)
                local_loss[mask_local] = -smoothing * (sum_logits_local[mask_local] / total_classes)
                mask_in = mask_local & in_partition
                if mask_in.any():
                    selected_logits = logits_scaled[torch.arange(n_rows)[mask_in], local_labels[mask_in]]
                    local_loss[mask_in] -= (1 - smoothing) * selected_logits
            else:
                mask_in = mask_local & in_partition
                if mask_in.any():
                    selected_logits = logits_scaled[torch.arange(n_rows)[mask_in], local_labels[mask_in]]
                    local_loss[mask_in] = -selected_logits

            # 聚合所有设备的损失
            torch.distributed.all_reduce(local_loss, op=torch.distributed.ReduceOp.SUM, group=process_group)

            # 添加全局LSE和z-loss
            loss = torch.zeros(n_rows, device=logits.device)
            loss[mask_local] = local_loss[mask_local] + lse_global[mask_local]
            z_loss = torch.zeros(n_rows, device=logits.device)
            if lse_square_scale > 0:
                z_loss[mask_local] = lse_square_scale * (lse_global[mask_local].square())
                loss[mask_local] += z_loss[mask_local]
            lse_saved = lse_global  # 保存全局LSE用于反向传播
        else:  # 单设备处理
            mask = (labels != ignored_index)
            loss = torch.zeros(n_rows, device=logits.device)
            z_loss = torch.zeros(n_rows, device=logits.device)

            if mask.any():
                if smoothing > 0:
                    correct_logits = logits_scaled[torch.arange(n_rows)[mask], labels[mask]]
                    sum_logits = logits_scaled.sum(dim=-1)[mask]
                    loss[mask] = lse_local[mask] - (1 - smoothing) * correct_logits - (smoothing / total_classes) * sum_logits
                else:
                    correct_logits = logits_scaled[torch.arange(n_rows)[mask], labels[mask]]
                    loss[mask] = lse_local[mask] - correct_logits
                
                if lse_square_scale > 0:
                    z_loss[mask] = lse_square_scale * (lse_local[mask].square())
                    loss[mask] += z_loss[mask]
            lse_saved = lse_local  # 保存局部LSE用于反向传播

        ctx.save_for_backward(logits, lse_saved, labels)
        ctx.mark_non_differentiable(z_loss)
        ctx.smoothing = smoothing
        ctx.logit_scale = logit_scale
        ctx.lse_square_scale = lse_square_scale
        ctx.ignored_index = ignored_index
        ctx.total_classes = total_classes
        ctx.class_start_idx = class_start_idx
        ctx.world_size = world_size
        ctx.n_cols = n_cols

        return loss, z_loss

    @staticmethod
    def backward(ctx, grad_losses, grad_z_losses):
        logits, lse, labels = ctx.saved_tensors
        n_rows, n_cols = logits.shape
        logits_scaled = logits * ctx.logit_scale

        # 计算概率分布
        probs = torch.exp(logits_scaled - lse.unsqueeze(-1))
        
        # 创建本地one-hot标签
        one_hot_local = torch.zeros_like(logits)
        mask_non_ignored = (labels != ctx.ignored_index)
        
        if ctx.world_size > 1:  # 张量并行处理
            in_partition = (labels >= ctx.class_start_idx) & (labels < ctx.class_start_idx + ctx.n_cols)
            mask = mask_non_ignored & in_partition
            if mask.any():
                local_labels = labels[mask] - ctx.class_start_idx
                one_hot_local[torch.arange(n_rows)[mask], local_labels] = 1.0
        else:  # 单设备处理
            if mask_non_ignored.any():
                one_hot_local[torch.arange(n_rows)[mask_non_ignored], labels[mask_non_ignored]] = 1.0

        # 计算梯度
        term = probs * (1 + 2 * ctx.lse_square_scale * lse.unsqueeze(-1))
        term -= (1 - ctx.smoothing) * one_hot_local
        term -= ctx.smoothing / ctx.total_classes
        dlogits_scaled = grad_losses.unsqueeze(-1) * term
        dlogits = dlogits_scaled * ctx.logit_scale

        return dlogits, None, None, None, None, None, None, None, None

def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float = 0.0,
    logit_scale: float = 1.0,
    lse_square_scale: float = 0.0,
    ignored_index=-100,
    inplace_backward: bool = False,
    process_group=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return CrossEntropyLossFunction.apply(
        logits,
        labels,
        label_smoothing,
        logit_scale,
        lse_square_scale,
        ignored_index,
        inplace_backward,
        process_group,
    )

class FusedCrossEntropyLoss(nn.Module):
    def __init__(
        self,
        ignore_index=-100,
        reduction="mean",
        label_smoothing=0.0,
        logit_scale=1.0,
        lse_square_scale=0.0,
        inplace_backward=False,
        process_group=None,
        return_z_loss=False,
    ):
        super().__init__()
        if reduction not in ["mean", "none", "sum"]:
            raise NotImplementedError("Only support reduction = 'mean' or 'none' or 'sum'")
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.logit_scale = logit_scale
        self.lse_square_scale = lse_square_scale
        self.inplace_backward = inplace_backward
        self.process_group = process_group
        self.return_z_loss = return_z_loss

    def forward(self, input, target):
        loss, z_loss = cross_entropy_loss(
            input,
            target,
            label_smoothing=self.label_smoothing,
            logit_scale=self.logit_scale,
            lse_square_scale=self.lse_square_scale,
            ignored_index=self.ignore_index,
            inplace_backward=self.inplace_backward,
            process_group=self.process_group,
        )
        
        if self.reduction == "mean":
            loss = loss.sum() / (target != self.ignore_index).sum()
        elif self.reduction == "sum":
            loss = loss.sum()
            
        if not self.return_z_loss:
            return loss

        if self.reduction == "mean":
            z_loss = z_loss.sum() / (target != self.ignore_index).sum()
        elif self.reduction == "sum":
            z_loss = z_loss.sum()
            
        return loss, z_loss