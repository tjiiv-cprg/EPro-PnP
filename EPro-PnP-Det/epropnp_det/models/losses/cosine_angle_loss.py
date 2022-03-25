"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
"""

import torch
import torch.nn as nn

from mmdet.models import LOSSES, weighted_loss


@weighted_loss
def cosine_angle_loss(pred, target):
    loss = 1 - torch.cos(pred - target)
    return loss


@LOSSES.register_module()
class CosineAngleLoss(nn.Module):

    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * cosine_angle_loss(
            pred,
            target,
            weight,
            reduction=reduction,
            avg_factor=avg_factor)
        return loss
