"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
This file is modified from
https://github.com/tjiiv-cprg/MonoRUn
"""

import torch

from mmdet.models import LOSSES, SmoothL1Loss, weighted_loss


@weighted_loss
def smooth_l1_loss_mod(pred, target, beta=1.0, **kwargs):
    assert beta > 0
    if isinstance(target, int):
        if target == 0:
            diff = torch.abs(pred)
        elif target == -1:
            diff = pred
        else:
            raise ValueError
    else:
        diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    return loss


@LOSSES.register_module()
class SmoothL1LossMod(SmoothL1Loss):

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * smooth_l1_loss_mod(
            pred,
            target,
            weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox
