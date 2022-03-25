"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
"""

import torch
import torch.nn as nn

from mmdet.core import reduce_mean
from mmdet.models import LOSSES, weighted_loss


@weighted_loss
def monte_carlo_pose_loss(pose_sample_logweights, cost_target):
    """
    Args:
        pose_sample_logweights: Shape (mc_samples, num_obj)
        cost_target: Shape (num_obj, )

    Returns:
        Tensor: Shape (num_obj, )
    """
    loss_tgt = cost_target
    loss_pred = torch.logsumexp(pose_sample_logweights, dim=0)  # (num_obj, )

    loss_pose = loss_tgt + loss_pred  # (num_obj, )
    loss_pose[loss_pose.isnan()] = 0
    return loss_pose


@LOSSES.register_module()
class MonteCarloPoseLoss(nn.Module):

    def __init__(self, loss_weight=1.0, init_norm_factor=1.0, momentum=0.01,
                 reduction='mean'):
        super(MonteCarloPoseLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.register_buffer('norm_factor', torch.tensor(init_norm_factor, dtype=torch.float))
        self.momentum = momentum

    def forward(self, pose_sample_logweights, cost_target, norm_factor,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """
        Args:
            pose_sample_logweights: Shape (mc_samples, num_obj)
            cost_target: Shape (num_obj, )
            norm_factor: Shape ()
        """
        if self.training:
            with torch.no_grad():
                norm_factor = reduce_mean(norm_factor)
                self.norm_factor.mul_(
                    1 - self.momentum).add_(self.momentum * norm_factor)
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = monte_carlo_pose_loss(
            pose_sample_logweights,
            cost_target,
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor,
        ) * (self.loss_weight / self.norm_factor)
        return loss
