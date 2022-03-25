"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
This file is modified from
https://github.com/tjiiv-cprg/MonoRUn
"""

import torch
from ..builder import PROJ_ERROR_CODERS


@PROJ_ERROR_CODERS.register_module()
class DistDimProjErrorCoder(object):

    def __init__(self,
                 target_std=0.2,
                 distance_min=0.1):
        self.target_std = target_std
        self.distance_min = distance_min

    def encode(self, x2d_diff, distance, dimensions, focal):
        """
        Args:
            x2d_diff (torch.Tensor): Shape (*, num_points, 2)
            distance (torch.Tensor): Shape (*, 1)
            dimensions (torch.Tensor): Shape (*, 3)
            focal (torch.Tensor): Shape (*, 1)

        Returns:
            torch.Tensor: Encoded projection error or std
        """
        length = torch.mean(dimensions, dim=-1, keepdim=True)
        denom = length * focal * self.target_std
        distance = distance.clamp(min=self.distance_min)
        proj_error = x2d_diff * (distance / denom)[..., None, :]
        return proj_error

    def decode(self, proj_error, distance, dimensions, focal):
        """
        Args:
            proj_error (torch.Tensor): Shape (*, num_points, 2)
            distance (torch.Tensor): Shape (*, 1)
            dimensions (torch.Tensor): Shape (*, 3)
            focal (torch.Tensor): Shape (*, 1)

        Returns:
            torch.Tensor: Encoded projection error or std
        """
        length = torch.mean(dimensions, dim=-1, keepdim=True)
        denom = length * focal * self.target_std
        distance = distance.clamp(min=self.distance_min)
        x2d_diff = proj_error * (denom / distance)[..., None, :]
        return x2d_diff
