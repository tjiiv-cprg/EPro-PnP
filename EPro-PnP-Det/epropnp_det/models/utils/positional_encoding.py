"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
"""

import torch
from mmcv.cnn.bricks.transformer import POSITIONAL_ENCODING

from mmdet.models.utils import SinePositionalEncoding


@POSITIONAL_ENCODING.register_module()
class SinePositionalEncodingMod(SinePositionalEncoding):

    def points_to_enc(self, points, img_sizes):
        """
        Args:
            points: (*, 2) in [x, y]
            img_sizes: (*, 2) in [h, w]

        Returns:
            pos (Tensor): Returned position embedding with shape
                (*, num_feats*2).
        """
        batch_size = points.shape[:-1]
        if self.normalize:
            # assume points already have offsets
            points = points / img_sizes[..., [1, 0]] * self.scale
        x_embed, y_embed = points.split(1, dim=-1)
        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32, device=points.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)
        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_x = torch.stack(
            (pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
            dim=-1).view(*batch_size, self.num_feats)
        pos_y = torch.stack(
            (pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()),
            dim=-1).view(*batch_size, self.num_feats)
        pos = torch.cat((pos_y, pos_x), dim=-1)
        return pos
