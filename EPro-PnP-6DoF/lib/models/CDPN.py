"""
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
This file is modified from
https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi
"""

import torch.nn as nn

from models.monte_carlo_pose_loss import MonteCarloPoseLoss


class CDPN(nn.Module):
    def __init__(self, backbone, rot_head_net, trans_head_net):
        super(CDPN, self).__init__()
        self.backbone = backbone
        self.rot_head_net = rot_head_net
        self.trans_head_net = trans_head_net
        self.monte_carlo_pose_loss = MonteCarloPoseLoss()

    def forward(self, x):                     # x.shape [bs, 3, 256, 256]
        features = self.backbone(x)           # features.shape [bs, 2048, 8, 8]
        cc_maps = self.rot_head_net(features) # joints.shape [bs, 1152, 64, 64]
        trans = self.trans_head_net(features)
        return cc_maps, trans
