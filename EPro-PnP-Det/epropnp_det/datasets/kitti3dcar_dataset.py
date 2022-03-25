"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
This file is modified from
https://github.com/tjiiv-cprg/MonoRUn
"""

from .kitti3d_dataset import KITTI3DDataset

from mmdet.datasets import DATASETS


@DATASETS.register_module()
class KITTI3DCarDataset(KITTI3DDataset):
    CLASSES = ('Car', )
