"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
"""

from mmdet.datasets import build_dataset as build_dataset_old
from . import CBGSDataset


def build_dataset(cfg, default_args=None):
    if not isinstance(cfg, (list, tuple)) and cfg['type'] == 'CBGSDataset':
        dataset = CBGSDataset(build_dataset(cfg['dataset'], default_args))
    else:
        dataset = build_dataset_old(cfg, default_args)
    return dataset
