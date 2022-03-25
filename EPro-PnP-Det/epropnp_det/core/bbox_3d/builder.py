"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
"""

from mmcv.utils import Registry, build_from_cfg

DIM_CODERS = Registry('dim_coder')
PROJ_ERROR_CODERS = Registry('proj_error_coder')
CENTER_TARGETS = Registry('center_target')


def build_dim_coder(cfg, **default_args):
    return build_from_cfg(cfg, DIM_CODERS, default_args)


def build_proj_error_coder(cfg, **default_args):
    return build_from_cfg(cfg, PROJ_ERROR_CODERS, default_args)


def build_center_target(cfg, **default_args):
    return build_from_cfg(cfg, CENTER_TARGETS, default_args)
