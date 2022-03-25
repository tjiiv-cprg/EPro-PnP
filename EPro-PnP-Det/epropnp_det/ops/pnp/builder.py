"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
"""

from mmcv.utils import Registry, build_from_cfg

PNP = Registry('pnp')
CAMERA = Registry('camera')
COSTFUN = Registry('cost_fun')


def build_pnp(cfg, **default_args):
    return build_from_cfg(cfg, PNP, default_args)

def build_camera(cfg, **default_args):
    return build_from_cfg(cfg, CAMERA, default_args)

def build_cost_fun(cfg, **default_args):
    return build_from_cfg(cfg, COSTFUN, default_args)
