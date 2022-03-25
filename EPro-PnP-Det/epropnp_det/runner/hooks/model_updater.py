"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
This file is modified from
https://github.com/tjiiv-cprg/MonoRUn
"""

import functools
from mmcv.runner.hooks.hook import HOOKS, Hook


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


@HOOKS.register_module()
class ModelUpdaterHook(Hook):
    """
    Args:
        step (list[int])
        cfgs (list[dict])
        by_epoch (bool)
    """

    def __init__(self, step, cfgs, by_epoch=True):
        self.by_epoch = by_epoch
        assert isinstance(step, list) and isinstance(cfgs, list) and isinstance(cfgs[0], dict)
        self.step = step
        self.cfgs = cfgs
        self.current_step_id = 0

    def get_step_id(self, runner):
        progress = runner.epoch if self.by_epoch else runner.iter
        step_id = len(self.step)
        for i, s in enumerate(self.step):
            if progress < s:
                step_id = i
                break
        if step_id > self.current_step_id:  # step forward
            self.set_cfg(runner, step_id)
            self.current_step_id = step_id

    def set_cfg(self, runner, step_id):
        cfg = self.cfgs[step_id - 1]
        for key, value in cfg.items():
            rsetattr(runner.model.module, key, value)

    def before_train_iter(self, runner):
        if not self.by_epoch:
            self.get_step_id(runner)

    def before_train_epoch(self, runner):
        if self.by_epoch:
            self.get_step_id(runner)
