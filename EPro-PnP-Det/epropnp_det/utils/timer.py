"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
"""

import numpy as np
import torch
from mmcv import Timer


class IterTimer:
    def __init__(self, name='time', sync=True, enabled=True):
        self.name = name
        self.times = []
        self.timer = Timer(start=False)
        self.sync = sync
        self.enabled = enabled

    def __enter__(self):
        if not self.enabled:
            return
        if self.sync:
            torch.cuda.synchronize()
        self.timer.start()
        return self

    def __exit__(self, type, value, traceback):
        if not self.enabled:
            return
        if self.sync:
            torch.cuda.synchronize()
        self.timer_record()
        self.timer._is_running = False

    def timer_start(self):
        self.timer.start()

    def timer_record(self):
        self.times.append(self.timer.since_last_check())

    def print_time(self):
        if not self.enabled:
            return
        print('Average {} = {:.4f}'.format(self.name, np.average(self.times)))


class IterTimers(dict):
    def __init__(self, *args, **kwargs):
        super(IterTimers, self).__init__(*args, **kwargs)

    def disable_all(self):
        for timer in self.values():
            timer.enabled = False

    def enable_all(self):
        for timer in self.values():
            timer.enabled = True

    def add_timer(self, name='time', sync=True, enabled=False):
        self[name] = IterTimer(
            name, sync=sync, enabled=enabled)


default_timers = IterTimers()
