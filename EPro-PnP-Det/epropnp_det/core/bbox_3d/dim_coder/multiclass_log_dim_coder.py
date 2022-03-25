"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
This file is modified from
https://github.com/tjiiv-cprg/MonoRUn
"""

import numpy as np
from ..builder import DIM_CODERS


@DIM_CODERS.register_module()
class MultiClassLogDimCoder(object):

    def __init__(self,
                 target_means=[
                     (4.62, 1.73, 1.96),
                     (6.94, 2.84, 2.52),
                     (12.56, 3.89, 2.94),
                     (11.22, 3.50, 2.95),
                     (6.68, 3.21, 2.85),
                     (1.70, 1.29, 0.61),
                     (2.11, 1.46, 0.78),
                     (0.73, 1.77, 0.67),
                     (0.41, 1.08, 0.41),
                     (0.50, 0.99, 2.52)],
                 target_stds=[
                     (0.46, 0.24, 0.16),
                     (2.11, 0.84, 0.45),
                     (4.50, 0.77, 0.54),
                     (2.06, 0.49, 0.33),
                     (3.23, 0.93, 1.07),
                     (0.26, 0.35, 0.16),
                     (0.33, 0.29, 0.17),
                     (0.19, 0.19, 0.14),
                     (0.14, 0.27, 0.13),
                     (0.17, 0.15, 0.62)]):
        super(MultiClassLogDimCoder, self).__init__()
        assert len(target_means) == len(target_stds)
        self.target_means = np.array(target_means, dtype=np.float32)
        self.target_stds = np.array(target_stds, dtype=np.float32)
        self.logtarget_means = np.log(self.target_means)
        self.logtarget_stds = self.target_stds / self.target_means

    def encode(self, dimensions, labels):
        logtarget_means = dimensions.new_tensor(self.logtarget_means)[labels]
        logtarget_stds = dimensions.new_tensor(self.logtarget_stds)[labels]
        dim_enc = (dimensions.log() - logtarget_means) / logtarget_stds
        return dim_enc

    def decode(self, dim, labels):
        logtarget_means = dim.new_tensor(self.logtarget_means)[labels]
        logtarget_stds = dim.new_tensor(self.logtarget_stds)[labels]
        dimensions = (dim * logtarget_stds + logtarget_means).exp()
        return dimensions
