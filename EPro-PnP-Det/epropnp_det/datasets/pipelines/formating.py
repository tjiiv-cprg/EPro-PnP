"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
This file is modified from
https://github.com/tjiiv-cprg/MonoRUn
"""

import numpy as np
from mmcv.parallel import DataContainer as DC

from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines.formating import DefaultFormatBundle, to_tensor


@PIPELINES.register_module()
class DefaultFormatBundle3D(DefaultFormatBundle):

    def __call__(self, results):
        results = super().__call__(results)
        for key in ['gt_bboxes_3d', 'cam_intrinsic', 'gt_attr', 'gt_velo']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]))
        for key in ['gt_x3d', 'gt_x2d']:
            if key not in results:
                continue
            results[key] = DC([to_tensor(key_single) for key_single in results[key]])
        for key in ['img_dense_x2d', 'img_dense_x2d_mask', 'depth', 'gt_cls_masks']:
            if key in results:
                value = results[key]
                value = np.ascontiguousarray(value.transpose(2, 0, 1)) \
                    if len(value.shape) == 3 else value[None, ...]
                results[key] = DC(to_tensor(value), stack=True)
        return results
