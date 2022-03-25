"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
This file is modified from
https://github.com/tjiiv-cprg/MonoRUn
"""

import os.path as osp

import mmcv
import numpy as np

from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile


@PIPELINES.register_module()
class LoadAnnotations3D(LoadAnnotations):

    def __init__(self,
                 with_bbox_3d=True,
                 with_coord_3d=True,
                 with_truncation=False,
                 with_attr=False,
                 with_velo=False, **kwargs):
        self.with_bbox_3d = with_bbox_3d
        self.with_coord_3d = with_coord_3d
        self.with_truncation = with_truncation
        self.with_attr = with_attr
        self.with_velo = with_velo
        super(LoadAnnotations3D, self).__init__(**kwargs)

    @staticmethod
    def _load_coord_3d(results):
        assert results['img_shape'] == results['ori_shape']
        crd_dict = mmcv.load(
            osp.join(results['coord_3d_prefix'], results['ann_info']['coord_3d']))
        if 'coord_3d_rot' in results['ann_info']:  # nuscenes
            # transposed for row vector rotation
            x3d_rot = results['ann_info']['coord_3d_rot'].T
        else:
            x3d_rot = None
        gt_x3d = []
        gt_x2d = []
        for i, bbox_3d in zip(results['ann_info']['object_ids'],
                              results['ann_info']['bboxes_3d']):
            x2d = crd_dict['uv_list'][i].astype(np.float32)
            x3d = crd_dict['oc_list'][i].astype(np.float32)
            if x3d_rot is not None:  # nuscenes
                x3d = x3d @ x3d_rot
            else:  # KITTI
                bh = bbox_3d[1]
                x3d[:, 1] += bh / 2
            gt_x3d.append(x3d)
            gt_x2d.append(x2d)
        results['gt_x3d'] = gt_x3d
        results['gt_x2d'] = gt_x2d
        return results

    @staticmethod
    def _load_bboxes_3d(results):
        results['gt_bboxes_3d'] = results['ann_info']['bboxes_3d']
        results['bbox_3d_fields'].append('gt_bboxes_3d')
        return results

    def __call__(self, results):
        results = super(LoadAnnotations3D, self).__call__(results)
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
        if self.with_coord_3d:
            results = self._load_coord_3d(results)
        if self.with_truncation:
            results['truncation'] = results['ann_info']['truncation']
        if self.with_attr:
            results['gt_attr'] = results['ann_info']['attr']
        if self.with_velo:
            results['gt_velo'] = results['ann_info']['velo']
        return results


@PIPELINES.register_module()
class LoadImageFromFile3D(LoadImageFromFile):

    def __init__(self,
                 with_img_dense_x2d=True,
                 with_depth=False,
                 **kwargs):
        self.with_img_dense_x2d = with_img_dense_x2d
        self.with_depth = with_depth
        super(LoadImageFromFile3D, self).__init__(**kwargs)

    @staticmethod
    def _load_depth(results):
        depth = mmcv.imread(
            osp.join(results['depth_prefix'],
                     results['img_info']['depth']),
            flag='unchanged')[..., None]  # (H, W, 1)
        ori_shape = results['ori_shape']
        if depth.shape[:2] != ori_shape[:2]:
            depth = mmcv.imresize(depth, (ori_shape[1], ori_shape[0]))
        results['depth'] = depth.astype(np.float32) / 256
        results['dense_fields'].append('depth')
        return results

    @staticmethod
    def _gen_img_dense_x2d(results):
        assert results['img_shape'] == results['ori_shape']
        h, w = results['img_shape'][:2]
        img_dense_x2d = np.mgrid[:h, :w].astype(np.float32)
        img_dense_x2d[[1, 0]] = img_dense_x2d[[0, 1]]  # to [u, v]
        results['img_dense_x2d'] = np.moveaxis(img_dense_x2d, 0, -1)  # (H, W, 2)
        results['img_dense_x2d_mask'] = np.ones((h, w, 1), dtype=np.float32)
        if 'dense_fields' in results:
            results['dense_fields'].append('img_dense_x2d')
            results['dense_fields'].append('img_dense_x2d_mask')
        else:
            results['dense_fields'] = ['img_dense_x2d', 'img_dense_x2d_mask']
        return results

    def __call__(self, results):
        results = super(LoadImageFromFile3D, self).__call__(results)
        if 'cam_intrinsic' in results['img_info']:
            results['cam_intrinsic'] = results['img_info']['cam_intrinsic']
        elif 'ann_info' in results and 'cam_intrinsic' in results['ann_info']:
            results['cam_intrinsic'] = results['ann_info']['cam_intrinsic']
        else:
            raise ValueError('cam_intrinsic not found')
        if self.with_img_dense_x2d:
            results = self._gen_img_dense_x2d(results)
        if self.with_depth:
            results = self._load_depth(results)
        if 'cam_id' in results['img_info']:
            results['cam_id'] = results['img_info']['cam_id']
        return results
