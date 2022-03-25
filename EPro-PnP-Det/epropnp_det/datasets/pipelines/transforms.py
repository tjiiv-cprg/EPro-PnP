"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
This file is modified from
https://github.com/tjiiv-cprg/MonoRUn
"""

import numpy as np
from numpy import random
import mmcv

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from mmdet.datasets import PIPELINES
from mmdet.datasets.pipelines import (Resize, RandomFlip, Pad, RandomCrop,
                                      MinIoURandomCrop, CutOut)


@PIPELINES.register_module()
class Resize3D(Resize):

    def _resize_dense(self, results):
        w_scale, h_scale = results['scale_factor'][:2]
        interpolation = 'bilinear' if np.sqrt(w_scale * h_scale) >= 1 \
            else 'area'
        for key in results.get('dense_fields', []):
            op = mmcv.imrescale if self.keep_ratio else mmcv.imresize
            dense = results[key]
            if isinstance(dense, list):
                results[key] = [
                    op(dense_single, results['scale'], interpolation=interpolation)
                    for dense_single in dense]
            else:
                results[key] = op(
                    dense, results['scale'], interpolation=interpolation)

    def __call__(self, results):
        results = super().__call__(results)
        self._resize_dense(results)
        return results


@PIPELINES.register_module()
class RandomFlip3D(RandomFlip):

    def __call__(self, results):
        results = super().__call__(results)
        if results['flip']:
            for key in results.get('dense_fields', []):
                dense = results[key]
                if isinstance(dense, list):
                    results[key] = [
                        mmcv.imflip(
                            dense_single, direction=results['flip_direction'])
                        for dense_single in dense]
                else:
                    results[key] = mmcv.imflip(
                        dense, direction=results['flip_direction'])
        return results


@PIPELINES.register_module()
class Pad3D(Pad):

    @staticmethod
    def _pad_dense(results):
        for key in results.get('dense_fields', []):
            dense = results[key]
            padding_kwargs = dict(padding_mode='edge') if key in ['img_dense_x2d', 'depth'] \
                else dict(padding_mode='constant', pad_val=0)
            if isinstance(dense, list):
                results[key] = [
                    mmcv.impad(
                        dense_single, shape=results['pad_shape'][:2], **padding_kwargs)
                    for dense_single in dense]
            else:
                results[key] = mmcv.impad(
                    dense, shape=results['pad_shape'][:2], **padding_kwargs)

    def __call__(self, results):
        results = super().__call__(results)
        self._pad_dense(results)
        return results


def crop_3d(results, crop_box, bbox2mask, bbox2label, bbox2bbox_3d,
            allow_negative_crop=False, trunc_ignore_thres=-1.0):
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
    # crop the image
    img = results['img']
    img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
    img_shape = img.shape
    results['img'] = img
    results['img_shape'] = img_shape

    # offset bboxes and crop masks
    for key in results.get('bbox_fields', []):
        # e.g. gt_bboxes and gt_bboxes_ignore
        bbox_offset = np.array([crop_x1, crop_y1, crop_x1, crop_y1],
                               dtype=np.float32)
        results[key] = results[key] - bbox_offset
        # crop mask fields, e.g. gt_masks and gt_masks_ignore
        mask_key = bbox2mask.get(key)
        if mask_key in results:
            results[mask_key] = results[mask_key].crop(
                np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))

    # clip to the image boundary, set valid inds, move the truncated to ignore
    for key in results.get('bbox_fields', []):
        bboxes_ori = results[key]
        bboxes = np.empty_like(bboxes_ori)
        bboxes[:, 0::2] = np.clip(bboxes_ori[:, 0::2], 0, img_shape[1])
        bboxes[:, 1::2] = np.clip(bboxes_ori[:, 1::2], 0, img_shape[0])
        valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
            bboxes[:, 3] > bboxes[:, 1])
        # If the crop does not contain any gt-bbox area and
        # allow_negative_crop is False, skip this image.
        if (key == 'gt_bboxes' and not valid_inds.any()
                and not allow_negative_crop):
            return None
        if key == 'gt_bboxes' and trunc_ignore_thres > 0:
            bboxes_ori_area = np.prod(
                bboxes_ori[:, 2:] - bboxes_ori[:, :2], axis=1)
            if 'truncation' in results:
                truncation = np.array(
                    results['truncation'], dtype=np.float32)
                bboxes_ori_area /= (1 - truncation).clip(min=1e-4)
            bboxes_aera = np.prod(bboxes[:, 2:] - bboxes[:, :2], axis=1)
            ignore_inds = \
                bboxes_aera < (1 - trunc_ignore_thres) * bboxes_ori_area
            ignore_inds = valid_inds & ignore_inds
            # disgard ignore from valid gt
            valid_inds = valid_inds & np.logical_not(ignore_inds)
            if 'gt_bboxes_ignore' in results:
                results['gt_bboxes_ignore'] = np.concatenate(
                    (results['gt_bboxes_ignore'],
                     bboxes[ignore_inds]), axis=0)
            if 'gt_labels_ignore' in results:
                results['gt_labels_ignore'] = np.concatenate(
                    (results['gt_labels_ignore'],
                     results['gt_labels'][ignore_inds]), axis=0)
            if 'gt_masks_ignore' in results:
                results['gt_masks_ignore'] += \
                    results['gt_masks'][ignore_inds.nonzero()[0]]
            if 'gt_bboxes_3d_ignore' in results:
                results['gt_bboxes_3d_ignore'] = np.concatenate(
                    (results['gt_bboxes_3d_ignore'],
                     results['gt_bboxes_3d'][ignore_inds]), axis=0)

        results[key] = bboxes[valid_inds, :]
        # label fields. e.g. gt_labels and gt_labels_ignore
        label_key = bbox2label.get(key)
        if label_key in results:
            results[label_key] = results[label_key][valid_inds]
        # mask fields, e.g. gt_masks and gt_masks_ignore
        mask_key = bbox2mask.get(key)
        if mask_key in results:
            results[mask_key] = results[mask_key][
                valid_inds.nonzero()[0]]
        # bbox_3d fields
        bbox_3d_key = bbox2bbox_3d.get(key)
        if bbox_3d_key in results:
            results[bbox_3d_key] = results[bbox_3d_key][valid_inds]
        if key == 'gt_bboxes':
            for misc_key in ['gt_x2d', 'gt_x3d', 'truncation', 'gt_attr', 'gt_velo']:
                if misc_key in results:
                    if isinstance(results[misc_key], list):
                        results[misc_key] = [
                            results[misc_key][i]
                            for i in np.flatnonzero(valid_inds)]
                    else:
                        results[misc_key] = results[misc_key][valid_inds]

    # crop semantic seg
    for key in results.get('seg_fields', []):
        results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]

    for key in results.get('dense_fields', []):
        dense = results[key]
        if isinstance(dense, list):
            results[key] = [
                dense_single[crop_y1:crop_y2, crop_x1:crop_x2]
                for dense_single in dense]
        else:
            results[key] = dense[crop_y1:crop_y2, crop_x1:crop_x2]

    return results


@PIPELINES.register_module()
class Crop3D(object):
    def __init__(self, crop_box,
                 trunc_ignore_thres=0.7,
                 allow_negative_crop=False,
                 bbox_clip_border=True):
        self.crop_box = crop_box
        self.trunc_ignore_thres = trunc_ignore_thres
        self.allow_negative_crop = allow_negative_crop
        self.bbox_clip_border = bbox_clip_border
        self.bbox2label = {
            'gt_bboxes': 'gt_labels',
            'gt_bboxes_ignore': 'gt_labels_ignore'
        }
        self.bbox2mask = {
            'gt_bboxes': 'gt_masks',
            'gt_bboxes_ignore': 'gt_masks_ignore'
        }
        self.bbox2bbox_3d = {
            'gt_bboxes': 'gt_bboxes_3d',
            'gt_bboxes_ignore': 'gt_bboxes_3d_ignore'
        }

    def __call__(self, results):
        return crop_3d(results, self.crop_box,
                       self.bbox2mask, self.bbox2label, self.bbox2bbox_3d,
                       allow_negative_crop=self.allow_negative_crop,
                       trunc_ignore_thres=self.trunc_ignore_thres)


@PIPELINES.register_module()
class RandomCrop3D(RandomCrop):
    """
    Note:
        - Truncated bboxes can be ignored, along with the associated
          labels and masks.
    """
    def __init__(self, crop_size, trunc_ignore_thres=-1, **kwargs):
        super(RandomCrop3D, self).__init__(crop_size, **kwargs)
        self.trunc_ignore_thres = trunc_ignore_thres
        self.bbox2bbox_3d = {
            'gt_bboxes': 'gt_bboxes_3d',
            'gt_bboxes_ignore': 'gt_bboxes_3d_ignore'
        }

    def _crop_data(self, results, crop_size, allow_negative_crop):
        assert crop_size[0] > 0 and crop_size[1] > 0
        img = results['img']
        margin_h = max(img.shape[0] - crop_size[0], 0)
        margin_w = max(img.shape[1] - crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]
        crop_box = (crop_x1, crop_y1, crop_x2, crop_y2)
        return crop_3d(results, crop_box,
                       self.bbox2mask, self.bbox2label, self.bbox2bbox_3d,
                       allow_negative_crop=self.allow_negative_crop,
                       trunc_ignore_thres=self.trunc_ignore_thres)


@PIPELINES.register_module()
class MinIoURandomCrop3D(MinIoURandomCrop):

    def __call__(self, results):
        if 'img_fields' in results:
            assert results['img_fields'] == ['img'], \
                'Only single img_fields is allowed'
        img = results['img']
        assert 'bbox_fields' in results
        boxes = [results[key] for key in results['bbox_fields']]
        boxes = np.concatenate(boxes, 0)
        h, w, c = img.shape
        while True:
            mode = random.choice(self.sample_mode)
            self.mode = mode
            if mode == 1:
                return results

            min_iou = mode
            for i in range(50):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(w - new_w)
                top = random.uniform(h - new_h)

                patch = np.array(
                    (int(left), int(top), int(left + new_w), int(top + new_h)))
                # Line or point crop is not allowed
                if patch[2] == patch[0] or patch[3] == patch[1]:
                    continue
                overlaps = bbox_overlaps(
                    patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)
                if len(overlaps) > 0 and overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                # only adjust boxes and instance masks when the gt is not empty
                if len(overlaps) > 0:
                    # adjust boxes
                    def is_center_of_bboxes_in_patch(boxes, patch):
                        center = (boxes[:, :2] + boxes[:, 2:]) / 2
                        mask = ((center[:, 0] > patch[0]) *
                                (center[:, 1] > patch[1]) *
                                (center[:, 0] < patch[2]) *
                                (center[:, 1] < patch[3]))
                        return mask

                    mask = is_center_of_bboxes_in_patch(boxes, patch)
                    if not mask.any():
                        continue
                    for key in results.get('bbox_fields', []):
                        boxes = results[key].copy()
                        mask = is_center_of_bboxes_in_patch(boxes, patch)
                        boxes = boxes[mask]
                        boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                        boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                        boxes -= np.tile(patch[:2], 2)

                        results[key] = boxes
                        # labels
                        label_key = self.bbox2label.get(key)
                        if label_key in results:
                            results[label_key] = results[label_key][mask]

                        # mask fields
                        mask_key = self.bbox2mask.get(key)
                        if mask_key in results:
                            results[mask_key] = results[mask_key][
                                mask.nonzero()[0]].crop(patch)
                # adjust the img no matter whether the gt is empty before crop
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                results['img'] = img
                results['img_shape'] = img.shape

                # seg fields
                for key in results.get('seg_fields', []):
                    results[key] = results[key][patch[1]:patch[3],
                                                patch[0]:patch[2]]

                for key in results.get('dense_fields', []):
                    dense = results[key]
                    if isinstance(dense, list):
                        results[key] = [
                            dense_single[patch[1]:patch[3], patch[0]:patch[2]]
                            for dense_single in dense]
                    else:
                        results[key] = dense[patch[1]:patch[3],
                                             patch[0]:patch[2]]
                return results
