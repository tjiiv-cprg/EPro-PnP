"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
This file is modified from
https://github.com/tjiiv-cprg/MonoRUn
"""

import numpy as np
import torch
import numba

from .rotate_iou_calculator import (
    bbox_rotate_overlaps, bbox_rotate_overlaps_aligned,
    bbox_rotate_overlaps_aligned_torch)


@numba.jit(nopython=True)
def bev_to_box3d_overlaps(boxes,
                          qboxes,
                          rinc,
                          criterion=-1,
                          z_axis=1,
                          z_center=0.5):
    """
        z_axis: the z (height) axis.
        z_center: unified z (height) center of box.
    """
    n, k = boxes.shape[0], qboxes.shape[0]
    for i in range(n):
        for j in range(k):
            if rinc[i, j] > 0:
                min_z = min(
                    boxes[i, z_axis] + boxes[i, z_axis + 3] * (1 - z_center),
                    qboxes[j, z_axis] + qboxes[j, z_axis + 3] * (1 - z_center))
                max_z = max(
                    boxes[i, z_axis] - boxes[i, z_axis + 3] * z_center,
                    qboxes[j, z_axis] - qboxes[j, z_axis + 3] * z_center)
                iw = min_z - max_z
                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = 1.0
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0


def bbox3d_overlaps(boxes, qboxes, criterion=-1, z_axis=1, z_center=0.5):
    """
    Args:
        boxes (ndarray): (N, 7), locations [x, y, z], dimensions [l, h, w],
            rot_y
        qboxes (ndarray):  (K, 7), locations [x, y, z], dimensions [l, h, w],
            rot_y
        criterion: -1 for iou
        z_axis: 1 for kitti camera format
        z_center:

    Returns:
        ndarray: (N, K), ious
    """
    bev_axes = list(range(7))
    bev_axes.pop(z_axis + 3)
    bev_axes.pop(z_axis)
    rinc = bbox_rotate_overlaps(boxes[:, bev_axes], qboxes[:, bev_axes], 2)
    bev_to_box3d_overlaps(boxes, qboxes, rinc, criterion, z_axis, z_center)
    return rinc


def bev_to_box3d_overlaps_aligned(boxes,
                                  qboxes,
                                  rinc,
                                  criterion=-1,
                                  z_axis=1,
                                  z_center=0.5):
    if boxes.shape[0] == 0:
        return np.zeros(0, dtype=boxes.dtype)
    # min_height = y + h * (1 - z_center) DOWNWARD positive!
    min_z = np.minimum(
        boxes[:, z_axis] + boxes[:, z_axis + 3] * (1 - z_center),
        qboxes[:, z_axis] + qboxes[:, z_axis + 3] * (1 - z_center))
    # max_height = y + h * z_center
    max_z = np.maximum(
        boxes[:, z_axis] - boxes[:, z_axis + 3] * z_center,
        qboxes[:, z_axis] - qboxes[:, z_axis + 3] * z_center)
    # note that y axis direction is downwards
    # iw: height of the intersection volume
    iw = np.maximum(min_z - max_z, 0)
    volumn1 = np.prod(boxes[:, 3:6], axis=1)
    volumn2 = np.prod(qboxes[:, 3:6], axis=1)
    inc = iw * rinc
    if criterion == -1:
        ua = (volumn1 + volumn2 - inc)
    elif criterion == 0:
        ua = volumn1
    elif criterion == 1:
        ua = volumn2
    else:
        ua = 1.0
    iou = inc / ua.clip(min=1e-6)
    return iou.clip(min=0., max=1.)


def bbox3d_overlaps_aligned(boxes, qboxes, criterion=-1, z_axis=1, z_center=0.5):
    """
    Args:
        boxes (ndarray): (N, 7), locations [x, y, z], dimensions [l, h, w],
            rot_y
        qboxes (ndarray):  (N, 7), locations [x, y, z], dimensions [l, h, w],
            rot_y
        criterion: -1 for iou
        z_axis:
        z_center: 1.0 for bottom origin and 0.0 for top origin

    Returns:
        ndarray: (N, 1), ious
    """
    bev_axes = list(range(7))
    bev_axes.pop(z_axis + 3)
    bev_axes.pop(z_axis)
    # get rotated bev bboxes intersection
    # criterion=2: returns area of 2d bev intersection
    rinc = bbox_rotate_overlaps_aligned(
        boxes[:, bev_axes], qboxes[:, bev_axes], criterion=2)
    # get 3d iou
    iou = bev_to_box3d_overlaps_aligned(
        boxes, qboxes, rinc, criterion, z_axis, z_center)
    return iou


def bev_to_box3d_overlaps_aligned_torch(boxes,
                                        qboxes,
                                        rinc,
                                        criterion=-1,
                                        z_axis=1,
                                        z_center=0.5):
    if boxes.shape[0] == 0:
        return boxes.new_zeros(size=(0,))
    # min_height = y + h * (1 - z_center) DOWNWARD positive!
    min_z = torch.min(
        boxes[:, z_axis] + boxes[:, z_axis + 3] * (1 - z_center),
        qboxes[:, z_axis] + qboxes[:, z_axis + 3] * (1 - z_center))
    # max_height = y + h * z_center
    max_z = torch.min(
        boxes[:, z_axis] - boxes[:, z_axis + 3] * z_center,
        qboxes[:, z_axis] - qboxes[:, z_axis + 3] * z_center)
    # note that y axis direction is downwards
    # iw: height of the intersection volume
    iw = (min_z - max_z).clamp(min=0)
    volumn1 = torch.prod(boxes[:, 3:6], dim=1)
    volumn2 = torch.prod(qboxes[:, 3:6], dim=1)
    inc = iw * rinc
    if criterion == -1:
        ua = (volumn1 + volumn2 - inc)
    elif criterion == 0:
        ua = volumn1
    elif criterion == 1:
        ua = volumn2
    else:
        ua = 1.0
    iou = inc / ua.clamp(min=1e-6)
    return iou.clamp(min=0., max=1.)


def bbox3d_overlaps_aligned_torch(
        boxes, qboxes, criterion=-1, z_axis=1, z_center=0.5):
    """
    Args:
        boxes (Tensor): (N, 7), locations [x, y, z], dimensions [l, h, w],
            rot_y
        qboxes (Tensor):  (N, 7), locations [x, y, z], dimensions [l, h, w],
            rot_y
        criterion: -1 for iou
        z_axis:
        z_center: 1.0 for bottom origin and 0.0 for top origin

    Returns:
        Tensor: (N, 1), ious
    """
    bev_axes = list(range(7))
    bev_axes.pop(z_axis + 3)
    bev_axes.pop(z_axis)
    with torch.no_grad():
        # get rotated bev bboxes intersection
        # criterion=2: returns area of 2d bev intersection
        rinc = bbox_rotate_overlaps_aligned_torch(
            boxes[:, bev_axes], qboxes[:, bev_axes], criterion=2)
        # get 3d iou
        iou = bev_to_box3d_overlaps_aligned_torch(
            boxes, qboxes, rinc, criterion, z_axis, z_center)
    return iou
