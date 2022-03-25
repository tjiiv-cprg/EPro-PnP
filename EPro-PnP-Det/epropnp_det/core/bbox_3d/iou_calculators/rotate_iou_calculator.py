"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
This file is modified from
https://github.com/tjiiv-cprg/MonoRUn
"""

import numpy as np
from numba import cuda
import torch
import ctypes

from .rotate_iou_kernel import (div_up, rotate_iou_kernel_eval,
                                rotate_iou_kernel_eval_aligned)


def bbox_rotate_overlaps(boxes, query_boxes, criterion=-1, device_id=0):
    """rotated box iou running in gpu. 500x faster than cpu version
    (take 5ms in one example with numba.cuda code).
    convert from [this project](
        https://github.com/hongzhenwang/RRPN-revise/tree/master/lib/rotation).

    Args:
        boxes (float tensor: [n, 5]): rbboxes. format: centers, dims,
            angles(clockwise when positive)
        query_boxes (float tensor: [k, 5]): [description]
        criterion:
        device_id (int, optional): Defaults to 0. [description]

    Returns:
        [type]: [description]
    """
    boxes = boxes.astype(np.float32)
    query_boxes = query_boxes.astype(np.float32)
    n = boxes.shape[0]
    k = query_boxes.shape[0]
    iou = np.zeros((n, k), dtype=np.float32)
    if n == 0 or k == 0:
        return iou
    threads_per_block = 8 * 8
    cuda.select_device(device_id)
    blockspergrid = (div_up(n, threads_per_block), div_up(k, threads_per_block))

    stream = cuda.stream()
    with stream.auto_synchronize():
        boxes_dev = cuda.to_device(boxes.reshape([-1]), stream)
        query_boxes_dev = cuda.to_device(query_boxes.reshape([-1]), stream)
        iou_dev = cuda.to_device(iou.reshape([-1]), stream)
        rotate_iou_kernel_eval[blockspergrid, threads_per_block, stream](
            n, k, boxes_dev, query_boxes_dev, iou_dev, criterion)
        iou_dev.copy_to_host(iou.reshape([-1]), stream=stream)
    return iou.astype(boxes.dtype)


def bbox_rotate_overlaps_aligned(boxes, qboxes, criterion=-1, device_id=0):
    n = boxes.shape[0]
    iou = np.empty(n, dtype=boxes.dtype)
    if n == 0:
        return iou
    threads_per_block = 64
    blockspergrid = div_up(n, threads_per_block)
    cuda.select_device(device_id)

    stream = cuda.stream()
    with stream.auto_synchronize():
        boxes_dev = cuda.to_device(boxes.reshape([-1]), stream)
        qboxes_dev = cuda.to_device(qboxes.reshape([-1]), stream)
        iou_dev = cuda.to_device(iou, stream)
        rotate_iou_kernel_eval_aligned[blockspergrid, threads_per_block, stream](
            n, boxes_dev, qboxes_dev, iou_dev, criterion)
        iou_dev.copy_to_host(iou.reshape([-1]), stream=stream)
    return iou


def get_devicendarray(t):
    assert t.type() == 'torch.cuda.FloatTensor'
    ctx = cuda.devices.get_context()
    mp = cuda.cudadrv.driver.MemoryPointer(ctx, ctypes.c_ulong(t.data_ptr()), t.numel() * 4)
    return cuda.cudadrv.devicearray.DeviceNDArray(
        t.size(), [i * 4 for i in t.stride()], np.dtype('float32'),
        gpu_data=mp, stream=torch.cuda.current_stream().cuda_stream)


def bbox_rotate_overlaps_aligned_torch(boxes, qboxes, criterion=-1):
    n = boxes.shape[0]
    iou = boxes.new_empty([n])
    if n == 0:
        return iou
    threads_per_block = 64
    blockspergrid = div_up(n, threads_per_block)

    boxes_dev = get_devicendarray(boxes.flatten())
    qboxes_dev = get_devicendarray(qboxes.flatten())
    iou_dev = get_devicendarray(iou)
    rotate_iou_kernel_eval_aligned[blockspergrid, threads_per_block](
        n, boxes_dev, qboxes_dev, iou_dev, criterion)
    cuda.synchronize()

    return iou
