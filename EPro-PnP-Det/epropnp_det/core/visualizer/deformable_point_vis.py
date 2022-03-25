"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
"""

import numpy as np
import cv2

from .. import gen_unit_noc, yaw_to_rot_mat


def draw_rectangles(img, rectangles, colors, transparencies, sort_idx=None, inplace=False):
    """
    Args:
        img (np.ndarray): Shape (h, w, 3)
        rectangles (np.ndarray): Shape (num_boxes, 4) in [x1, y1, x2, y2]
        colors (np.ndarray): Shape (num_boxes, 3)
        transparencies (np.ndarray): Shape (num_boxes, )
    """
    if not inplace:
        img = img.copy()
    h, w, _ = img.shape
    np.clip(rectangles[..., 0::2], 0, w, out=rectangles[..., 0::2])
    np.clip(rectangles[..., 1::2], 0, h, out=rectangles[..., 1::2])
    
    if sort_idx is None:
        sort_idx = range(rectangles.shape[0])
    for i in sort_idx:
        img[rectangles[i, 1]:rectangles[i, 3], rectangles[i, 0]:rectangles[i, 2], :] = \
            img[rectangles[i, 1]:rectangles[i, 3], rectangles[i, 0]:rectangles[i, 2], :] * transparencies[i] \
            + colors[i, :] * (1 - transparencies[i])
    return img


def deformable_point_vis(ori_img, result, score_thr, num_head, img_contrast=0.8, point_size=0.04,
                         point_size_power=0.8, deepen=2.0, x_color=(10, 40, 250), y_color=(10, 250, 10),
                         eps=1e-6, scale=1.0):
    bbox_3d = np.concatenate(result['bbox_3d_results'], axis=0)
    score_mask = bbox_3d[:, 7] >= score_thr

    bbox_3d = bbox_3d[score_mask]
    bbox_2d = np.concatenate(result['bbox_results'], axis=0)[score_mask]
    x2d = np.concatenate(result['x2d'], axis=0)[score_mask]
    x3d = np.concatenate(result['x3d'], axis=0)[score_mask]
    w2d = np.concatenate(result['w2d'], axis=0)[score_mask]

    if scale != 1.0:
        ori_img = cv2.resize(ori_img, (0, 0), fx=scale, fy=scale)
        x2d *= scale

    base = ori_img.astype(np.float32) * img_contrast + 255 * (1 - img_contrast) / 2

    num_obj, num_pts, _ = x2d.shape
    head_pts = num_pts // num_head

    weight = w2d / w2d.mean(axis=1, keepdims=True).clip(min=1e-6)  # (num_obj, num_pts, 2) normalized weights
    transparency = np.exp(-weight.mean(axis=-1) * deepen)  # (num_obj, num_pts)

    bbox_area = np.prod(bbox_2d[:, 2:4] - bbox_2d[:, 0:2], axis=1)
    point_size = np.round((np.sqrt(bbox_area) * point_size) ** point_size_power * scale + 1)
    point_radius = point_size[:, None, None] / 2
    rectangles = np.concatenate(
        (np.round(x2d - point_radius).astype(np.int64),
         np.round(x2d + point_radius).astype(np.int64)), axis=-1)

    z = (yaw_to_rot_mat(bbox_3d[:, 6])[:, None, 2:3, :] @ x3d[..., None]
         ).squeeze(-1).squeeze(-1) + bbox_3d[:, 5:6]  # (num_obj, num_pts)
    sort_idx = np.argsort(z.flatten())[::-1]

    xy_relative_weight = weight / weight.sum(axis=-1, keepdims=True).clip(min=eps)
    xy_color = xy_relative_weight @ np.array((x_color, y_color), dtype=np.float32)  # (num_obj, num_pts, 3)
    obj_color = 256 / (1 + np.exp(np.random.normal(loc=-0.5, size=(num_obj, 3)).astype(np.float32)))  # (num_obj, 3)
    obj_color = np.broadcast_to(obj_color[:, None, :], xy_color.shape)
    head_color = (gen_unit_noc(num_head).numpy() / 2 + 0.5) * 256  # (num_head, 3)
    head_color = np.broadcast_to(head_color[:, None, ::-1], (num_obj, num_head, head_pts, 3))  # to BGR

    rectangles = rectangles.reshape(num_obj * num_pts, 4)
    obj_color = obj_color.reshape(num_obj * num_pts, 3)
    head_color = head_color.reshape(num_obj * num_pts, 3)
    xy_color = xy_color.reshape(num_obj * num_pts, 3)
    transparency = transparency.flatten()

    pts_obj = draw_rectangles(
        base, rectangles, obj_color, transparency, sort_idx
    ).clip(min=0, max=255).astype(np.uint8)
    pts_head = draw_rectangles(
        base, rectangles, head_color, transparency, sort_idx
    ).clip(min=0, max=255).astype(np.uint8)
    pts_xy = draw_rectangles(
        base, rectangles, xy_color, transparency, sort_idx, inplace=True
    ).clip(min=0, max=255).astype(np.uint8)

    return pts_obj, pts_head, pts_xy
